"""Public SGLang Prometheus metrics: parser, typed snapshot, deltas.

This module is pure Python (stdlib only).  It parses the Prometheus text
format served by SGLang's ``/metrics`` endpoint (enabled with
``--enable-metrics``) and turns it into a typed cache-stat snapshot.

Missing-metric rule
-------------------
A metric that is absent from the scrape is recorded as ``None``, never
as a silent zero.  ``0.0`` is only recorded when the metric is actually
present in the scrape with value 0.  Deltas between two snapshots are
``None`` whenever either side is missing, so an unsupported metric can
never be misreported as a zero delta.

Metric inventory (verified against SGLang commit
``4ad990ba7d75bb9f948f5f6bd8d79a66b5d3fd63``, ``metrics_collector.py``):

* ``sglang:prefill_effective_tokens_total`` Counter, label ``mode`` in
  {``input``, ``device_hit``, ``host_hit``, ``storage_hit``} — per-tier
  effective prefill tokens (host_hit is the L2 restore evidence).
* ``sglang:load_back_tokens_total`` Counter, label ``pool`` — tokens
  loaded back from host DRAM (L2) to GPU (H->D restore evidence).
* ``sglang:hicache_backup_tokens_total`` Counter, label ``pool`` —
  tokens backed up from GPU to host DRAM (L2) (D->H write evidence).
* ``sglang:hicache_backup_bytes_total`` / ``sglang:load_back_bytes_total``
  Counters — bytes moved D->H / H->D (bandwidth numerator).
* ``sglang:hicache_host_used_tokens`` / ``sglang:hicache_host_total_tokens``
  Gauges — host (L2) pool occupancy and capacity in tokens.
* ``sglang:cached_tokens_total`` Counter, label ``cache_source`` in
  {``device``, ``host``, ``storage``} — cumulative cached prompt tokens
  by tier.
* ``sglang:cache_hit_rate`` Gauge — windowed prefix cache hit rate.
* ``sglang:kv_used_tokens`` / ``kv_available_tokens`` / ``kv_evictable_tokens``
  Gauges — full-attention KV pool absolute token counts.
* ``sglang:max_total_num_tokens`` Gauge — KV pool capacity (startup const).
* ``sglang:num_running_reqs`` / ``sglang:num_queue_reqs`` Gauges.
* ``sglang:num_requests_total`` Counter.
* ``sglang:gen_throughput`` Gauge.
* ``sglang:time_to_first_token_seconds`` Histogram (sum/count).
* ``sglang:page_size`` / ``sglang:num_pages`` Gauges (startup consts).

Samples carry labels such as ``model_name``, ``engine_type``, ``tp_rank``,
``pp_rank``, ``moe_ep_rank``.  All label selectors used here are
tier/scope labels (``mode``, ``pool``, ``cache_source``); other labels
are ignored so the parser is robust to label-set changes.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

# ---------------------------------------------------------------------------
# Prometheus text-format parser
# ---------------------------------------------------------------------------

_LABEL_ESCAPES = {
    '\\"': '"',
    "\\\\": "\\",
    "\\n": "\n",
}

_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{.*\})?\s+(?P<value>[^\s]+)\s*$"
)


def _parse_label_value(raw: str) -> str:
    """Unescape a quoted Prometheus label value."""
    out: list[str] = []
    i = 0
    while i < len(raw):
        ch = raw[i]
        if ch == "\\" and i + 1 < len(raw):
            pair = raw[i : i + 2]
            if pair in _LABEL_ESCAPES:
                out.append(_LABEL_ESCAPES[pair])
                i += 2
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _parse_labels(spec: str) -> dict[str, str]:
    """Parse a ``{name="value", ...}`` label set into a dict."""
    if not spec or spec == "{}":
        return {}
    inner = spec[1:-1]
    labels: dict[str, str] = {}
    # Split on commas that are outside quotes.
    parts: list[str] = []
    buf: list[str] = []
    in_quotes = False
    i = 0
    while i < len(inner):
        ch = inner[i]
        if ch == '"':
            in_quotes = not in_quotes
            buf.append(ch)
        elif ch == "\\" and in_quotes and i + 1 < len(inner):
            buf.append(ch)
            buf.append(inner[i + 1])
            i += 1
        elif ch == "," and not in_quotes:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    parts.append("".join(buf))
    for part in parts:
        part = part.strip()
        if not part:
            continue
        key, _, value = part.partition("=")
        key = key.strip()
        value = value.strip()
        if value.startswith('"') and value.endswith('"'):
            value = _parse_label_value(value[1:-1])
        labels[key] = value
    return labels


def _parse_value(raw: str) -> float:
    """Parse a Prometheus sample value (int/float/Inf/NaN)."""
    raw = raw.strip()
    if raw in ("+Inf", "Inf", "inf"):
        return float("inf")
    if raw == "-Inf":
        return float("-inf")
    if raw in ("NaN", "nan"):
        return float("nan")
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"Invalid Prometheus sample value: {raw!r}") from e


@dataclass
class Sample:
    """One Prometheus sample: label set + numeric value."""

    labels: dict[str, str]
    value: float


@dataclass
class HistogramSample:
    """Aggregated histogram (sum/count) for one family."""

    sum: Optional[float] = None
    count: Optional[float] = None


@dataclass
class PrometheusScrape:
    """Parsed scrape of SGLang's /metrics endpoint."""

    samples: dict[str, list[Sample]] = field(default_factory=dict)
    histograms: dict[str, HistogramSample] = field(default_factory=dict)
    raw_text: str = ""

    def has(self, name: str) -> bool:
        return name in self.samples or name in self.histograms


def parse_prometheus_text(text: str) -> PrometheusScrape:
    """Parse Prometheus text format into a :class:`PrometheusScrape`.

    Histogram families contribute ``<name>_bucket``/``<name>_sum``/
    ``<name>_count`` samples; the parser folds ``_sum`` and ``_count``
    into :attr:`histograms` and keeps everything else in :attr:`samples`.
    """
    scrape = PrometheusScrape(raw_text=text)
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE_RE.match(line)
        if not m:
            # Unknown line shape: skip rather than fail the whole scrape.
            continue
        name = m.group("name")
        labels = _parse_labels(m.group("labels") or "{}")
        value = _parse_value(m.group("value"))

        if name.endswith("_sum") or name.endswith("_count"):
            base = name[: -len("_sum")] if name.endswith("_sum") else name[: -len("_count")]
            hist = scrape.histograms.setdefault(base, HistogramSample())
            if name.endswith("_sum"):
                hist.sum = value
            else:
                hist.count = value
        elif "_bucket" in name:
            # Bucket lines are not needed for the sums/counts we consume.
            continue
        else:
            scrape.samples.setdefault(name, []).append(Sample(labels=labels, value=value))
    return scrape


def select_values(
    scrape: PrometheusScrape, name: str, **label_filters: str
) -> list[float]:
    """Return values of all samples of ``name`` matching label filters.

    A filter matches when the sample's label set contains the given
    key/value pair.  Multiple matches are returned in scrape order;
    callers that need a single value should use :func:`metric_value`.
    """
    matches: list[float] = []
    for sample in scrape.samples.get(name, []):
        ok = all(sample.labels.get(k) == v for k, v in label_filters.items())
        if ok:
            matches.append(sample.value)
    return matches


def metric_value(
    scrape: PrometheusScrape, name: str, **label_filters: str
) -> Optional[float]:
    """Return a single metric value or ``None`` when absent.

    If multiple samples match, returns the maximum.  For counters this
    is conservative (a scrape should already be aggregated by the
    multiprocess collector); for ``mostrecent`` gauges it is exact.
    """
    values = select_values(scrape, name, **label_filters)
    if not values:
        return None
    return max(values)


def histogram_value(
    scrape: PrometheusScrape, base_name: str, field: str = "count"
) -> Optional[float]:
    """Return histogram ``sum`` or ``count``, or ``None`` when absent."""
    hist = scrape.histograms.get(base_name)
    if hist is None:
        return None
    return getattr(hist, field)


# ---------------------------------------------------------------------------
# Typed cache-stat snapshot
# ---------------------------------------------------------------------------

#: Metric families that carry the per-tier ``mode`` label.
_PREFILL_EFFECTIVE = "sglang:prefill_effective_tokens_total"


@dataclass
class CacheStats:
    """Typed snapshot of public SGLang cache metrics.

    Every field is ``None`` when the underlying metric is absent from
    the scrape (missing ≠ zero).  ``raw`` carries the parsed scrape for
    auditability.
    """

    timestamp: float = 0.0

    # Effective prefill tokens by tier (Counter deltas since server start).
    prefill_input_tokens: Optional[float] = None
    prefill_device_hit_tokens: Optional[float] = None
    prefill_host_hit_tokens: Optional[float] = None
    prefill_storage_hit_tokens: Optional[float] = None

    # Host (L2) transfer counters (Counter, cumulative).
    load_back_tokens_total: Optional[float] = None       # H -> D restore
    load_back_bytes_total: Optional[float] = None        # H -> D restore bytes
    hicache_backup_tokens_total: Optional[float] = None  # D -> H write
    hicache_backup_bytes_total: Optional[float] = None   # D -> H write bytes

    # Host (L2) pool gauges (snapshot).
    hicache_host_used_tokens: Optional[float] = None
    hicache_host_total_tokens: Optional[float] = None

    # Cumulative cached prompt tokens by tier (Counter, cache_source label).
    cached_tokens_total: Optional[float] = None
    cached_tokens_device: Optional[float] = None
    cached_tokens_host: Optional[float] = None
    cached_tokens_storage: Optional[float] = None

    # Windowed hit rate + KV pool gauges.
    cache_hit_rate: Optional[float] = None
    kv_used_tokens: Optional[float] = None
    kv_available_tokens: Optional[float] = None
    kv_evictable_tokens: Optional[float] = None
    max_total_num_tokens: Optional[float] = None
    page_size: Optional[float] = None

    # Scheduler / load gauges.
    num_running_reqs: Optional[float] = None
    num_queue_reqs: Optional[float] = None
    num_requests_total: Optional[float] = None
    gen_throughput: Optional[float] = None

    # TTFT histogram aggregates (Counter-like sum/count).
    ttft_seconds_sum: Optional[float] = None
    ttft_seconds_count: Optional[float] = None

    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize, keeping ``raw`` under its own key."""
        d = asdict(self)
        return d

    @property
    def kv_free_tokens(self) -> Optional[float]:
        """Approximate free (unallocated) KV pool slots in tokens."""
        if self.kv_available_tokens is None:
            return None
        return self.kv_available_tokens + (self.kv_evictable_tokens or 0.0)


def snapshot_from_scrape(scrape: PrometheusScrape, timestamp: float | None = None) -> CacheStats:
    """Build a :class:`CacheStats` from a parsed scrape.

    Tier metrics with missing ``mode``/``pool``/``cache_source`` labels
    are summed across all label values of that scope.
    """
    ts = time.time() if timestamp is None else timestamp

    def _tier(mode: str) -> Optional[float]:
        return metric_value(scrape, _PREFILL_EFFECTIVE, mode=mode)

    def _sum_family(name: str) -> Optional[float]:
        values = select_values(scrape, name)
        if not values:
            return None
        return sum(values)

    ttft_base = "sglang:time_to_first_token_seconds"

    return CacheStats(
        timestamp=ts,
        prefill_input_tokens=_tier("input"),
        prefill_device_hit_tokens=_tier("device_hit"),
        prefill_host_hit_tokens=_tier("host_hit"),
        prefill_storage_hit_tokens=_tier("storage_hit"),
        load_back_tokens_total=_sum_family("sglang:load_back_tokens_total"),
        load_back_bytes_total=metric_value(scrape, "sglang:load_back_bytes_total"),
        hicache_backup_tokens_total=_sum_family("sglang:hicache_backup_tokens_total"),
        hicache_backup_bytes_total=metric_value(scrape, "sglang:hicache_backup_bytes_total"),
        hicache_host_used_tokens=metric_value(scrape, "sglang:hicache_host_used_tokens"),
        hicache_host_total_tokens=metric_value(scrape, "sglang:hicache_host_total_tokens"),
        cached_tokens_total=_sum_family("sglang:cached_tokens_total"),
        cached_tokens_device=metric_value(scrape, "sglang:cached_tokens_total", cache_source="device"),
        cached_tokens_host=metric_value(scrape, "sglang:cached_tokens_total", cache_source="host"),
        cached_tokens_storage=metric_value(scrape, "sglang:cached_tokens_total", cache_source="storage"),
        cache_hit_rate=metric_value(scrape, "sglang:cache_hit_rate"),
        kv_used_tokens=metric_value(scrape, "sglang:kv_used_tokens"),
        kv_available_tokens=metric_value(scrape, "sglang:kv_available_tokens"),
        kv_evictable_tokens=metric_value(scrape, "sglang:kv_evictable_tokens"),
        max_total_num_tokens=metric_value(scrape, "sglang:max_total_num_tokens"),
        page_size=metric_value(scrape, "sglang:page_size"),
        num_running_reqs=metric_value(scrape, "sglang:num_running_reqs"),
        num_queue_reqs=metric_value(scrape, "sglang:num_queue_reqs"),
        num_requests_total=metric_value(scrape, "sglang:num_requests_total"),
        gen_throughput=metric_value(scrape, "sglang:gen_throughput"),
        ttft_seconds_sum=histogram_value(scrape, ttft_base, "sum"),
        ttft_seconds_count=histogram_value(scrape, ttft_base, "count"),
        raw={
            "families": {
                name: [{"labels": s.labels, "value": s.value} for s in samples]
                for name, samples in scrape.samples.items()
            },
            "histograms": {
                name: {"sum": h.sum, "count": h.count}
                for name, h in scrape.histograms.items()
            },
        },
    )


# ---------------------------------------------------------------------------
# Before/after deltas
# ---------------------------------------------------------------------------

#: Snapshot fields that are absolute gauges; their delta is the signed
#: change (e.g. host pool occupancy going up when data is backed up).
_GAUGE_FIELDS = {
    "hicache_host_used_tokens",
    "hicache_host_total_tokens",
    "cache_hit_rate",
    "kv_used_tokens",
    "kv_available_tokens",
    "kv_evictable_tokens",
    "max_total_num_tokens",
    "page_size",
    "num_running_reqs",
    "num_queue_reqs",
    "gen_throughput",
}

#: Snapshot fields that are cumulative counters; their delta is the
#: increment over the measurement window.
_COUNTER_FIELDS = {
    "prefill_input_tokens",
    "prefill_device_hit_tokens",
    "prefill_host_hit_tokens",
    "prefill_storage_hit_tokens",
    "load_back_tokens_total",
    "load_back_bytes_total",
    "hicache_backup_tokens_total",
    "hicache_backup_bytes_total",
    "cached_tokens_total",
    "cached_tokens_device",
    "cached_tokens_host",
    "cached_tokens_storage",
    "num_requests_total",
    "ttft_seconds_sum",
    "ttft_seconds_count",
}

_SNAPSHOT_FIELDS = tuple(_GAUGE_FIELDS | _COUNTER_FIELDS)


def _sub(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Subtract two optional values; None if either side is missing."""
    if a is None or b is None:
        return None
    return a - b


@dataclass
class CacheStatsDelta:
    """Before/after deltas of :class:`CacheStats` fields.

    ``None`` means the metric was missing on at least one side and the
    delta is unsupported — never a silent zero.
    """

    timestamp: float = 0.0
    deltas: dict[str, Optional[float]] = field(default_factory=dict)

    def get(self, field: str) -> Optional[float]:
        return self.deltas.get(field)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "deltas": {k: v for k, v in self.deltas.items()},
        }


def diff_snapshots(after: CacheStats, before: CacheStats) -> CacheStatsDelta:
    """Compute per-field deltas (after - before) with None propagation."""
    deltas: dict[str, Optional[float]] = {}
    for f in _SNAPSHOT_FIELDS:
        deltas[f] = _sub(getattr(after, f), getattr(before, f))
    return CacheStatsDelta(timestamp=after.timestamp, deltas=deltas)


def fields_with_evidence(delta: CacheStatsDelta, threshold: float = 0.0) -> list[str]:
    """Return counter fields whose delta is strictly positive.

    Used by validation to summarize which cache tiers showed activity.
    """
    return [
        f
        for f in _COUNTER_FIELDS
        if delta.get(f) is not None and delta.get(f) > threshold
    ]


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def cache_stats_to_json(stats: CacheStats) -> dict:
    """Serialize a snapshot to a JSON-safe dict (raw families included)."""
    d = stats.to_dict()
    return d


def json_dumps_compact(obj) -> str:
    """Deterministic compact JSON for test comparisons."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))
