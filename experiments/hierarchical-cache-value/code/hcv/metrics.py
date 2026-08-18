"""Public SGLang Prometheus metrics: parser, typed snapshot, deltas.

Pure Python (stdlib only).  Parses the Prometheus text format served by
SGLang's public ``/metrics`` endpoint (enabled with ``--enable-metrics``)
and turns it into a typed cache-stat snapshot.

Missing-metric rule
-------------------
A metric absent from a scrape is recorded as ``None``, never as a silent
zero.  ``0.0`` is recorded only when the metric is actually present with
value 0.  Deltas between two snapshots are ``None`` whenever either side
is missing, so an unsupported metric can never be misreported as a zero
delta.

Metric inventory (public metrics observed on the pinned build,
``sglang-hicache-cu129-torch211`` @ ``4ad990ba7d75bb9f948f5f6bd8d79a66b5d3fd63``,
``metrics_collector.py``):

* ``sglang:prefill_effective_tokens_total`` Counter, label ``mode`` in
  {``input``, ``device_hit``, ``host_hit``, ``storage_hit``} — per-tier
  effective prefill tokens (``host_hit`` is the L2 restore evidence;
  ``input`` is the recomputation evidence).
* ``sglang:load_back_tokens_total`` Counter, label ``pool`` — tokens
  loaded back from host DRAM to GPU (H->D restore evidence).  When the
  runtime exposes distinguishable pools they are kept separately for
  per-state-group evidence.
* ``sglang:hicache_backup_tokens_total`` Counter, label ``pool`` —
  tokens backed up from GPU to host DRAM (D->H write evidence).
* ``sglang:hicache_backup_bytes_total`` / ``sglang:load_back_bytes_total``
  Counters — bytes moved D->H / H->D.
* ``sglang:hicache_host_used_tokens`` / ``sglang:hicache_host_total_tokens``
  Gauges — host pool occupancy and capacity in tokens.
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

Samples may carry additional labels (``model_name``, ``engine_type``,
``tp_rank``, ...).  All label selectors used here are tier/scope labels
(``mode``, ``pool``, ``cache_source``); other labels are ignored so the
parser is robust to label-set changes.
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

    Histogram families contribute ``<name>_bucket`` / ``<name>_sum`` /
    ``<name>_count`` samples; ``_sum`` and ``_count`` are folded into
    :attr:`histograms`; ``_bucket`` lines are skipped; everything else
    stays in :attr:`samples`.
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
            base = (
                name[: -len("_sum")]
                if name.endswith("_sum")
                else name[: -len("_count")]
            )
            hist = scrape.histograms.setdefault(base, HistogramSample())
            if name.endswith("_sum"):
                hist.sum = value
            else:
                hist.count = value
        elif "_bucket" in name:
            continue
        else:
            scrape.samples.setdefault(name, []).append(Sample(labels=labels, value=value))
    return scrape


def select_values(
    scrape: PrometheusScrape, name: str, **label_filters: str
) -> list[float]:
    """Return values of all samples of ``name`` matching label filters.

    A filter matches when the sample's label set contains the given
    key/value pair.  Multiple matches are returned in scrape order.
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

    If multiple samples match, returns the maximum (conservative for
    already-aggregated counters, exact for ``mostrecent`` gauges).
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

_PREFILL_EFFECTIVE = "sglang:prefill_effective_tokens_total"


@dataclass
class CacheStats:
    """Typed snapshot of public SGLang cache metrics.

    Every field is ``None`` when the underlying metric is absent from
    the scrape (missing != zero).  ``raw`` carries the parsed scrape for
    auditability.  ``pool_deltas_*`` fields carry per-``pool`` label
    breakdowns when the runtime exposes them (state-group evidence).
    """

    timestamp: float = 0.0

    # Effective prefill tokens by tier (Counter deltas since server start).
    prefill_input_tokens: Optional[float] = None
    prefill_device_hit_tokens: Optional[float] = None
    prefill_host_hit_tokens: Optional[float] = None
    prefill_storage_hit_tokens: Optional[float] = None

    # Host transfer counters (Counter, cumulative).
    load_back_tokens_total: Optional[float] = None       # H -> D restore
    load_back_bytes_total: Optional[float] = None        # H -> D restore bytes
    hicache_backup_tokens_total: Optional[float] = None  # D -> H write
    hicache_backup_bytes_total: Optional[float] = None   # D -> H write bytes

    # Per-pool breakdowns (state-group evidence; {} when unobservable).
    load_back_tokens_by_pool: dict = field(default_factory=dict)
    hicache_backup_tokens_by_pool: dict = field(default_factory=dict)

    # Host pool gauges (snapshot).
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
    num_pages: Optional[float] = None

    # Scheduler / load gauges.
    num_running_reqs: Optional[float] = None
    num_queue_reqs: Optional[float] = None
    num_requests_total: Optional[float] = None
    num_retracted_requests_total: Optional[float] = None
    gen_throughput: Optional[float] = None

    # TTFT histogram aggregates (Counter-like sum/count).
    ttft_seconds_sum: Optional[float] = None
    ttft_seconds_count: Optional[float] = None

    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize, keeping ``raw`` under its own key."""
        return asdict(self)

    @property
    def kv_free_tokens(self) -> Optional[float]:
        """Approximate free (unallocated) KV pool slots in tokens."""
        if self.kv_available_tokens is None:
            return None
        return self.kv_available_tokens + (self.kv_evictable_tokens or 0.0)

    def tier_hits_total(self) -> Optional[float]:
        """Total tier hit tokens (device + host + storage) if observable."""
        parts = [
            self.prefill_device_hit_tokens,
            self.prefill_host_hit_tokens,
            self.prefill_storage_hit_tokens,
        ]
        if any(p is None for p in parts):
            return None
        return sum(p for p in parts if p is not None)  # type: ignore[arg-type]


def snapshot_from_scrape(
    scrape: PrometheusScrape, timestamp: float | None = None
) -> CacheStats:
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

    def _by_pool(name: str) -> dict[str, float]:
        """Map pool-label -> summed value for a family."""
        out: dict[str, float] = {}
        for sample in scrape.samples.get(name, []):
            pool = sample.labels.get("pool")
            if pool is None:
                continue
            out[pool] = out.get(pool, 0.0) + sample.value
        return dict(sorted(out.items()))

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
        load_back_tokens_by_pool=_by_pool("sglang:load_back_tokens_total"),
        hicache_backup_tokens_by_pool=_by_pool("sglang:hicache_backup_tokens_total"),
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
        num_pages=metric_value(scrape, "sglang:num_pages"),
        num_running_reqs=metric_value(scrape, "sglang:num_running_reqs"),
        num_queue_reqs=metric_value(scrape, "sglang:num_queue_reqs"),
        num_requests_total=metric_value(scrape, "sglang:num_requests_total"),
        num_retracted_requests_total=_sum_family("sglang:num_retracted_requests_total"),
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

#: Snapshot fields that are absolute gauges; their delta is the signed change.
_GAUGE_FIELDS = {
    "hicache_host_used_tokens",
    "hicache_host_total_tokens",
    "cache_hit_rate",
    "kv_used_tokens",
    "kv_available_tokens",
    "kv_evictable_tokens",
    "max_total_num_tokens",
    "page_size",
    "num_pages",
    "num_running_reqs",
    "num_queue_reqs",
    "gen_throughput",
}

#: Snapshot fields that are cumulative counters; their delta is the increment.
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
    "num_retracted_requests_total",
    "ttft_seconds_sum",
    "ttft_seconds_count",
}

_SNAPSHOT_FIELDS = tuple(sorted(_GAUGE_FIELDS | _COUNTER_FIELDS))


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
    pool_deltas: dict = field(default_factory=dict)  # {"load_back_tokens_total": {pool: delta}}

    def get(self, field: str) -> Optional[float]:
        return self.deltas.get(field)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "deltas": {k: v for k, v in self.deltas.items()},
            "pool_deltas": self.pool_deltas,
        }


def _diff_pool_dicts(after: dict, before: dict) -> dict[str, float]:
    """Per-pool delta (after - before); missing pools contribute 0."""
    pools = sorted(set(after) | set(before))
    out: dict[str, float] = {}
    for p in pools:
        a = after.get(p, 0.0)
        b = before.get(p, 0.0)
        out[p] = a - b
    return out


def diff_snapshots(after: CacheStats, before: CacheStats) -> CacheStatsDelta:
    """Compute per-field deltas (after - before) with None propagation."""
    deltas: dict[str, Optional[float]] = {}
    for f in _SNAPSHOT_FIELDS:
        deltas[f] = _sub(getattr(after, f), getattr(before, f))
    pool_deltas = {
        "load_back_tokens_total": _diff_pool_dicts(
            after.load_back_tokens_by_pool, before.load_back_tokens_by_pool
        ),
        "hicache_backup_tokens_total": _diff_pool_dicts(
            after.hicache_backup_tokens_by_pool, before.hicache_backup_tokens_by_pool
        ),
    }
    return CacheStatsDelta(timestamp=after.timestamp, deltas=deltas, pool_deltas=pool_deltas)


def fields_with_evidence(delta: CacheStatsDelta, threshold: float = 0.0) -> list[str]:
    """Return counter fields whose delta is strictly positive."""
    return [
        f
        for f in _COUNTER_FIELDS
        if delta.get(f) is not None and delta.get(f) > threshold
    ]


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def json_dumps_compact(obj) -> str:
    """Deterministic compact JSON for test comparisons."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))
