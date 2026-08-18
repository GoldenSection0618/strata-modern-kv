"""Serial probe execution with isolated before/after per-tier counters.

Each probe isolates one request between two /metrics snapshots so the
delta is attributable to that single request.  This is authoritative
when request metadata (``cached_tokens_details``) is missing from the
pinned Qwen hybrid native ``/generate``.

Probe expectations (serial):
* ``recompute``: input-path delta >= prefix length, device hit == 0,
  host hit == 0 (cold miss recomputes).
* ``gpu_hit``: device-hit delta >= prefix length, host hit == 0,
  input-path delta < prefix length (prefix served from GPU).
* ``gpu_only_eviction`` (negative control, GPU-only server): after L1
  eviction, revisit shows device hit == 0, host hit == 0, input-path
  delta >= prefix length (state lost -> recompute).
* ``cpu_hit`` (hierarchical server): after L1 eviction, revisit shows
  host-hit delta >= prefix length, device hit == 0, input-path delta
  < prefix length (no recompute of the prefix), and load-back delta
  >= prefix length (restore traffic).  A host hit combined with
  input-path growth of ~prefix length is a silent restore-to-recompute
  fallback and fails the probe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from hcv.hierarchy import (
    PROBE_CPU_HIT,
    PROBE_GPU_HIT,
    PROBE_GPU_ONLY_EVICTION,
    PROBE_RECOMPUTE,
    ProbeResult,
)
from hcv.http_client import SGLangHTTPClient
from hcv.metrics import CacheStatsDelta, diff_snapshots, snapshot_from_scrape


@dataclass
class ProbeSpec:
    """What to send for one probe."""

    name: str
    input_ids: list[int]
    prefix_length: int
    request_id: int


def _delta(d: CacheStatsDelta, field: str) -> Optional[float]:
    return d.get(field)


def evaluate_probe(name: str, probe: ProbeResult, prefix_length: int) -> ProbeResult:
    """Check a serial probe against its expectations (pure, testable)."""
    inp = probe.input_delta
    dev = probe.device_hit_delta
    host = probe.host_hit_delta
    lb = probe.load_back_delta

    reasons: list[str] = []
    def require_zero(label: str, value: Optional[float]) -> None:
        if value is None:
            reasons.append(f"{label} delta missing")
        elif value != 0:
            reasons.append(f"{label} delta {value} != 0")
    if name == PROBE_RECOMPUTE:
        if inp is None or inp < prefix_length:
            reasons.append(f"input delta {inp} < prefix length {prefix_length}")
        require_zero("device hit", dev)
        require_zero("host hit", host)
    elif name == PROBE_GPU_HIT:
        if dev is None or dev < prefix_length:
            reasons.append(f"device hit delta {dev} < prefix length {prefix_length}")
        require_zero("host hit", host)
        if inp is None:
            reasons.append("input delta missing")
        elif inp >= prefix_length:
            reasons.append(f"input delta {inp} >= prefix length: prefix recomputed")
    elif name == PROBE_GPU_ONLY_EVICTION:
        require_zero("device hit", dev)
        require_zero("host hit", host)
        if inp is None or inp < prefix_length:
            reasons.append(f"input delta {inp} < prefix length: eviction did not force recompute")
    elif name == PROBE_CPU_HIT:
        if host is None or host < prefix_length:
            reasons.append(f"host hit delta {host} < prefix length {prefix_length}")
        require_zero("device hit", dev)
        if inp is None:
            reasons.append("input delta missing")
        elif inp >= prefix_length:
            reasons.append(
                f"input delta {inp} >= prefix length: silent restore-to-recompute fallback"
            )
        if lb is None:
            reasons.append("load_back delta missing")
        elif lb < prefix_length:
            reasons.append(f"load_back delta {lb} < prefix length: restore traffic missing")
    else:
        reasons.append(f"unknown probe name {name!r}")

    probe.ok = not reasons
    probe.reason = "; ".join(reasons) if reasons else "ok"
    return probe


def run_serial_probe(
    client: SGLangHTTPClient,
    spec: ProbeSpec,
    evaluate: bool = True,
) -> ProbeResult:
    """Run one serial probe with isolated before/after counters."""
    before = snapshot_from_scrape(client.scrape_metrics())
    gen = client.generate(
        request_id=spec.request_id,
        input_ids=spec.input_ids,
        max_new_tokens=1,
    )
    after = snapshot_from_scrape(client.scrape_metrics())
    delta = diff_snapshots(after, before)
    pool_load_back = delta.pool_deltas.get("load_back_tokens_total", {})
    aggregate_load_back = _delta(delta, "load_back_tokens_total")
    # The pinned hybrid runtime exposes H->D restores per pool (kv/mamba)
    # but may omit the unlabeled aggregate.  Preserve the public-counter
    # evidence by summing observable pools; do not manufacture zero when
    # neither representation exists.
    effective_load_back = (
        aggregate_load_back
        if aggregate_load_back is not None
        else (sum(pool_load_back.values()) if pool_load_back else None)
    )

    probe = ProbeResult(
        name=spec.name,
        prefix_length=spec.prefix_length,
        ok=gen.ok,
        reason="" if gen.ok else f"generate failed: {gen.error}",
        deltas=delta.to_dict(),
        input_delta=_delta(delta, "prefill_input_tokens"),
        device_hit_delta=_delta(delta, "prefill_device_hit_tokens"),
        host_hit_delta=_delta(delta, "prefill_host_hit_tokens"),
        load_back_delta=effective_load_back,
        pool_deltas=pool_load_back,
    )
    if evaluate:
        probe = evaluate_probe(spec.name, probe, spec.prefix_length)
    return probe
