"""Concurrent window load driver with origin accounting.

The driver submits requests from a deterministic trace at a target
concurrency, samples public /metrics between windows, and computes
per-window tier-hit deltas.

Concurrent window origin classification
---------------------------------------
At a window start, ``prefill_device_hit_tokens`` can include prefixes
that were *restored from host while queued* — the scheduler admitted the
request only after its prefix state was brought back to GPU.  A naive
device-hit reading would therefore mislabel a host-restored window as
device-resident.  The classification therefore uses:

    total_tier_hits = device_hit + host_hit + storage_hit deltas
    host_origin = max(host_hit, min(load_back_tokens_delta, total_tier_hits))
    device_origin = total_tier_hits - host_origin

``load_back_tokens_total`` is capped at the tier-hit volume so a single
token cannot be double-counted as both a host restore and a device hit.

Effective concurrency and queue depth are tracked separately.  Queueing
is not called preemption; only the runtime's retracted-request counter is
used as active-request preemption evidence.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from hcv.http_client import SGLangHTTPClient
from hcv.metrics import CacheStats, CacheStatsDelta, diff_snapshots, snapshot_from_scrape
from hcv.workload import Trace, TraceRecord

logger = logging.getLogger(__name__)

ORIGIN_DEVICE = "device_resident"
ORIGIN_HOST = "host_restored"
ORIGIN_MIXED = "mixed"
ORIGIN_UNKNOWN = "unknown"


@dataclass
class WindowStats:
    """One measurement window's aggregate statistics."""

    window_index: int
    start_monotonic: float
    end_monotonic: float
    requests_started: int = 0
    requests_completed: int = 0
    tier_hits_total: Optional[float] = None
    device_hit_delta: Optional[float] = None
    host_hit_delta: Optional[float] = None
    storage_hit_delta: Optional[float] = None
    load_back_delta: Optional[float] = None
    capped_load_back: Optional[float] = None
    origin: str = ORIGIN_UNKNOWN
    max_queue_reqs: Optional[float] = None
    max_running_reqs: Optional[float] = None
    effective_concurrency: Optional[float] = None
    retracted_requests_delta: Optional[float] = None
    ttft_sum_ms: float = 0.0
    ttft_count: int = 0

    def to_dict(self) -> dict:
        return {
            "window_index": self.window_index,
            "start_monotonic": round(self.start_monotonic, 3),
            "end_monotonic": round(self.end_monotonic, 3),
            "requests_started": self.requests_started,
            "requests_completed": self.requests_completed,
            "tier_hits_total": self.tier_hits_total,
            "device_hit_delta": self.device_hit_delta,
            "host_hit_delta": self.host_hit_delta,
            "storage_hit_delta": self.storage_hit_delta,
            "load_back_delta": self.load_back_delta,
            "capped_load_back": self.capped_load_back,
            "origin": self.origin,
            "max_queue_reqs": self.max_queue_reqs,
            "max_running_reqs": self.max_running_reqs,
            "effective_concurrency": self.effective_concurrency,
            "retracted_requests_delta": self.retracted_requests_delta,
            "ttft_sum_ms": round(self.ttft_sum_ms, 3),
            "ttft_count": self.ttft_count,
        }


def classify_window_origin(
    device_hit_delta: Optional[float],
    host_hit_delta: Optional[float],
    storage_hit_delta: Optional[float],
    load_back_delta: Optional[float],
    tier_hits_total: Optional[float],
    host_share_threshold: float = 0.5,
) -> tuple[str, Optional[float]]:
    """Classify a window's start origin; returns (origin, capped_load_back).

    Uses total tier hits plus *capped* ``load_back_tokens_total`` (see
    module docstring) so admission-time device hits that actually came
    from host restores are not mislabeled as device-resident.
    """
    if tier_hits_total is None or tier_hits_total <= 0:
        return ORIGIN_UNKNOWN, None
    if device_hit_delta is None or host_hit_delta is None or load_back_delta is None:
        return ORIGIN_UNKNOWN, None
    capped_load = min(load_back_delta, tier_hits_total)
    device = device_hit_delta
    host = host_hit_delta
    storage = storage_hit_delta if storage_hit_delta is not None else 0.0
    total = device + host + storage
    if total <= 0:
        return ORIGIN_UNKNOWN, capped_load
    host_origin = max(host, capped_load)
    host_share = host_origin / total
    if host_share >= host_share_threshold:
        return ORIGIN_HOST, capped_load
    if host_share > 0.0:
        return ORIGIN_MIXED, capped_load
    return ORIGIN_DEVICE, capped_load


@dataclass
class LoadRunResult:
    """Aggregate result of a concurrent load run."""

    windows: list = field(default_factory=list)
    requests: list = field(default_factory=list)   # per-request results
    preemption_windows: int = 0
    max_queue_reqs: Optional[float] = None
    concurrency_drift_windows: int = 0
    gen_errors: list = field(default_factory=list)
    start_snapshot: dict = field(default_factory=dict)
    end_snapshot: dict = field(default_factory=dict)
    total_delta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "windows": [w.to_dict() for w in self.windows],
            "requests": self.requests,
            "preemption_windows": self.preemption_windows,
            "max_queue_reqs": self.max_queue_reqs,
            "concurrency_drift_windows": self.concurrency_drift_windows,
            "gen_errors": self.gen_errors,
            "start_snapshot": self.start_snapshot,
            "end_snapshot": self.end_snapshot,
            "total_delta": self.total_delta,
        }


def run_load(
    client: SGLangHTTPClient,
    trace: Trace,
    records: list[TraceRecord],
    concurrency: int,
    request_id_offset: int = 0,
    window_requests: int = 64,
    max_windows: int = 200,
    concurrency_tolerance: float = 0.5,
) -> LoadRunResult:
    """Run the given records at fixed concurrency with window sampling.

    ``records`` may be a slice of the trace (e.g. the formal phase after
    warm-up).  A fixed worker pool issues the records in deterministic
    trace order; the main thread samples public /metrics at window
    boundaries (every ``window_requests`` completions) and computes
    per-window tier-hit deltas against the previous boundary snapshot.
    """
    import threading

    result = LoadRunResult()
    start_stats: CacheStats = snapshot_from_scrape(client.scrape_metrics())
    result.start_snapshot = _snapshot_dict(start_stats)

    # --- worker pool -------------------------------------------------------
    next_index = 0
    lock = threading.Lock()
    results_by_id: dict[int, dict] = {}
    completion_event = threading.Event()
    completed = [0]
    gen_errors: list[dict] = []

    def worker() -> None:
        nonlocal next_index
        while True:
            with lock:
                idx = next_index
                next_index += 1
            if idx >= len(records):
                return
            rec = records[idx]
            rid = request_id_offset + idx
            t_send = time.time()
            gen = client.generate(
                request_id=rid, input_ids=rec.input_ids, max_new_tokens=rec.output_tokens
            )
            gen.t_send = t_send
            d = gen.to_dict()
            d["trace_idx"] = rec.idx
            d["family_id"] = rec.family_id
            d["is_revisit"] = rec.is_revisit
            d["revisit_of_idx"] = rec.revisit_of_idx
            d["reuse_distance"] = rec.reuse_distance
            d["prefix_tokens"] = rec.prefix_tokens
            d["suffix_tokens"] = rec.suffix_tokens
            d["prefix_key"] = rec.prefix_key
            with lock:
                results_by_id[rid] = d
                if not gen.ok:
                    gen_errors.append({"request_id": rid, "error": gen.error, "status": gen.status})
                completed[0] += 1
            if completed[0] >= len(records):
                completion_event.set()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, concurrency))]
    for t in threads:
        t.start()

    # --- window sampling ---------------------------------------------------
    window_start_mono = time.monotonic()
    window_start_stats = start_stats
    window_idx = 0
    completed_at_boundary = 0
    max_queue: Optional[float] = None

    # Keep the loop alive through the terminal boundary.  A worker can set
    # ``completion_event`` between loop-condition checks; exiting on
    # ``completed == len(records)`` would then discard the final (possibly
    # partial) measurement window.
    while True:
        now = time.monotonic()
        if completed[0] >= completed_at_boundary + window_requests or completion_event.is_set():
            end_stats: CacheStats = snapshot_from_scrape(client.scrape_metrics())
            delta: CacheStatsDelta = diff_snapshots(end_stats, window_start_stats)
            tier = (
                delta.get("prefill_device_hit_tokens"),
                delta.get("prefill_host_hit_tokens"),
                delta.get("prefill_storage_hit_tokens"),
            )
            tier_total = (
                sum(t for t in tier if t is not None)
                if any(t is not None for t in tier)
                else None
            )
            origin, capped = classify_window_origin(
                tier[0], tier[1], tier[2],
                delta.get("load_back_tokens_total"), tier_total,
            )
            duration = max(now - window_start_mono, 1e-6)
            with lock:
                done_ids = set(results_by_id)
            previous_done_ids = locals().get("previous_done_ids", set())
            window_ids = sorted(done_ids - previous_done_ids)
            ttft_sum = 0.0
            ttft_count = 0
            for rid in window_ids:
                d = results_by_id.get(rid)
                if d and d.get("ok"):
                    ttft_sum += float(d.get("ttft_ms", 0.0))
                    ttft_count += 1
            # Little's law estimate of effective concurrency in this window.
            eff_conc = None
            if ttft_count > 0:
                eff_conc = (ttft_sum / 1000.0) / duration
            w = WindowStats(
                window_index=window_idx,
                start_monotonic=window_start_mono,
                end_monotonic=now,
                requests_started=len(window_ids),
                requests_completed=len(window_ids),
                tier_hits_total=tier_total,
                device_hit_delta=tier[0],
                host_hit_delta=tier[1],
                storage_hit_delta=tier[2],
                load_back_delta=delta.get("load_back_tokens_total"),
                capped_load_back=capped,
                origin=origin,
                max_queue_reqs=end_stats.num_queue_reqs,
                max_running_reqs=end_stats.num_running_reqs,
                effective_concurrency=eff_conc,
                retracted_requests_delta=delta.get("num_retracted_requests_total"),
                ttft_sum_ms=ttft_sum,
                ttft_count=ttft_count,
            )
            if (delta.get("num_retracted_requests_total") or 0.0) > 0:
                result.preemption_windows += 1
            if eff_conc is not None and abs(eff_conc - concurrency) > concurrency_tolerance:
                result.concurrency_drift_windows += 1
            result.windows.append(w)
            window_idx += 1
            window_start_mono = now
            window_start_stats = end_stats
            completed_at_boundary = completed[0]
            previous_done_ids = done_ids
            if max_queue is None or (end_stats.num_queue_reqs or 0.0) > max_queue:
                max_queue = end_stats.num_queue_reqs
            if completion_event.is_set():
                break
            if window_idx >= max_windows:
                break
        time.sleep(0.05)

    for t in threads:
        t.join(timeout=600)

    with lock:
        result.requests = [results_by_id[rid] for rid in sorted(results_by_id)]
    result.gen_errors = gen_errors
    result.max_queue_reqs = max_queue
    end_stats = snapshot_from_scrape(client.scrape_metrics())
    result.end_snapshot = _snapshot_dict(end_stats)
    result.total_delta = diff_snapshots(end_stats, start_stats).to_dict()
    return result


def _snapshot_dict(stats: CacheStats) -> dict:
    return {
        "prefill_input_tokens": stats.prefill_input_tokens,
        "prefill_device_hit_tokens": stats.prefill_device_hit_tokens,
        "prefill_host_hit_tokens": stats.prefill_host_hit_tokens,
        "prefill_storage_hit_tokens": stats.prefill_storage_hit_tokens,
        "load_back_tokens_total": stats.load_back_tokens_total,
        "hicache_backup_tokens_total": stats.hicache_backup_tokens_total,
        "kv_available_tokens": stats.kv_available_tokens,
        "kv_evictable_tokens": stats.kv_evictable_tokens,
        "kv_used_tokens": stats.kv_used_tokens,
        "max_total_num_tokens": stats.max_total_num_tokens,
        "num_requests_total": stats.num_requests_total,
        "num_running_reqs": stats.num_running_reqs,
        "num_queue_reqs": stats.num_queue_reqs,
        "num_retracted_requests_total": stats.num_retracted_requests_total,
    }
