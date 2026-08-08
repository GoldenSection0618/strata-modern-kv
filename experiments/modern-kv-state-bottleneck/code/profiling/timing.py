"""TTFT measurement and decomposition helpers."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class TimingResult:
    """Single request timing measurement."""

    t_send: float           # perf_counter at request dispatch
    t_first_token: float    # perf_counter at first token received
    ttft_ms: float          # (t_first_token - t_send) * 1000
    gpu_samples: list       # GPU memory time series during request
    kv_stats_before: dict   # vLLM stats before request
    kv_stats_after: dict    # vLLM stats after request
    gpu_analysis: dict      # Analyzed GPU memory data

    def to_dict(self) -> dict:
        d = asdict(self)
        # Convert tuple lists for JSON serialization
        d["gpu_samples"] = [[t, b] for t, b in self.gpu_samples]
        return d


def measure_ttft(
    llm,
    prompt: dict,
    sampling_params,
    gpu_monitor,
    stats_collector,
) -> TimingResult:
    """Send a request and measure TTFT with full instrumentation.

    Args:
        llm: vLLM LLM instance
        prompt: TokensPrompt dict {"prompt_token_ids": [...]}
        sampling_params: vLLM SamplingParams
        gpu_monitor: GPUMonitor instance (will be started/stopped)
        stats_collector: VLLMStatsCollector instance

    Returns:
        TimingResult with all collected data
    """
    from .gpu_monitor import GPUMonitor as _GM  # avoid circular import

    # Snapshot before
    kv_before = stats_collector.collect().to_dict()

    # Start monitoring
    gpu_monitor.start()
    t_send = time.perf_counter()

    # Send request
    outputs = llm.generate([prompt], sampling_params)

    t_first_token = time.perf_counter()
    gpu_samples = gpu_monitor.stop()

    # Snapshot after
    kv_after = stats_collector.collect().to_dict()

    # Analyze GPU samples
    gpu_analysis = _GM.analyze_samples(
        gpu_samples, t_send, t_first_token
    )

    ttft_ms = (t_first_token - t_send) * 1000

    return TimingResult(
        t_send=t_send,
        t_first_token=t_first_token,
        ttft_ms=round(ttft_ms, 3),
        gpu_samples=gpu_samples,
        kv_stats_before=kv_before,
        kv_stats_after=kv_after,
        gpu_analysis=gpu_analysis,
    )


def decompose_ttft(
    ttft_recompute: float,
    ttft_gpu_hit: float,
    ttft_cpu_hit: float | None,
) -> dict:
    """Decompose TTFT into compute, io_stall, and reuse components.

    All inputs in milliseconds.  Under low load (queueing ≈ 0).

    Returns dict with derived metrics in ms.
    """
    result = {
        "compute_path_ms": round(ttft_recompute, 3),
        "reuse_saving_ms": round(ttft_recompute - ttft_gpu_hit, 3),
        "io_stall_ms": None,
        "cpu_restore_penalty_ms": None,
    }

    if ttft_cpu_hit is not None:
        result["io_stall_ms"] = round(ttft_cpu_hit - ttft_gpu_hit, 3)
        result["cpu_restore_penalty_ms"] = round(ttft_cpu_hit - ttft_gpu_hit, 3)

    return result
