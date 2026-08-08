"""Experiment 3: Request-Rate Scaling — configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Exp3Config:
    """All parameters for a single Experiment 3 run.

    One run = one model × one residency mode × one offered load point.
    """

    # --- Model ---
    model_id: str = "Qwen/Qwen3.5-9B"
    model_path: str = ""

    # --- Fixed workload ---
    context_length: int = 32768
    prefix_ratio: float = 0.5          # 16K shared prefix / 16K unique suffix
    max_output_tokens: int = 1         # TTFT only
    base_text_path: str = ""

    # --- Residency ---
    residency_mode: str = "cpu_hit"    # primary mode

    # --- Load sweep ---
    # Offered rates (req/s) are determined by calibration and frozen
    # before the formal sweep.  This list is populated at runtime.
    offered_rates: list[float] = field(default_factory=list)

    # --- Concurrency ---
    # High enough ceiling to avoid artificial capping in normal region
    concurrency_ceiling: int = 64

    # --- Per-load-point repetition ---
    n_warmup: int = 5          # warm-up requests before measurement
    n_measured: int = 30       # measured requests per load point
    observation_window_s: float = 0.0  # 0 = use n_measured, not time window

    # --- vLLM engine ---
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 65536
    block_size: int = 16
    kv_cache_metrics: bool = True
    dtype: str = "bfloat16"
    kv_offloading_size: int | None = None
    cpu_offload_gb: float = 0.0

    # --- GPU monitoring ---
    gpu_monitor_interval_ms: int = 10

    # --- Output ---
    output_dir: str = "results/exp3"

    # --- Hardware ---
    gpu_id: int = 0

    # --- Calibration ---
    calibration_rates: list[float] = field(
        default_factory=lambda: [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0]
    )
    calibration_duration_s: float = 20.0

    # --- Metadata ---
    experiment: str = "exp3-request-rate-scaling"
    runtime_version: str = "vllm-0.26.0"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    def save(self, path: str | Path):
        Path(path).write_text(self.to_json(), encoding="utf-8")
