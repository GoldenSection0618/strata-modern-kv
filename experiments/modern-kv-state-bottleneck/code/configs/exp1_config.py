"""Experiment 1: Context Length Scaling — configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Exp1Config:
    """All parameters for a single Experiment 1 run.

    One run = one model × one context length × one residency mode.
    """

    # --- Model ---
    model_id: str = "Qwen/Qwen3.5-9B"
    model_path: str = ""  # set to local snapshot path at runtime

    # --- Context sweep ---
    context_lengths: list[int] = field(
        default_factory=lambda: [4096, 8192, 16384, 32768]
    )

    # --- Workload ---
    prefix_ratio: float = 0.5
    max_output_tokens: int = 1  # TTFT only
    base_text_path: str = ""  # corpus file; if empty, use built-in fallback

    # --- Residency ---
    residency_modes: list[str] = field(
        default_factory=lambda: ["recompute", "gpu_hit", "cpu_hit"]
    )

    # --- Repetition ---
    n_warmup: int = 3
    n_repeats: int = 10

    # --- vLLM engine ---
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 65536
    block_size: int = 16
    kv_cache_metrics: bool = True
    dtype: str = "bfloat16"
    # CPU-resident hit tuning
    kv_offloading_size: int | None = None  # None = let vLLM auto-decide
    cpu_offload_gb: float = 0.0

    # --- GPU monitoring ---
    gpu_monitor_interval_ms: int = 10

    # --- Output ---
    output_dir: str = "results/exp1"

    # --- Hardware ---
    gpu_id: int = 0

    # --- Metadata ---
    experiment: str = "exp1-context-length-scaling"
    runtime_version: str = "vllm-0.26.0"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    def save(self, path: str | Path):
        Path(path).write_text(self.to_json(), encoding="utf-8")
