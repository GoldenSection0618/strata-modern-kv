"""Experiment 2: Shared-Prefix Scaling — configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Exp2Config:
    """All parameters for a single Experiment 2 run.

    One run = one model × one context length × one prefix ratio × one residency mode.
    """

    # --- Model ---
    model_id: str = "Qwen/Qwen3.5-9B"
    model_path: str = ""

    # --- Fixed context length ---
    context_length: int = 32768
    aux_context_length: int = 16384

    # --- Prefix ratio sweep ---
    prefix_ratios: list[float] = field(
        default_factory=lambda: [0.0, 0.25, 0.50, 0.75, 0.875]
    )

    # --- Workload ---
    max_output_tokens: int = 1
    base_text_path: str = ""

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
    kv_offloading_size: int | None = None
    cpu_offload_gb: float = 0.0

    # --- GPU monitoring ---
    gpu_monitor_interval_ms: int = 10

    # --- Output ---
    output_dir: str = "results/exp2"

    # --- Hardware ---
    gpu_id: int = 0

    # --- Metadata ---
    experiment: str = "exp2-shared-prefix-scaling"
    runtime_version: str = "vllm-0.26.0"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    def save(self, path: str | Path):
        Path(path).write_text(self.to_json(), encoding="utf-8")
