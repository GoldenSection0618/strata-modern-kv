"""SGLang server configuration and authoritative server argument building.

Residency -> server flags (verified against SGLang server_args.py at
commit ``4ad990ba7d75bb9f948f5f6bd8d79a66b5d3fd63``):

* ``recompute`` — ``--disable-radix-cache`` (radix cache off, no HiCache)
* ``gpu_hit``   — radix cache enabled (default), HiCache disabled
* ``cpu_hit``   — ``--enable-hierarchical-cache`` plus the pinned
  ``--hicache-*`` flags (L1 GPU + L2 host)

Recorded metadata (required by the measurement constraints):
``hicache_io_backend``, ``hicache_mem_layout``, page size, write policy,
host cache size/ratio, and the SGLang commit.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

RESIDENCY_MODES = ("recompute", "gpu_hit", "cpu_hit")

DEFAULT_SGLANG_COMMIT = "4ad990ba7d75bb9f948f5f6bd8d79a66b5d3fd63"

HICACHE_WRITE_POLICIES = ("write_back", "write_through", "write_through_selective")
HICACHE_IO_BACKENDS = ("direct", "kernel")
HICACHE_MEM_LAYOUTS = ("layer_first", "page_first", "page_first_direct")


@dataclass
class SGLangServerConfig:
    """All server-side parameters for one SGLang run."""

    # --- model ---
    model_path: str
    model_id: str

    # --- residency ---
    residency_mode: str = "cpu_hit"

    # --- network ---
    host: str = "127.0.0.1"
    port: int = 0  # 0 -> OS-assigned free port

    # --- memory / scheduling ---
    mem_fraction_static: float = 0.85
    page_size: int = 64
    tp_size: int = 1
    max_running_requests: Optional[int] = None
    dtype: str = "bfloat16"

    # --- HiCache (cpu_hit only; recorded even when unused) ---
    hicache_ratio: float = 2.0
    hicache_size_gb: int = 0  # 0 -> use hicache_ratio
    hicache_io_backend: str = "kernel"
    hicache_mem_layout: str = "page_first"
    hicache_write_policy: str = "write_through"

    # --- provenance ---
    sglang_commit: Optional[str] = None  # resolved at runtime

    extra_args: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.residency_mode not in RESIDENCY_MODES:
            raise ValueError(
                f"residency_mode must be one of {RESIDENCY_MODES}, got "
                f"{self.residency_mode!r}"
            )
        if self.hicache_write_policy not in HICACHE_WRITE_POLICIES:
            raise ValueError(f"invalid hicache_write_policy: {self.hicache_write_policy}")
        if self.hicache_io_backend not in HICACHE_IO_BACKENDS:
            raise ValueError(f"invalid hicache_io_backend: {self.hicache_io_backend}")
        if self.hicache_mem_layout not in HICACHE_MEM_LAYOUTS:
            raise ValueError(f"invalid hicache_mem_layout: {self.hicache_mem_layout}")
        if self.port == 0:
            from sglang_hicache.server_lifecycle import pick_free_port

            self.port = pick_free_port()

    # -- server args ---------------------------------------------------------

    def build_argv(self) -> list[str]:
        """Authoritative ``launch_server`` argument list for this config."""
        args = [
            "--model-path", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--dtype", self.dtype,
            "--tp", str(self.tp_size),
            "--mem-fraction-static", str(self.mem_fraction_static),
            "--page-size", str(self.page_size),
            "--enable-metrics",
            "--enable-cache-report",
        ]
        if self.max_running_requests is not None:
            args += ["--max-running-requests", str(self.max_running_requests)]

        if self.residency_mode == "recompute":
            args.append("--disable-radix-cache")
        elif self.residency_mode == "cpu_hit":
            args.append("--enable-hierarchical-cache")
            if self.hicache_size_gb and self.hicache_size_gb > 0:
                args += ["--hicache-size", str(self.hicache_size_gb)]
            else:
                args += ["--hicache-ratio", str(self.hicache_ratio)]
            args += [
                "--hicache-io-backend", self.hicache_io_backend,
                "--hicache-mem-layout", self.hicache_mem_layout,
                "--hicache-write-policy", self.hicache_write_policy,
            ]

        args += self.extra_args
        return args

    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    # -- metadata ------------------------------------------------------------

    def resolve_commit(self) -> Optional[str]:
        """Resolve the pinned SGLang commit from explicit sources only.

        Order: explicit value -> env SGLANG_COMMIT -> bootstrap-written
        ``sglang_commit.txt`` in the env dir -> default constant (marked
        unverified when not otherwise confirmed).
        """
        if self.sglang_commit:
            return self.sglang_commit
        env_val = os.environ.get("SGLANG_COMMIT")
        if env_val:
            return env_val
        env_dir = os.environ.get("SGLANG_ENV_DIR", "")
        if env_dir:
            commit_file = Path(env_dir) / "sglang_commit.txt"
            if commit_file.is_file():
                try:
                    return commit_file.read_text(encoding="utf-8").strip()
                except OSError:
                    pass
        return DEFAULT_SGLANG_COMMIT

    def engine_config_dict(self) -> dict:
        """Recorded engine configuration (pinned flags + provenance)."""
        return {
            "runtime": "sglang",
            "residency_mode": self.residency_mode,
            "model_path": self.model_path,
            "model_id": self.model_id,
            "host": self.host,
            "port": self.port,
            "dtype": self.dtype,
            "mem_fraction_static": self.mem_fraction_static,
            "page_size": self.page_size,
            "tp_size": self.tp_size,
            "max_running_requests": self.max_running_requests,
            "enable_prefix_caching": self.residency_mode != "recompute",
            "enable_cache_report": True,
            "enable_hierarchical_cache": self.residency_mode == "cpu_hit",
            "hicache_io_backend": self.hicache_io_backend,
            "hicache_mem_layout": self.hicache_mem_layout,
            "hicache_write_policy": self.hicache_write_policy,
            "hicache_ratio": self.hicache_ratio,
            "hicache_size_gb": self.hicache_size_gb,
            "sglang_commit": self.resolve_commit(),
        }

    def to_json(self) -> str:
        return json.dumps(self.engine_config_dict(), indent=2, ensure_ascii=False)
