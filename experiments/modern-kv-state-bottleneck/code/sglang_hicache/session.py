"""Shared server session orchestration for the SGLang Exp1-3 entry points.

One session = launch server -> wait ready -> run experiment -> graceful
shutdown.  Real hangs are left to ``scancel`` at the Slurm job level
(see ``server_lifecycle``).
"""

from __future__ import annotations

import logging
import platform
import socket
import sys
from pathlib import Path
from typing import Callable, Optional

from sglang_hicache.config import SGLangServerConfig
from sglang_hicache.http_client import SGLangHTTPClient
from sglang_hicache.metrics import (
    CacheStats,
    parse_prometheus_text,
    snapshot_from_scrape,
)
from sglang_hicache.server_lifecycle import SGLangServerProcess

logger = logging.getLogger(__name__)


def make_snapshot_fn(client: SGLangHTTPClient) -> Callable[[], CacheStats]:
    """Return a snapshot callable that scrapes /metrics and parses it."""

    def _snapshot() -> CacheStats:
        text = client.fetch_metrics_text()
        scrape = parse_prometheus_text(text)
        return snapshot_from_scrape(scrape)

    return _snapshot


def collect_snapshot(client: SGLangHTTPClient) -> CacheStats:
    return make_snapshot_fn(client)()


def start_server_session(
    server_config: SGLangServerConfig,
    log_dir: str | Path,
    python_executable: Optional[str] = None,
    ready_timeout_s: float = 900.0,
) -> tuple[SGLangServerProcess, SGLangHTTPClient, Callable[[], CacheStats]]:
    """Launch the server, wait for readiness, return the session handles."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"sglang-server-{server_config.port}.out"
    stderr_path = log_dir / f"sglang-server-{server_config.port}.err"

    server = SGLangServerProcess(
        argv=server_config.build_argv(),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        python_executable=python_executable,
        ready_timeout_s=ready_timeout_s,
    )
    client = SGLangHTTPClient(server_config.base_url())

    server.start()
    try:
        server.wait_ready(client.health)
    except Exception:
        server.stop()
        raise

    snapshot_fn = make_snapshot_fn(client)
    return server, client, snapshot_fn


def hardware_metadata() -> dict:
    """Record hardware/platform facts available without GPU tooling."""
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "slurm_job_id": _env_or_none("SLURM_JOB_ID"),
        "slurm_nodelist": _env_or_none("SLURM_JOB_NODELIST"),
        "slurm_partition": _env_or_none("SLURM_JOB_PARTITION"),
        "cuda_visible_devices": _env_or_none("CUDA_VISIBLE_DEVICES"),
    }


def _env_or_none(key: str) -> Optional[str]:
    import os

    return os.environ.get(key)


def server_metadata(
    server_config: SGLangServerConfig,
    log_dir: str | Path,
) -> dict:
    """Metadata block recorded in every SGLang run's metadata.json."""
    return {
        "runtime": "sglang",
        "server": server_config.engine_config_dict(),
        "server_logs": {
            "stdout": str(Path(log_dir) / f"sglang-server-{server_config.port}.out"),
            "stderr": str(Path(log_dir) / f"sglang-server-{server_config.port}.err"),
        },
        "hardware": hardware_metadata(),
    }
