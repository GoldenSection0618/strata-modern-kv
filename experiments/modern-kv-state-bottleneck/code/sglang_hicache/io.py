"""Shared JSON output helpers for the SGLang Exp1-3 entry points.

Keeps the raw / summary / validation / metadata separation identical to
the vLLM path (raw results are never overwritten; every run gets a
unique directory).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_run_tag(override: str | None = None) -> str:
    """Unique per-run tag: ``<UTC timestamp>[-job<SLURM_JOB_ID>]``.

    Every run gets its own output directory (``run-<tag>``) so repeated
    runs never overwrite raw files.  ``override`` (sbatch ``RUN_TAG``)
    wins when provided; otherwise the tag is derived from the UTC clock
    and, inside a Slurm job, the job id.
    """
    if override:
        return override.strip()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    job = os.environ.get("SLURM_JOB_ID", "").strip()
    return f"{ts}-job{job}" if job else ts


def run_output_dir(base: str | Path, run_tag: str, residency: str | None = None) -> Path:
    """Run-tagged output directory: ``<base>[/<residency>]/run-<run_tag>``.

    Keeps the raw/summary/validation/metadata separation inside each
    unique run directory; analysis discovers results recursively.
    """
    out = Path(base)
    if residency:
        out = out / residency
    return out / f"run-{run_tag}"


def write_json(obj, path: str | Path, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=indent, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Saved: %s", path)
    return path


def save_raw_result(result: dict, output_dir: str | Path, rep: int) -> Path:
    raw_dir = Path(output_dir) / "raw"
    return write_json(result, raw_dir / f"rep_{rep:02d}.json")


def save_metadata(metadata: dict, output_dir: str | Path) -> Path:
    return write_json(metadata, Path(output_dir) / "metadata.json")


def save_validation(validation: dict, output_dir: str | Path) -> Path:
    return write_json(validation, Path(output_dir) / "validation.json")


def save_summary(summary: dict, output_dir: str | Path) -> Path:
    return write_json(summary, Path(output_dir) / "summary.json")


def save_unsupported(
    reason_payload: dict, output_dir: str | Path
) -> Path:
    """Write the negative/unsupported marker for a failed validation gate."""
    return write_json(reason_payload, Path(output_dir) / "unsupported.json")
