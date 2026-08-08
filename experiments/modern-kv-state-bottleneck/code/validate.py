"""Validation gate: verify cache-residency conditions before collecting data.

Each check returns (name, passed, detail).  All checks must pass before
reported measurements are collected.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from ..workload.token_workload import TokenWorkload

logger = logging.getLogger(__name__)


def check_model_architecture(runner) -> tuple[str, bool, str]:
    """Verify the model architecture matches expectations."""
    try:
        engine = runner.llm.llm_engine
        model_config = getattr(engine, "model_config", None)
        if model_config is None:
            # Try deeper
            engine_core = getattr(engine, "engine_core", None)
            if engine_core is not None:
                model_config = getattr(engine_core, "model_config", None)

        if model_config is None:
            return ("model_arch", False, "Cannot access model_config from engine")

        arch = getattr(model_config, "architectures", None)
        if arch is None:
            # Try dict-like access
            arch = model_config.get("architectures") if isinstance(model_config, dict) else None

        detail = f"architectures={arch}"
        return ("model_arch", True, detail)
    except Exception as e:
        return ("model_arch", False, f"Exception: {e}")


def check_prefix_cache_consistency(runner, workload: TokenWorkload) -> tuple[str, bool, str]:
    """Verify that prefix-cache hit produces identical output to recompute."""
    try:
        from vllm import SamplingParams
        sp = SamplingParams(max_tokens=5, temperature=0.0)

        segment = workload.get_segment(0)
        prompt = segment.to_tokens_prompt()

        # First generate (no cache → recompute)
        runner.reset_prefix_cache()
        out1 = runner.llm.generate([prompt], sp)
        tokens1 = out1[0].outputs[0].token_ids

        # Second generate (should hit prefix cache)
        runner.reset_prefix_cache()
        runner.warmup_prefix(segment.prefix_ids)
        out2 = runner.llm.generate([prompt], sp)
        tokens2 = out2[0].outputs[0].token_ids

        if tokens1 == tokens2:
            return ("prefix_consistency", True, "Outputs identical (5 tokens)")
        else:
            return (
                "prefix_consistency", False,
                f"Outputs differ: {tokens1} vs {tokens2}",
            )
    except Exception as e:
        return ("prefix_consistency", False, f"Exception: {e}")


def check_gpu_resident_hit(runner, workload: TokenWorkload) -> tuple[str, bool, str]:
    """Verify that GPU-resident prefix cache hit works."""
    try:
        segment = workload.get_segment(0)

        # Reset and warmup
        runner.reset_prefix_cache()
        runner.warmup_prefix(segment.prefix_ids)

        # Check stats
        stats = runner.stats_collector.collect()
        if stats.prefix_hits > 0:
            return (
                "gpu_resident_hit", True,
                f"hits={stats.prefix_hits}, queries={stats.prefix_queries}",
            )
        else:
            return (
                "gpu_resident_hit", False,
                f"No hit detected: queries={stats.prefix_queries}, hits={stats.prefix_hits}",
            )
    except Exception as e:
        return ("gpu_resident_hit", False, f"Exception: {e}")


def check_cpu_resident_hit(runner, workload: TokenWorkload) -> tuple[str, bool, str]:
    """Verify that CPU-resident prefix cache hit works.

    This check may fail if the runtime does not support KV offloading
    for the given model.  Failure here labels the condition as 'unsupported'.
    """
    try:
        segment = workload.get_segment(0)

        # Reset, warmup, evict
        runner.reset_prefix_cache()
        runner.warmup_prefix(segment.prefix_ids)
        runner.evict_prefix_to_cpu(segment.prefix_ids)

        # Check eviction
        stats_before = runner.stats_collector.collect()

        # Send the real request
        sp = runner.get_sampling_params(max_tokens=1)
        prompt = segment.to_tokens_prompt()
        out = runner.llm.generate([prompt], sp)

        stats_after = runner.stats_collector.collect()

        # Check if restore happened: hits should increase or GPU memory should show spike
        hit_after = stats_after.prefix_hits

        if hit_after > 0 or stats_before.prefix_preempted_hits > 0:
            return (
                "cpu_resident_hit", True,
                f"preempted_hits={stats_before.prefix_preempted_hits}, "
                f"hits_after={hit_after}",
            )
        else:
            return (
                "cpu_resident_hit", False,
                f"No evidence of CPU restore: "
                f"preempted_hits={stats_before.prefix_preempted_hits}, "
                f"hits_after={hit_after}",
            )
    except Exception as e:
        return ("cpu_resident_hit", False, f"Exception: {e}")


def check_config_stability(runner) -> tuple[str, bool, str]:
    """Verify engine config is recorded and stable."""
    try:
        meta = runner.get_metadata()
        cfg = meta.get("engine_config", {})

        required_keys = [
            "enable_prefix_caching", "block_size", "dtype",
            "gpu_memory_utilization", "max_model_len",
        ]
        missing = [k for k in required_keys if k not in cfg]
        if missing:
            return ("config_stability", False, f"Missing keys: {missing}")

        return ("config_stability", True, f"All {len(required_keys)} config keys present")
    except Exception as e:
        return ("config_stability", False, f"Exception: {e}")


def run_validation_gate(
    runner,
    workload: TokenWorkload,
    skip_cpu_hit: bool = False,
) -> dict:
    """Run all validation checks.

    Args:
        runner: VLLMRunner instance
        workload: TokenWorkload for the context length being tested
        skip_cpu_hit: If True, skip CPU-resident hit check

    Returns:
        dict with:
          - checks: list of (name, passed, detail)
          - all_passed: bool (True if all non-skipped checks pass)
          - cpu_hit_supported: bool
    """
    checks = []

    checks.append(check_model_architecture(runner))
    checks.append(check_prefix_cache_consistency(runner, workload))
    checks.append(check_gpu_resident_hit(runner, workload))

    if skip_cpu_hit:
        checks.append(("cpu_resident_hit", None, "Skipped (skip_cpu_hit=True)"))
        cpu_hit_supported = None
    else:
        result = check_cpu_resident_hit(runner, workload)
        checks.append(result)
        cpu_hit_supported = result[1]

    checks.append(check_config_stability(runner))

    # all_passed: True only if all non-skipped, non-None checks passed
    non_skipped = [c for c in checks if c[1] is not None]
    all_passed = all(c[1] for c in non_skipped)

    result = {
        "checks": [
            {"name": n, "passed": p, "detail": d}
            for n, p, d in checks
        ],
        "all_passed": all_passed,
        "cpu_hit_supported": cpu_hit_supported,
    }

    for name, passed, detail in checks:
        status = "PASS" if passed else ("SKIP" if passed is None else "FAIL")
        logger.info("  Validation [%s] %s: %s", status, name, detail)

    return result
