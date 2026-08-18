"""SGLang validation gate (shared by Exp1-3).

Mirrors the vLLM ``validate.py`` gate structure and output schema
(``validation.json`` with ``checks`` / ``all_passed`` /
``cpu_hit_supported``) so results from either runtime have the same
shape.  All checks run against the public HTTP surface:

* server readiness + model info (``/model_info``);
* prefix-consistency: greedy output token ids agree between a recompute
  request and a cache-hit request (exact ids from
  ``meta_info.output_token_logprobs``);
* per-residency evidence evaluation (pure functions in ``residency.py``);
* config stability: resolved ``/server_info`` flags match the pinned
  ``SGLangServerConfig``.

A failed hit-validation gate prevents measurement and is preserved as a
negative / unsupported result — it is never silently retried into a
passing result.
"""

from __future__ import annotations

import logging
from typing import Callable

from sglang_hicache.http_client import SGLangHTTPClient, GenerateResult
from sglang_hicache.metrics import CacheStats
from sglang_hicache.residency import (
    evaluate_cpu_hit_evidence,
    evaluate_gpu_hit_evidence,
    evaluate_recompute_evidence,
    prepare_cpu_hit,
    prepare_gpu_hit,
    prepare_recompute,
)
from sglang_hicache.config import SGLangServerConfig

logger = logging.getLogger(__name__)

#: Check names (stable identifiers used by docs and analysis).
CHECK_READY = "server_ready"
CHECK_MODEL_ARCH = "model_arch"
CHECK_PREFIX_CONSISTENCY = "prefix_consistency"
CHECK_RECOMPUTE_NO_CACHE = "recompute_no_cache"
CHECK_GPU_HIT = "gpu_resident_hit"
CHECK_CPU_HIT = "cpu_resident_hit"
CHECK_CONFIG = "config_stability"


def check_server_ready(client: SGLangHTTPClient) -> tuple[str, bool, str]:
    try:
        ok = client.health()
    except Exception as e:  # noqa: BLE001
        return (CHECK_READY, False, f"health check error: {e}")
    return (CHECK_READY, ok, "health=ok" if ok else "health check failed")


def resolve_model_path(path: str) -> str:
    """Canonical-path resolution for model-checkpoint comparison.

    Documented resolution: strip whitespace, expand ``~``, then
    ``os.path.realpath`` (resolves symlinks and ``..`` components).  Only
    this canonicalization is allowed when comparing the resolved server
    model path against the expected checkpoint.
    """
    import os

    return os.path.realpath(os.path.expanduser(str(path).strip()))


def model_path_matches(reported: str, expected: str) -> tuple[bool, str]:
    """Pure comparison of the resolved model path against the expected one.

    Returns ``(ok, detail)``.  A nonempty expected checkpoint must match
    the reported path under canonical-path resolution; a nonempty wrong
    model must fail.
    """
    if not expected:
        return (
            False,
            "no expected model path configured; cannot verify the model "
            "checkpoint identity",
        )
    if not reported:
        return (
            False,
            f"server reported an empty model_path (expected {expected!r})",
        )
    reported_resolved = resolve_model_path(reported)
    expected_resolved = resolve_model_path(expected)
    if reported_resolved == expected_resolved:
        return (
            True,
            f"model_path matches expected checkpoint "
            f"(resolved {reported_resolved!r})",
        )
    return (
        False,
        f"model_path {reported!r} (resolved {reported_resolved!r}) does not "
        f"match expected checkpoint {expected!r} (resolved "
        f"{expected_resolved!r})",
    )


def check_model_arch(
    client: SGLangHTTPClient, expected_model_path: str
) -> tuple[str, bool, str]:
    """The resolved /model_info model path must match the expected checkpoint.

    ``expected_model_path`` is the pinned ``--model-path``; the server must
    report the same checkpoint under canonical-path resolution
    (:func:`model_path_matches`).  Architectures/model_type are recorded in
    the detail for provenance but do not substitute for the path check.
    """
    try:
        info = client.model_info()
    except Exception as e:  # noqa: BLE001
        return (CHECK_MODEL_ARCH, False, f"cannot fetch /model_info: {e}")
    model_path = info.get("model_path")
    arches = info.get("architectures")
    model_type = info.get("model_type")
    detail = f"model_path={model_path!r}, architectures={arches}, model_type={model_type}"
    if not model_path:
        return (CHECK_MODEL_ARCH, False, f"model_info missing model_path: {detail}")
    ok, reason = model_path_matches(str(model_path), expected_model_path)
    return (CHECK_MODEL_ARCH, ok, f"{reason}; {detail}")


def check_prefix_consistency(
    client: SGLangHTTPClient,
    input_ids: list[int],
    resolver: Callable[[], GenerateResult],
    request_id_base: int = 2000,
) -> tuple[str, bool, str]:
    """Greedy output token id equality: recompute vs cache-hit.

    The reference (recompute) output is produced on a freshly flushed
    cache; the resolver returns the output of a *cached* request after
    the appropriate residency preparation.  Both must produce the same
    greedy output token id (``meta_info.output_token_logprobs``).
    """
    try:
        client.flush_cache()
        ref = client.generate(input_ids, max_new_tokens=1, request_id=request_id_base)
        hit = resolver()
    except Exception as e:  # noqa: BLE001
        return (CHECK_PREFIX_CONSISTENCY, False, f"Exception: {e}")

    if not ref.ok or not hit.ok:
        return (
            CHECK_PREFIX_CONSISTENCY,
            False,
            f"request failed: ref_ok={ref.ok} ({ref.error[:120]}), "
            f"hit_ok={hit.ok} ({hit.error[:120]})",
        )
    if ref.output_token_id is None or hit.output_token_id is None:
        return (
            CHECK_PREFIX_CONSISTENCY,
            False,
            "output token ids unavailable (return_logprob not honored by server)",
        )
    same = ref.output_token_id == hit.output_token_id
    detail = (
        f"output token ids identical ({ref.output_token_id})"
        if same
        else f"output token ids differ: {ref.output_token_id} vs {hit.output_token_id}"
    )
    return (CHECK_PREFIX_CONSISTENCY, same, detail)


def check_recompute_no_cache(
    client: SGLangHTTPClient, input_ids: list[int], prefix_len: int
) -> tuple[str, bool, str]:
    """recompute: repeated prefix must not be served from cache."""
    prepare_recompute(client)
    first = client.generate(input_ids, max_new_tokens=1, request_id=3000)
    second = client.generate(input_ids, max_new_tokens=1, request_id=3001)
    if not first.ok or not second.ok:
        return (
            CHECK_RECOMPUTE_NO_CACHE,
            False,
            f"request failed: first_ok={first.ok}, second_ok={second.ok}",
        )
    passed, reason = evaluate_recompute_evidence(second.meta_info, prefix_len)
    return (CHECK_RECOMPUTE_NO_CACHE, passed, reason)


def check_gpu_resident_hit(
    client: SGLangHTTPClient,
    input_ids: list[int],
    prefix_ids: list[int],
    snapshot_fn: Callable[[], CacheStats],
    request_id_base: int = 4000,
) -> tuple[str, bool, str]:
    """gpu_hit: warm prefix into GPU L1, then measure a full request."""
    try:
        prepare_gpu_hit(client, prefix_ids, request_id=request_id_base)
        before = snapshot_fn()
        result = client.generate(input_ids, max_new_tokens=1, request_id=request_id_base + 1)
        after = snapshot_fn()
    except Exception as e:  # noqa: BLE001
        return (CHECK_GPU_HIT, False, f"Exception: {e}")

    from sglang_hicache.metrics import diff_snapshots

    delta = diff_snapshots(after, before)
    passed, reason = evaluate_gpu_hit_evidence(result.meta_info, len(prefix_ids), delta)
    return (CHECK_GPU_HIT, passed, reason)


def check_cpu_resident_hit(
    client: SGLangHTTPClient,
    input_ids: list[int],
    prefix_ids: list[int],
    snapshot_fn: Callable[[], CacheStats],
    context_length: int,
    seed: int = 42,
    request_id_base: int = 5000,
) -> tuple[str, bool, str]:
    """cpu_hit: warm, evict prefix to host L2, then measure the restore."""
    try:
        prep = prepare_cpu_hit(
            client,
            prefix_ids,
            snapshot_fn,
            context_length=context_length,
            seed=seed,
            request_id_base=request_id_base,
        )
        before = snapshot_fn()
        # Measured request id must not collide with warmup/filler ids.
        result = client.generate(
            input_ids, max_new_tokens=1, request_id=request_id_base + 10000
        )
        after = snapshot_fn()
    except Exception as e:  # noqa: BLE001
        return (CHECK_CPU_HIT, False, f"Exception: {e}")

    from sglang_hicache.metrics import diff_snapshots

    delta = diff_snapshots(after, before)
    passed, reason = evaluate_cpu_hit_evidence(result.meta_info, len(prefix_ids), delta)
    detail = (
        f"{reason} | prep: filler_tokens={prep.filler_tokens_sent}, "
        f"host_delta={prep.host_used_delta if prep.host_used_delta is not None else 'unsupported'}"
    )
    return (CHECK_CPU_HIT, passed, detail)


def config_mismatches(info: dict, expected: SGLangServerConfig) -> list[str]:
    """Pure comparison of resolved /server_info flags vs pinned config."""
    checks = {
        "disable_radix_cache": info.get("disable_radix_cache"),
        "enable_hierarchical_cache": info.get("enable_hierarchical_cache"),
        "enable_cache_report": info.get("enable_cache_report"),
        "page_size": info.get("page_size"),
        "hicache_io_backend": info.get("hicache_io_backend"),
        "hicache_mem_layout": info.get("hicache_mem_layout"),
        "hicache_write_policy": info.get("hicache_write_policy"),
        "hicache_ratio": info.get("hicache_ratio"),
        "hicache_size": info.get("hicache_size"),
    }
    # Common runtime flags apply to every residency. HiCache tuning flags are
    # only meaningful when hierarchical cache is enabled (`cpu_hit`). For
    # recompute/gpu_hit SGLang keeps its own inactive HiCache defaults, so
    # comparing them with the pinned cpu_hit values creates a false failure.
    expected_map = {
        "disable_radix_cache": expected.residency_mode == "recompute",
        "enable_hierarchical_cache": expected.residency_mode == "cpu_hit",
        "enable_cache_report": True,
        "page_size": expected.page_size,
    }
    if expected.residency_mode == "cpu_hit":
        expected_map.update({
            "hicache_io_backend": expected.hicache_io_backend,
            "hicache_mem_layout": expected.hicache_mem_layout,
            "hicache_write_policy": expected.hicache_write_policy,
            "hicache_ratio": expected.hicache_ratio,
        })
        if expected.hicache_size_gb and expected.hicache_size_gb > 0:
            expected_map["hicache_size"] = expected.hicache_size_gb

    mismatches = []
    for key, exp in expected_map.items():
        actual = checks.get(key)
        if actual is None:
            mismatches.append(f"{key}=missing")
        elif isinstance(exp, bool) and actual is not None:
            if bool(actual) != exp:
                mismatches.append(f"{key}={actual} (expected {exp})")
        elif isinstance(exp, (int, float)):
            try:
                if abs(float(actual) - float(exp)) > 1e-9:
                    mismatches.append(f"{key}={actual} (expected {exp})")
            except (TypeError, ValueError):
                mismatches.append(f"{key}={actual} (expected {exp})")
        elif str(actual) != str(exp):
            mismatches.append(f"{key}={actual} (expected {exp})")
    return mismatches


def check_config_stability(
    client: SGLangHTTPClient, expected: SGLangServerConfig
) -> tuple[str, bool, str]:
    """Resolved /server_info flags must match the pinned config."""
    try:
        info = client.server_info()
    except Exception as e:  # noqa: BLE001
        return (CHECK_CONFIG, False, f"cannot fetch /server_info: {e}")

    mismatches = config_mismatches(info, expected)
    version = info.get("version")
    checks = {
        k: info.get(k)
        for k in (
            "disable_radix_cache", "enable_hierarchical_cache",
            "enable_cache_report", "page_size",
            "hicache_io_backend", "hicache_mem_layout", "hicache_write_policy",
            "hicache_ratio", "hicache_size", "mem_fraction_static",
        )
    }
    detail = f"version={version}, resolved flags: {checks}"
    if mismatches:
        return (CHECK_CONFIG, False, f"{detail}; mismatches: {mismatches}")
    return (CHECK_CONFIG, True, detail)


def run_validation_gate(
    client: SGLangHTTPClient,
    server_config: SGLangServerConfig,
    input_ids: list[int],
    prefix_ids: list[int],
    snapshot_fn: Callable[[], CacheStats],
    *,
    context_length: int,
    seed: int = 42,
) -> dict:
    """Run the full validation gate; returns the ``validation.json`` dict."""
    checks = [check_server_ready(client)]
    checks.append(check_model_arch(client, server_config.model_path))
    checks.append(check_config_stability(client, server_config))

    if server_config.residency_mode == "recompute":
        checks.append(
            check_recompute_no_cache(client, input_ids, len(prefix_ids))
        )
        checks.append((CHECK_CPU_HIT, None, "Skipped (recompute: no cache tiers)"))
        cpu_hit_supported = None
    else:
        checks.append(
            check_prefix_consistency(
                client,
                input_ids,
                resolver=lambda: _hit_request(client, input_ids, request_id=2100),
                request_id_base=2000,
            )
        )
        checks.append(
            check_gpu_resident_hit(
                client, input_ids, prefix_ids, snapshot_fn, request_id_base=4000
            )
        )
        if server_config.residency_mode == "cpu_hit":
            checks.append(
                check_cpu_resident_hit(
                    client,
                    input_ids,
                    prefix_ids,
                    snapshot_fn,
                    context_length=context_length,
                    seed=seed,
                    request_id_base=5000,
                )
            )
            cpu_hit_supported = checks[-1][1]
        else:
            checks.append((CHECK_CPU_HIT, None, "Skipped (residency=gpu_hit)"))
            cpu_hit_supported = None

    non_skipped = [c for c in checks if c[1] is not None]
    all_passed = all(c[1] for c in non_skipped)

    result = {
        "checks": [
            {"name": n, "passed": p, "detail": d}
            for n, p, d in checks
        ],
        "all_passed": all_passed,
        "cpu_hit_supported": cpu_hit_supported,
        "runtime": "sglang",
    }

    for name, passed, detail in checks:
        status = "PASS" if passed else ("SKIP" if passed is None else "FAIL")
        logger.info("  Validation [%s] %s: %s", status, name, detail)

    return result


def _hit_request(
    client: SGLangHTTPClient, input_ids: list[int], request_id: int
) -> GenerateResult:
    """Run a request after residency preparation (used by consistency check)."""
    return client.generate(input_ids, max_new_tokens=1, request_id=request_id)
