"""Full-hierarchy capability gate runner.

Executes the four serial probes across GPU-only and hierarchical
servers, plus a concurrent steady-state window probe on the
hierarchical server, and writes ``validation.json`` (the reportability
gate: a Slurm ``COMPLETED`` alone never validates a result).

Probes (see ``hcv.probes`` for expectations):

1. recompute (cold miss)
2. GPU hit (warm revisit, L1 resident)
3. GPU-only eviction negative control (evict via filler on a GPU-only
   server, revisit must recompute, no host hit)
4. CPU hit (evict via filler on the hierarchical server, revisit must
   restore from host, no recompute, no device hit)

The concurrent window probe uses deterministic prefix families (never
one shared prefix) and classifies window origins with total tier hits
plus capped ``load_back_tokens_total``.

Outcomes: ``full`` | ``partial`` | ``unsupported`` |
``invalid_infrastructure`` (see ``hcv.hierarchy``).
"""

from __future__ import annotations

import logging
import os
import sys

from hcv.config import (
    ARCH_HIERARCHICAL,
    GATE_FULL,
    GATE_INVALID_INFRA,
    ExperimentConfig,
    pair_configs,
)
from hcv.filler import build_filler_plan, observed_l1_capacity, run_filler
from hcv.hierarchy import (
    PROBE_CPU_HIT,
    PROBE_GPU_HIT,
    PROBE_GPU_ONLY_EVICTION,
    PROBE_RECOMPUTE,
    GateResult,
    classify_gate,
)
from hcv.http_client import SGLangHTTPClient
from hcv.load_driver import run_load
from hcv.probes import ProbeSpec, run_serial_probe
from hcv.residency import prepare_cold
from hcv.run_common import (
    base_metadata,
    job_id,
    launch_server,
    log_dir_default,
    parse_common_args,
    resolve_config,
    resolve_run_tag,
    run_dir_for,
    setup_logging,
    static_checks,
    utc_now,
)
from hcv.schema import RunLayout, append_jsonl, write_json_atomic
from hcv.workload import build_trace

logger = logging.getLogger(__name__)


def _metrics_usable(client: SGLangHTTPClient) -> bool:
    """Verify the public /metrics endpoint serves the tier counters."""
    try:
        scrape = client.scrape_metrics()
        return scrape.has("sglang:prefill_effective_tokens_total")
    except Exception:  # noqa: BLE001
        return False


def _probe_family_ids(prefix_length: int, seed: int, family_no: int) -> list[int]:
    """Deterministic probe prefix (family no. distinct from filler)."""
    from hcv.workload import _ids

    return _ids(seed * 131 + family_no * 7919, prefix_length)


def _suffix_ids(seed: int, length: int) -> list[int]:
    from hcv.workload import _ids

    return _ids(seed * 17 + length, length)


def _filler_plan(client, cfg, prefix_length):
    from hcv.metrics import snapshot_from_scrape

    stats = snapshot_from_scrape(client.scrape_metrics())
    capacity = observed_l1_capacity(stats)
    source = "observed_max_total_num_tokens" if stats.max_total_num_tokens is not None else "observed_kv_sum"
    return build_filler_plan(
        context_length=cfg.prefix_length,
        protected_prefix_tokens=prefix_length + 256,
        observed_capacity=capacity,
        capacity_source=source,
        filler_prefix_length=cfg.prefix_length,
    )


def run_gpu_only_probes(
    client: SGLangHTTPClient,
    cfg: ExperimentConfig,
    prefix_ids: list[int],
    request_id_base: int,
) -> dict:
    """Run recompute / GPU hit / eviction negative control on one server.

    Returns ProbeResult objects under the probe names (callers convert
    to dicts when recording evidence).
    """
    prefix_length = len(prefix_ids)
    suffix = _suffix_ids(cfg.seed, 16)
    input_ids = prefix_ids + suffix

    prepare_cold(client)

    recompute = run_serial_probe(
        client,
        ProbeSpec(PROBE_RECOMPUTE, input_ids, prefix_length, request_id_base + 0),
    )
    gpu_hit = run_serial_probe(
        client,
        ProbeSpec(PROBE_GPU_HIT, input_ids, prefix_length, request_id_base + 1),
    )

    # --- eviction negative control ---------------------------------------
    # Use observed L1 capacity when available; fill deterministically with
    # unique prefixes until L1 saturation evidence appears.  Protected
    # prefix tokens are the probe prefix (must be displaced).
    plan = _filler_plan(client, cfg, prefix_length)
    filler = run_filler(client, plan, seed=cfg.seed)
    neg = run_serial_probe(
        client,
        ProbeSpec(PROBE_GPU_ONLY_EVICTION, input_ids, prefix_length, request_id_base + 2),
    )
    return {
        "recompute": recompute,
        "gpu_hit": gpu_hit,
        "negative_control": neg,
        "filler": filler,
    }


def run_hierarchical_probes(
    client: SGLangHTTPClient,
    cfg: ExperimentConfig,
    prefix_ids: list[int],
    request_id_base: int,
) -> dict:
    """Run recompute / GPU hit / CPU hit probes + concurrent windows.

    Returns ProbeResult objects (callers convert to dicts when recording
    evidence).
    """
    prefix_length = len(prefix_ids)
    suffix = _suffix_ids(cfg.seed, 16)
    input_ids = prefix_ids + suffix

    prepare_cold(client)

    recompute = run_serial_probe(
        client,
        ProbeSpec(PROBE_RECOMPUTE, input_ids, prefix_length, request_id_base + 0),
    )
    gpu_hit = run_serial_probe(
        client,
        ProbeSpec(PROBE_GPU_HIT, input_ids, prefix_length, request_id_base + 1),
    )

    plan = _filler_plan(client, cfg, prefix_length)
    filler = run_filler(client, plan, seed=cfg.seed + 1)
    cpu_hit = run_serial_probe(
        client,
        ProbeSpec(PROBE_CPU_HIT, input_ids, prefix_length, request_id_base + 2),
    )

    # --- concurrent steady-state windows (deterministic prefix families) --
    window_trace = build_trace(
        seed=cfg.seed + 7,
        num_prefix_families=cfg.num_prefix_families,
        family_size=cfg.family_size,
        prefix_length=cfg.prefix_length,
        suffix_length_min=16,
        suffix_length_max=32,
        output_length=1,
        revisit_fraction=0.5,
        request_count=cfg.request_count,
    )
    run = run_load(
        client,
        window_trace,
        window_trace.records,
        concurrency=min(4, cfg.concurrency),
        request_id_offset=request_id_base + 100,
        window_requests=min(32, cfg.request_count),
        max_windows=16,
    )
    return {
        "recompute": recompute,
        "gpu_hit": gpu_hit,
        "cpu_hit": cpu_hit,
        "filler": filler,
        "concurrent_windows": run,
    }


def main(argv=None) -> int:
    setup_logging()
    args = parse_common_args(argv)
    cfg = resolve_config(args, experiment="validation")
    tag = resolve_run_tag(cfg)
    log_dir = args.get("log_dir") or log_dir_default()

    layout = RunLayout(run_dir_for(cfg, tag)).create()
    append_jsonl(layout.measurements_path, {"kind": "gate_start", "run_tag": tag,
                                            "utc": utc_now(), "job": job_id()})

    try:
        checks = static_checks(cfg, log_dir)
    except RuntimeError as e:
        # Invalid infrastructure: record it, do not launch anything.
        logger.error("static checks failed: %s", e)
        gate = GateResult(
            status=GATE_INVALID_INFRA, model=cfg.model, architecture=ARCH_HIERARCHICAL,
            probe_results={}, reasons=[str(e)],
            infra_checks={"env_ok": False, "error": str(e)},
        )
        _finalize(layout, cfg, tag, {"env_markers_ok": False, "env_markers_errors": [str(e)],
                                    "jit_toolchain_ok": False, "jit_toolchain_errors": [],
                                    "model_path_ok": False, "model_path_errors": [],
                                    "log_path_ok": False, "log_path_errors": []},
                  gate, {}, [], None, None)
        return 1
    gpu_cfg, hier_cfg = pair_configs(cfg)

    infra = {
        "env_ok": checks["env_markers_ok"],
        "env_error": "; ".join(checks["env_markers_errors"]),
        "jit_ok": checks["jit_toolchain_ok"],
        "model_path_ok": checks["model_path_ok"],
        "log_path_ok": checks["log_path_ok"],
        "server_ok": False,
        "metrics_ok": False,
    }

    probes: dict = {}
    concurrent_windows: list = []
    aggregate_host_hit: float | None = None
    aggregate_load_back: float | None = None

    try:
        # --- GPU-only server ----------------------------------------------
        proc, client, _port = launch_server(gpu_cfg, log_dir, layout)
        try:
            infra["server_ok"] = True
            infra["metrics_ok"] = _metrics_usable(client)
            gpu_probes = run_gpu_only_probes(client, gpu_cfg, _probe_family_ids(
                gpu_cfg.prefix_length, gpu_cfg.seed, 1), 1000)
            probes.update({PROBE_RECOMPUTE: gpu_probes["recompute"],
                           PROBE_GPU_HIT: gpu_probes["gpu_hit"],
                           PROBE_GPU_ONLY_EVICTION: gpu_probes["negative_control"]})
            append_jsonl(layout.measurements_path, {"kind": "gpu_only_probes",
                                                    "run_tag": tag,
                                                    "probes": {
                                                        k: v.to_dict() if hasattr(v, "to_dict") else v
                                                        for k, v in gpu_probes.items()
                                                    }})
        finally:
            proc.stop()
    except Exception as e:  # noqa: BLE001
        infra["server_error"] = f"gpu_only server: {e}"
        logger.error("GPU-only gate phase failed: %s", e)
        gate = classify_gate(cfg.model, ARCH_HIERARCHICAL, cfg.model_state_groups,
                             {}, None, None, infra_checks=infra)
        _finalize(layout, cfg, tag, checks, gate, probes, concurrent_windows,
                  aggregate_host_hit, aggregate_load_back)
        return 1 if gate.status == GATE_INVALID_INFRA else 0

    try:
        # --- hierarchical server ------------------------------------------
        infra["server_ok"] = False
        infra["metrics_ok"] = False
        proc, client, _port = launch_server(hier_cfg, log_dir, layout)
        try:
            infra["server_ok"] = True
            infra["metrics_ok"] = _metrics_usable(client)
            hier_probes = run_hierarchical_probes(client, hier_cfg, _probe_family_ids(
                hier_cfg.prefix_length, hier_cfg.seed, 2), 2000)
            probes.update({PROBE_RECOMPUTE: hier_probes["recompute"],
                           PROBE_GPU_HIT: hier_probes["gpu_hit"],
                           PROBE_CPU_HIT: hier_probes["cpu_hit"]})
            aggregate_host_hit = hier_probes["cpu_hit"].host_hit_delta
            aggregate_load_back = hier_probes["cpu_hit"].load_back_delta
            concurrent_windows = hier_probes["concurrent_windows"].windows
            append_jsonl(layout.measurements_path, {"kind": "hierarchical_probes",
                                                    "run_tag": tag,
                                                    "probes": {
                                                        k: v.to_dict() if hasattr(v, "to_dict") else v
                                                        for k, v in hier_probes.items()
                                                    }})
        finally:
            proc.stop()
    except Exception as e:  # noqa: BLE001
        infra["server_error"] = f"hierarchical server: {e}"
        logger.error("Hierarchical gate phase failed: %s", e)

    gate = classify_gate(
        cfg.model, ARCH_HIERARCHICAL, cfg.model_state_groups,
        probes, aggregate_host_hit, aggregate_load_back,
        infra_checks=infra, concurrent_windows=concurrent_windows,
    )
    _finalize(layout, cfg, tag, checks, gate, probes, concurrent_windows,
              aggregate_host_hit, aggregate_load_back)
    return 0 if gate.status != GATE_INVALID_INFRA else 1


def _finalize(layout, cfg, tag, checks, gate, probes, windows,
              agg_host, agg_load) -> None:
    """Write validation.json + metadata.json for this gate run."""
    write_json_atomic(layout.validation_path, gate.to_dict())
    md = base_metadata(cfg, tag, checks)
    md.update({
        "hierarchy_status": gate.status,
        "validity_status": "valid" if gate.status == GATE_FULL else gate.status,
        "invalid_reason": "; ".join(gate.reasons) if gate.reasons else "",
        "observed_gpu_occupancy": None,
        "repetition_index": 0,
    })
    md["_gate_aggregate"] = {"host_hit_delta": agg_host, "load_back_delta": agg_load}
    write_json_atomic(layout.metadata_path, md)
    summary = {
        "status": gate.status,
        "reportable": gate.status == GATE_FULL,
        "probe_ok": {k: v.ok for k, v in probes.items()},
        "state_groups": [e.to_dict() for e in gate.state_group_evidence],
        "reasons": gate.reasons,
        "validation_path": layout.validation_path,
    }
    write_json_atomic(os.path.join(layout.results_dir, "summary.json"), summary)
    logger.info("gate status=%s reportable=%s", gate.status, gate.status == GATE_FULL)
    logger.info("reasons: %s", "; ".join(gate.reasons) if gate.reasons else "(none)")


if __name__ == "__main__":
    sys.exit(main())
