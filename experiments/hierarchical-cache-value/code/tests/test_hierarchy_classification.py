"""Tests for hcv.hierarchy: gate classification, group evidence, pool maps."""

from __future__ import annotations

from hcv.config import ARCH_HIERARCHICAL, GATE_FULL, GATE_INVALID_INFRA, GATE_PARTIAL, GATE_UNSUPPORTED
from hcv.hierarchy import (
    PROBE_CPU_HIT,
    PROBE_GPU_HIT,
    PROBE_GPU_ONLY_EVICTION,
    PROBE_RECOMPUTE,
    ProbeResult,
    build_group_evidence,
    classify_gate,
    map_pool_to_group,
)

PREFIX = 512

QWEN_GROUPS = ["attention_kv", "gated_delta_recurrent"]


def _probe(name, prefix_length=PREFIX, **kwargs):
    kw = dict(prefix_length=prefix_length, ok=True, reason="",
              input_delta=None, device_hit_delta=None, host_hit_delta=None,
              load_back_delta=None, pool_deltas={}, deltas={})
    kw.update(kwargs)
    return ProbeResult(name=name, **kw)


def _ok_probes(with_pools=False):
    pool_deltas = {"kv": PREFIX, "mamba": PREFIX} if with_pools else {}
    return {
        PROBE_RECOMPUTE: _probe(PROBE_RECOMPUTE, input_delta=PREFIX + 32,
                                device_hit_delta=0.0, host_hit_delta=0.0),
        PROBE_GPU_HIT: _probe(PROBE_GPU_HIT, input_delta=32.0,
                              device_hit_delta=PREFIX, host_hit_delta=0.0),
        PROBE_GPU_ONLY_EVICTION: _probe(PROBE_GPU_ONLY_EVICTION, input_delta=PREFIX + 32,
                                        device_hit_delta=0.0, host_hit_delta=0.0),
        PROBE_CPU_HIT: _probe(PROBE_CPU_HIT, input_delta=32.0, device_hit_delta=0.0,
                              host_hit_delta=PREFIX, load_back_delta=PREFIX,
                              pool_deltas=pool_deltas),
    }


def test_full_only_with_per_group_evidence():
    probes = _ok_probes(with_pools=True)
    gate = classify_gate("qwen", ARCH_HIERARCHICAL, QWEN_GROUPS, probes,
                         aggregate_host_hit_delta=PREFIX,
                         aggregate_load_back_delta=PREFIX)
    assert gate.status == GATE_FULL
    assert gate.reportable
    groups = {e.group: e.observable for e in gate.state_group_evidence}
    assert groups["attention_kv"] is True
    assert groups["gated_delta_recurrent"] is True


def test_aggregate_only_is_partial_never_full():
    # cpu hit works but the runtime exposes no per-pool breakdown
    probes = _ok_probes(with_pools=False)
    gate = classify_gate("qwen", ARCH_HIERARCHICAL, QWEN_GROUPS, probes,
                         aggregate_host_hit_delta=PREFIX,
                         aggregate_load_back_delta=PREFIX)
    assert gate.status == GATE_PARTIAL
    assert not gate.reportable
    for e in gate.state_group_evidence:
        assert e.observable is False
        assert "aggregate-only" in e.note


def test_missing_cpu_hit_is_unsupported_when_no_host_evidence():
    probes = _ok_probes(with_pools=True)
    probes[PROBE_CPU_HIT] = _probe(PROBE_CPU_HIT, ok=False,
                                   reason="host hit delta None < prefix length",
                                   input_delta=PREFIX, device_hit_delta=0.0,
                                   host_hit_delta=None, load_back_delta=None)
    gate = classify_gate("qwen", ARCH_HIERARCHICAL, QWEN_GROUPS, probes,
                         aggregate_host_hit_delta=None, aggregate_load_back_delta=None)
    assert gate.status == GATE_UNSUPPORTED


def test_partial_when_some_host_evidence_but_probe_fails():
    probes = _ok_probes(with_pools=True)
    probes[PROBE_CPU_HIT] = _probe(PROBE_CPU_HIT, ok=False,
                                   reason="input delta >= prefix length (fallback)",
                                   input_delta=PREFIX + 32, device_hit_delta=0.0,
                                   host_hit_delta=PREFIX, load_back_delta=PREFIX)
    gate = classify_gate("qwen", ARCH_HIERARCHICAL, QWEN_GROUPS, probes,
                         aggregate_host_hit_delta=PREFIX,
                         aggregate_load_back_delta=PREFIX)
    assert gate.status == GATE_PARTIAL


def test_invalid_infrastructure():
    probes = {}
    gate = classify_gate("qwen", ARCH_HIERARCHICAL, QWEN_GROUPS, probes,
                         None, None,
                         infra_checks={"env_ok": False,
                                       "env_error": "sglang_commit.txt mismatch"})
    assert gate.status == GATE_INVALID_INFRA
    assert any("commit" in r for r in gate.reasons)


def test_missing_probe_is_unsupported():
    probes = {PROBE_RECOMPUTE: _probe(PROBE_RECOMPUTE)}
    gate = classify_gate("qwen", ARCH_HIERARCHICAL, QWEN_GROUPS, probes,
                         None, None)
    assert gate.status == GATE_UNSUPPORTED


def test_pool_mapping():
    assert map_pool_to_group("kv") == "attention_kv"
    assert map_pool_to_group("mamba") == "gated_delta_recurrent"
    assert map_pool_to_group("swa") == "local_sliding_window"
    assert map_pool_to_group("global") == "global_attention"
    assert map_pool_to_group("unknown_pool") == "pool:unknown_pool"


def test_group_evidence_pool_split():
    ev = build_group_evidence(QWEN_GROUPS, {"kv": 100.0, "mamba": 50.0}, 150.0, 150.0)
    by_group = {e.group: e for e in ev}
    assert by_group["attention_kv"].restore_tokens == 100.0
    assert by_group["gated_delta_recurrent"].restore_tokens == 50.0
    assert all(e.observable for e in ev)


def test_gemma_kv_pool_maps_to_required_global_attention_group():
    ev = build_group_evidence(
        ["local_sliding_window", "global_attention"],
        {"kv": 512.0, "swa": 512.0},
        512.0,
        1024.0,
    )
    by_group = {e.group: e for e in ev}
    assert by_group["local_sliding_window"].restore_tokens == 512.0
    assert by_group["global_attention"].restore_tokens == 512.0
    assert all(e.observable for e in ev)


def test_serial_probe_expectations_via_evaluate():
    from hcv.probes import evaluate_probe

    p = _probe(PROBE_RECOMPUTE, input_delta=PREFIX + 10,
               device_hit_delta=0.0, host_hit_delta=0.0)
    assert evaluate_probe(PROBE_RECOMPUTE, p, PREFIX).ok
    # host hit + full recompute = silent fallback -> fail
    bad = _probe(PROBE_CPU_HIT, input_delta=PREFIX + 10, device_hit_delta=0.0,
                 host_hit_delta=PREFIX, load_back_delta=PREFIX)
    assert not evaluate_probe(PROBE_CPU_HIT, bad, PREFIX).ok
    assert "fallback" in bad.reason


def test_missing_required_zero_counter_fails_probe():
    from hcv.probes import evaluate_probe

    p = _probe(PROBE_GPU_HIT, input_delta=32.0, device_hit_delta=PREFIX,
               host_hit_delta=None)
    assert not evaluate_probe(PROBE_GPU_HIT, p, PREFIX).ok


def test_gate_result_serializes_window_objects():
    from hcv.hierarchy import GateResult

    class Window:
        def to_dict(self):
            return {"window_index": 0, "origin": "device_resident"}

    gate = GateResult(
        status=GATE_PARTIAL,
        model="qwen",
        architecture=ARCH_HIERARCHICAL,
        concurrent_windows=[Window()],
    )
    assert gate.to_dict()["concurrent_windows"] == [
        {"window_index": 0, "origin": "device_resident"}
    ]


def test_cpu_hit_accepts_effective_per_pool_load_back():
    from hcv.probes import evaluate_probe

    probe = _probe(
        PROBE_CPU_HIT,
        input_delta=64.0,
        device_hit_delta=0.0,
        host_hit_delta=PREFIX,
        load_back_delta=PREFIX + 2.0,
        pool_deltas={"kv": PREFIX, "mamba": 2.0},
    )
    assert evaluate_probe(PROBE_CPU_HIT, probe, PREFIX).ok
