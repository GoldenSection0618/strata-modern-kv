"""Tests for hcv.config: precedence, smoke protection, pairing, enums."""

from __future__ import annotations

import json
import os
import tempfile

import hcv.config as C


def _write_json(path: str, data: dict) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    return path


def test_defaults_are_frozen_baseline():
    cfg = C.load_config()
    assert cfg.hicache_io_backend == "direct"
    assert cfg.hicache_mem_layout == "page_first_direct"
    assert cfg.hicache_write_policy == "write_through"
    assert cfg.page_size == 64
    assert cfg.sglang_commit == C.PINNED_SGLANG_COMMIT
    assert cfg.sources["hicache_io_backend"] == "default"


def test_precedence_file_env_cli():
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = _write_json(os.path.join(tmp, "cfg.json"),
                               {"model": "qwen", "concurrency": 4, "n_repeats": 5})
        env = {"HCV_CONCURRENCY": "6"}
        cli = {"concurrency": 8, "n_repeats": 9}
        cfg = C.load_config(config_path=cfg_path, env=env, cli=cli)
        assert cfg.concurrency == 8          # cli wins over env and file
        assert cfg.n_repeats == 9            # cli wins over file
        assert cfg.sources["concurrency"] == "cli"
        # env alone wins over file
        cfg2 = C.load_config(config_path=cfg_path, env=env)
        assert cfg2.concurrency == 6
        assert cfg2.sources["concurrency"] == "env"
        # file wins over defaults
        cfg3 = C.load_config(config_path=cfg_path)
        assert cfg3.n_repeats == 5
        assert cfg3.sources["n_repeats"] == "file"


def test_smoke_does_not_overwrite_explicit_user_values():
    # user sets n_repeats explicitly via cli -> smoke must not touch it
    cfg = C.load_config(cli={"n_repeats": 7, "n_warmup": 3}, smoke=True,
                        explicit_fields={"n_repeats", "n_warmup"})
    assert cfg.n_repeats == 7
    assert cfg.n_warmup == 3
    assert cfg.smoke is True
    # defaults are reducible by smoke
    cfg2 = C.load_config(smoke=True)
    assert cfg2.n_repeats == 1
    assert cfg2.sources["n_repeats"] == "smoke"


def test_smoke_reduces_config_file_values():
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = _write_json(os.path.join(tmp, "cfg.json"),
                               {"n_repeats": 3, "n_warmup": 128, "family_size": 24})
        cfg = C.load_config(config_path=cfg_path, smoke=True)
        assert cfg.n_repeats == 1
        assert cfg.n_warmup == 1
        assert cfg.family_size == 16         # minimum stable smoke window
        assert cfg.sources["family_size"] == "smoke"


def test_smoke_uses_small_real_l1_pool_without_overwriting_explicit_value():
    cfg = C.load_config(cli={"experiment": "validation"}, smoke=True)
    assert cfg.max_total_num_tokens == 32768
    assert cfg.sources["max_total_num_tokens"] == "smoke"

    explicit = C.load_config(
        cli={"max_total_num_tokens": 65536},
        smoke=True,
        explicit_fields={"max_total_num_tokens"},
    )
    assert explicit.max_total_num_tokens == 65536
    assert explicit.sources["max_total_num_tokens"] == "cli"


def test_pressure_smoke_preserves_working_set_and_uses_real_pool():
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = _write_json(os.path.join(tmp, "exp2.json"), {
            "experiment": "exp2",
            "num_prefix_families": 16,
            "family_size": 24,
            "n_warmup": 128,
            "calib_probe_requests": 120,
            "n_repeats": 3,
        })
        cfg = C.load_config(config_path=cfg_path, smoke=True)
    assert cfg.n_repeats == 1
    assert cfg.num_prefix_families == 16
    assert cfg.family_size == 24
    assert cfg.n_warmup == 128
    assert cfg.calib_probe_requests == 120
    assert cfg.max_total_num_tokens == 0


def test_pinned_commit_enforced():
    try:
        C.load_config(cli={"sglang_commit": "deadbeef"})
        raise AssertionError("expected ValueError for unpinned commit")
    except ValueError:
        pass


def test_enum_validation():
    for bad in ("invalid_backend", "invalid_layout"):
        try:
            if bad == "invalid_backend":
                C.load_config(cli={"hicache_io_backend": bad})
            else:
                C.load_config(cli={"hicache_mem_layout": bad})
            raise AssertionError(f"expected ValueError for {bad}")
        except ValueError:
            pass


def test_paired_configs_differ_only_in_architecture():
    base = C.load_config()
    gpu, hier = C.pair_configs(base)
    C.assert_paired(gpu, hier)
    assert gpu.architecture == C.ARCH_GPU_ONLY
    assert hier.architecture == C.ARCH_HIERARCHICAL
    a, b = gpu.to_dict(), hier.to_dict()
    diffs = [k for k in a if a[k] != b.get(k)]
    assert set(diffs) <= C._PAIR_ALLOWED_DIFF


def test_pair_rejects_workload_drift():
    base = C.load_config()
    gpu, hier = C.pair_configs(base)
    hier.concurrency = hier.concurrency + 1  # non-architecture drift
    try:
        C.assert_paired(gpu, hier)
        raise AssertionError("expected ValueError for workload drift")
    except ValueError:
        pass


def test_build_server_argv():
    cfg = C.load_config()
    hier = C.load_config(cli={"architecture": "hierarchical",
                              "hicache_size_tokens": 1000000})
    argv_gpu = cfg.build_server_argv(port=3000)
    argv_hier = hier.build_server_argv(port=3001)
    assert "--enable-hierarchical-cache" not in argv_gpu
    assert "--enable-hierarchical-cache" in argv_hier
    assert "--hicache-io-backend" in argv_hier
    assert "--hicache-mem-layout" in argv_hier
    assert "--hicache-write-policy" in argv_hier
    assert "--page-size" in argv_gpu
    assert "--enable-metrics" in argv_gpu
    assert "--hicache-size" in argv_hier

    capped = C.load_config(cli={"max_total_num_tokens": 32768})
    argv_capped = capped.build_server_argv(port=3002)
    assert "--max-total-tokens" in argv_capped
    assert argv_capped[argv_capped.index("--max-total-tokens") + 1] == "32768"
    assert "--max-total-num-tokens" not in argv_capped


def test_model_registry():
    assert C.MODEL_SPECS["qwen"]["state_groups"] == ["attention_kv", "gated_delta_recurrent"]
    assert C.MODEL_SPECS["gemma"]["state_groups"] == ["local_sliding_window", "global_attention"]
    assert C.MODEL_SPECS["qwen"]["role"] == "primary"
    assert C.MODEL_SPECS["gemma"]["role"] == "secondary"


def test_unknown_field_rejected():
    try:
        C.load_config(cli={"not_a_field": 1})
        raise AssertionError("expected ValueError for unknown field")
    except ValueError:
        pass
