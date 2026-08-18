# Code — Hierarchical Cache Value evaluation

Implementation of the four experiments in `hierarchical-cache-value` plus
the full-hierarchy capability gate. All runtime interaction uses the
public SGLang HTTP surface (`/health`, `/generate`, `/flush_cache`,
`/metrics`, `/get_model_info`); the local package is named `hcv` so it
never shadows the upstream `sglang` package (`python -m
sglang.launch_server` always resolves the installed upstream module).

## Frozen runtime baseline (this group)

| Setting | Value |
|---|---|
| environment prefix | `$DL_ROOT/envs/sglang-hicache-cu129-torch211` |
| SGLang commit | `4ad990ba7d75bb9f948f5f6bd8d79a66b5d3fd63` |
| SGLang version | `0.5.6.post3.dev8468+g4ad990ba7` |
| I/O backend | `direct` (GPU-assisted `kernel` I/O belongs to group 3) |
| host layout | `page_first_direct` |
| write policy | `write_through` |
| page size | 64 |
| metrics | public `/metrics` enabled |
| precision | bfloat16 |
| GPU-only vs hierarchical difference | hierarchy enablement + CPU tier only |

No fallback to `envs/sglang`, `envs/sglang-cu129`, `qwen` or `gemma4`.

## Layout

```text
code/
├── hcv/                  # local package (stdlib only)
│   ├── config.py         # frozen runtime config, precedence, pairing checks
│   ├── metrics.py        # public Prometheus parser, snapshots, deltas (missing=None)
│   ├── http_client.py    # /health /generate /flush_cache /metrics /get_model_info
│   ├── server_lifecycle.py  # bounded server launch/ready/shutdown
│   ├── workload.py       # deterministic trace construction + invariants
│   ├── filler.py         # L1 pressure budgeting (observed capacity / fallback)
│   ├── hierarchy.py      # full/partial/unsupported/invalid-infra classification
│   ├── probes.py         # serial probes with isolated before/after counters
│   ├── residency.py      # cold/warm preparation + observed cache state
│   ├── load_driver.py    # concurrent windows + origin accounting
│   ├── provenance.py     # env markers, JIT toolchain, runtime identity
│   ├── schema.py         # run layout + raw/processed/result schemas
│   ├── run_common.py     # shared runner scaffolding
│   ├── run_validation.py # full-hierarchy gate runner
│   ├── calibrate.py      # Exp2 calibration (min GPU budget floor)
│   ├── run_exp1.py       # GPU-only vs hierarchical x cold/warm
│   ├── run_exp2.py       # pressure points (Low/Medium/High/[VeryHigh])
│   ├── run_exp3.py       # reuse sweep (fixed slots, matched unique prefixes)
│   ├── run_exp4.py       # frozen V0/V1/V2 cross-model validation
│   ├── analysis.py       # deterministic raw -> processed -> results
│   ├── dry_run.py        # DRY_RUN=1 static validation (no server)
│   └── preflight.py      # BF16 + C++20/CUDA JIT preflight (compute node)
├── configs/              # per-experiment JSON configs
├── sbatch/               # ylh-hcv-* sbatch files + submit scripts
└── tests/                # pure-Python unit tests (run via Slurm)
```

## Key invariants

* Missing metrics are `null`/unsupported, never zero; deltas propagate
  `None`.
* Pinned Qwen hybrid native `/generate` may omit
  `cached_tokens_details`; isolated before/after per-tier counters are
  authoritative.
* Serial CPU-hit validation: host-hit delta ≥ prefix length, zero
  device-hit delta, input-path delta < prefix length (no silent
  restore-to-recompute fallback). GPU hit: device-hit delta ≥ prefix
  length, zero host-hit delta.
* Concurrent windows classify origin as
  `host_origin=max(host_hit,min(load_back,total_hit))` and
  `device_origin=total_hit-host_origin`; missing load-back evidence stays
  unknown rather than becoming zero.
* Filler early-stop uses L1 evidence only (`kv_evictable`/`kv_available`);
  under write-through, host occupancy grows before L1 eviction and is
  never a stop condition. Pressure = observed L1 capacity when available,
  else `max(6*context_length, 262144) + protected_prefix_tokens + 4096`.
* Full Qwen hierarchy requires per-observable-state-group restore
  evidence for attention KV AND Gated DeltaNet recurrent state;
  aggregate-only evidence classifies `partial`, never `full`.
* `validation.json` controls reportability; a Slurm `COMPLETED` alone
  never validates a result.
* Paired GPU-only/hierarchical configs differ only in hierarchy
  enablement + CPU tier (checked by `assert_paired`).
* Raw data is never overwritten: every run uses a unique
  `run-<UTC>-job<id>` directory.
* User overrides are resolved once (defaults < config file < `HCV_*` env
  < CLI) and are never silently overwritten by `SMOKE=1`.

## Usage

All heavy work runs through complete `ylh-hcv-*` sbatch files on the
compute node. The submit scripts are thin wrappers that export
`HCV_CODE_DIR` and pass user overrides.

```bash
cd experiments/hierarchical-cache-value/code/sbatch

# --- staged validation order (recommended) -----------------------------
# 1. dry-run: static validation, no server; submit as a minimal Slurm job
DRY_RUN=1 bash submit_validation.sh

# 2-4. hierarchy gate in smoke mode: exercises minimal recompute, serial
#      GPU hit, then serial CPU hit and the full/partial/unsupported
#      classification (small filler in smoke)
SMOKE=1 bash submit_validation.sh          # MODEL=qwen default
#     after COMPLETED, inspect:
#     results/validation/run-*/validation.json   -> status must be full

# 5. one formal point, concurrency ceiling 1 vs 4 (measurement semantics
#    before committing to the full formal sweep)
CONCURRENCY=1 CELL_LIMIT=1 bash submit_exp1.sh
CONCURRENCY=4 CELL_LIMIT=1 bash submit_exp1.sh

# 6. complete smoke of all experiments (still SMOKE=1)
SMOKE=1 bash submit_exp1.sh
SMOKE=1 bash submit_exp2.sh                # calibration + Low/Medium/High
SMOKE=1 bash submit_exp3.sh                # reuse 0.00/0.25/0.50/0.75
SMOKE=1 bash submit_exp4.sh                # freeze + run (MODEL=gemma)

# 7. full experiments
bash submit_exp1.sh
bash submit_exp2.sh
bash submit_exp3.sh
bash submit_exp4.sh

# --- validation / analysis / tests -------------------------------------
bash submit_analysis.sh tests              # pure-Python unit tests (Slurm)
bash submit_analysis.sh analysis           # raw -> processed -> results

# --- per-experiment knobs ----------------------------------------------
# exp1: MODEL=qwen|gemma, ARCH=gpu_only|hierarchical, STATE=cold|warm
# exp2: ARCH, PRESSURE=Low|Medium|High|VeryHigh (VeryHigh optional)
# exp3: ARCH, REUSE=0.00|0.25|0.50|0.75, PRESSURE=fixed exp2 label
# exp4: MODEL=gemma (secondary); freeze phase then run phase
```

### DRY_RUN=1

Validates exact environment markers (six marker files + pinned commit),
prefix-local compiler executables, model path, log path, config
precedence (recorded per-field sources), resolved smoke/formal values,
server argv for both architectures, the GPU-only/hierarchical pairing
invariant, and the `ylh-hcv-*` job-name prefix when inside a job. It
never starts a server, never imports torch, and never touches a GPU.
Dry-run success is environment conformance only — it is NOT
full-hierarchy proof (see `hcv.hierarchy`).

### Preflight (every real sbatch)

Before server launch each sbatch runs `hcv.preflight`: a real BF16
matmul on the GPU plus a C++20 + CUDA kernel compile/run with the
prefix-local `nvcc`/`gcc`/`g++`, and `hcv.provenance check` verifies the
marker files and pinned commit. Failures abort the job before the server
starts.

## Results

Raw runs land in `results/<experiment>/run-<UTC>-job<id>/` with
`metadata.json`, `validation.json` (gate outcome), `raw/measurements.jsonl`
(per-request/window/filler/calibration evidence), `raw/snapshots.jsonl`
and `server/` logs. `hcv.analysis` deterministically produces
`processed_out/processed/*.json` and `processed_out/results/*.csv`
without touching raw data.

## Tests

`tests/run_tests.py` runs all `test_*` functions in `tests/test_*.py`
(no pytest dependency, no server, no GPU, no torch import). Run only via
Slurm: `bash submit_analysis.sh tests`.
