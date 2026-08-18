# SGLang / HiCache Execution Path (Exp1-3)

> Last verified: 2026-08-13. Environment, JIT toolchain, runner dry runs,
> and the three formal Qwen Exp3 residency modes are verified; only runs
> that pass the residency gates below may be reported.

This document describes the explicit SGLang execution path added for
Experiments 1-3, how it maps the three cache-residency conditions onto
SGLang's public interfaces, what evidence validates each condition, and
how the legacy vLLM path relates to it.

## 1. Architecture decision

SGLang is used **through its public server and HTTP/metrics boundaries
only**. The implementation:

- launches `python -m sglang.launch_server` as a child process
  (`sglang_hicache/server_lifecycle.py`, `LAUNCH_MODULE` pinned to the
  upstream `sglang.launch_server`);
- sends exact `input_ids` to the native `/generate` endpoint
  (`sglang_hicache/http_client.py`);
- scrapes public Prometheus metrics from `/metrics`
  (`sglang_hicache/metrics.py`);
- never imports SGLang internals and copies no SGLang code into this
  repository.

The local experiment package is deliberately named **`sglang_hicache`**
(never `sglang`) so that a child `python -m sglang.launch_server` started
from `code/` resolves the **upstream** SGLang package and is never
shadowed by the local experiment package.

The vLLM backend (`runners/vllm_runner.py`, `run_exp1.py` …) remains the
legacy/reference path with its already-collected recompute results. The
SGLang path shares workload construction rules and output schemas where
practical (identical summary keys, `raw/`/`summary.json`/`validation.json`/
`metadata.json` separation inside each unique run directory,
exp4-synthesis-compatible directory names).

## 2. Pinned runtime

| Item | Value |
|---|---|
| SGLang commit | `4ad990ba7d75bb9f948f5f6bd8d79a66b5d3fd63` |
| Source reference (read-only) | `~/yanglihan/dl-stack/projects/sglang` at `7120f3ee…` |
| Installed source | cached archive for the pinned `4ad990ba…` commit |
| Environment | `~/yanglihan/dl-stack/envs/sglang-hicache-cu129-torch211` |
| CUDA stack | Torch `2.11.0+cu129`; prefix nvcc `12.9.86`; prefix g++ `12.4.0` |
| Server entry | `python -m sglang.launch_server` |
| Metrics | `--enable-metrics --enable-cache-report` (mounts `/metrics` and enables public cache-source counters) |

Environment bootstrap and verification are documented in
`code/envs/sglang/SGLANG_ENV_PLAN.md` +
`code/envs/sglang/bootstrap_sglang_cu129_env.sbatch`; the standalone
`install_sglang_jit_toolchain.sbatch` captures the required C++20 toolchain.
Provenance is recorded via
`git rev-parse` + `git status` of the clone, the installed
`sglang.__version__`, and a full `pip freeze` snapshot (no checksums); each
run records the commit in `metadata.json` (`sglang_commit`). The bootstrap
**never runs `conda create` over an existing prefix** — an existing target
directory must be provably the exact completed environment or the job
fails without modifying it.

## 3. Residency conditions (exact meanings)

### `recompute`

- Server flag: `--disable-radix-cache` (radix cache off; HiCache off).
- Preparation: `POST /flush_cache`.
- Validation (`evaluate_recompute_evidence`): a repeated full prompt must
  report `meta_info.cached_tokens == 0`. Any cached tokens fail the gate.

### `gpu_hit`

- Server flags: default radix cache on; HiCache **off**.
- Preparation: flush, then warm the shared prefix with a prefix-only
  `/generate` (`max_new_tokens=1`).
- Validation (`evaluate_gpu_hit_evidence`): the measured request must match
  the prefix from device L1. Exact per-request metadata passes when available;
  for pinned Qwen3.5 hybrid, the authoritative fallback is one synchronous
  request surrounded by metric snapshots, with
  `prefill_effective_tokens_total{mode="device_hit"}` delta >= prefix length
  and no positive host-hit delta.

### `cpu_hit`

- Server flags: `--enable-hierarchical-cache` plus pinned `--hicache-*`
  flags (L1 GPU + L2 host).
- Preparation (`prepare_cpu_hit`): flush → warm prefix → create L1
  pressure with deterministic distinct filler KV until the GPU pool is
  full (budget read from `sglang:max_total_num_tokens` / `kv_used_tokens`
  metrics) so LRU eviction moves the warmed prefix from GPU to host L2.
- When pool-capacity metrics are absent, fallback pressure is
  `max(6*context_length, 262144) + protected_prefix_tokens + 4096`; the
  262,144 ceiling is conservatively above the observed A100 L1 capacity
  (`227,712` tokens). Under `write_through`, host occupancy rises before L1
  eviction, so it is observational evidence only and never an early stop.
- Validation (`evaluate_cpu_hit_evidence`): a PASS requires one synchronous
  measured request surrounded by metric snapshots, with
  `prefill_effective_tokens_total{mode="host_hit"}` delta >= prefix length
  and no positive device-hit delta. `cached_tokens_details` and
  `load_back_tokens_total` are retained as corroboration when exposed.
  Missing metrics remain `null`/unsupported, never a silent zero. A condition
  is **never** called `cpu_hit` based only on overall `cache_hit_rate`.

If the pinned implementation cannot expose sufficient evidence, the gate
fails and the run is labelled `unsupported` with an exact reason
(`unsupported.json`), never silently re-labelled or zero-filled.

## 4. Public metrics consumed

| Metric | Type | Use |
|---|---|---|
| `sglang:prefill_effective_tokens_total{mode=...}` | Counter | per-tier hit evidence (input/device_hit/host_hit/storage_hit) |
| `sglang:load_back_tokens_total{pool=...}` | Counter | H→D restore evidence |
| `sglang:hicache_backup_tokens_total{pool=...}` | Counter | D→H write-back evidence |
| `sglang:hicache_backup_bytes_total` / `load_back_bytes_total` | Counter | transfer bytes (bandwidth numerator) |
| `sglang:hicache_host_used_tokens` / `hicache_host_total_tokens` | Gauge | L2 pool occupancy/capacity |
| `sglang:cached_tokens_total{cache_source=...}` | Counter | cumulative cached tokens by tier |
| `sglang:cache_hit_rate` | Gauge | windowed overall hit rate (never used alone) |
| `sglang:kv_used_tokens` / `kv_available_tokens` / `kv_evictable_tokens` | Gauge | GPU KV pool absolute counts |
| `sglang:max_total_num_tokens` | Gauge | GPU pool capacity (filler budget) |
| `sglang:num_running_reqs` / `num_queue_reqs` | Gauge | scheduler load |
| `sglang:num_requests_total` / `gen_throughput` | Counter/Gauge | throughput context |
| `sglang:time_to_first_token_seconds` | Histogram | server-side TTFT aggregate |

**Missing-metric rule:** a metric absent from a scrape is recorded as
`null`/unsupported; only a metric that is actually present with value 0 is
recorded as 0. Deltas between before/after snapshots are `null` when
either side is missing (`sglang_hicache/metrics.py`).

## 5. Per-request response metadata

The native `/generate` response `meta_info` can carry (verified against the
pinned commit):

- `prompt_tokens`, `completion_tokens`, `cached_tokens`;
- `cached_tokens_details` = `{"device": X, "host": Y}` (plus optional
  `storage`/`storage_backend` when L3 is enabled) — the per-request
  tier breakdown;
- `output_token_logprobs` (with `return_logprob=True`) — the exact
  output token id used by the prefix-consistency check.

Known pinned-runtime boundary: for Qwen3.5 hybrid, native `/generate` reports
`cached_tokens=0` and omits `cached_tokens_details` even with
`--enable-cache-report`. The public per-tier counters do report the exact
device/host hit in an isolated request window (jobs `1273484`/`1273485`).
`return_cached_tokens_details` is an OpenAI-endpoint request option, not a
field of native `GenerateReqInput`; it must not be added to `/generate`.

## 6. Components

```text
code/sglang_hicache/
├── __init__.py
├── server_lifecycle.py   # child process, bounded readiness, graceful stop
├── http_client.py        # stdlib HTTP client (/generate, /flush_cache,
│                         #   /metrics, /health, /v1/tokenize, /model_info,
│                         #   /server_info)
├── metrics.py            # Prometheus parser + typed CacheStats snapshot
│                         #   + before/after deltas (missing != 0)
├── summary.py            # pure percentiles / load summaries / calibration
├── residency.py          # prep procedures + pure evidence evaluation
├── validation.py         # validation gate (same schema as vLLM path)
├── workload.py           # exact-token workload via server-side tokenizer
├── prefix_pool.py        # Exp3 prefix families + tier-dominance decisions
├── config.py             # server args + provenance metadata
├── io.py                 # raw/summary/validation/metadata writers + run tags
├── session.py            # server session orchestration
├── load_driver.py        # concurrent HTTP load driver (Exp3)
├── run_exp1.py / run_exp2.py / run_exp3.py
└── sbatch/               # ylh- sbatch + submit scripts (smoke vs full)
code/envs/sglang/         # canonical environment plan and executed bootstrap
code/tests/               # pure-Python unit tests (no CUDA/network/weights)
```

## 7. Results layout and analysis compatibility

| Experiment | Directory under `results/sglang/` |
|---|---|
| Exp1 | `exp1/<model>/<ctx>/<mode>/run-<tag>/` |
| Exp2 | `exp2/<model>/<ctx>-<pct>pct-<mode>/<mode>/run-<tag>/` |
| Exp3 | `exp3/<model>/<ctx_k>k-<pct>pct-<mode>/run-<tag>/` (e.g. `32k-50pct-cpu_hit`) |

**Every run gets a unique `run-<tag>` directory** so repeated runs never
overwrite raw files.  The tag is a UTC timestamp plus the SLURM job id
(`<YYYYmmddTHHMMSSZ>-job<id>`), with a user `RUN_TAG` override (passed as
`--run-tag`); it is recorded in `metadata.json` as `run_tag`.  Analysis
scripts discover results recursively (`rglob`), so the added level is
invisible to them.

Each run directory contains `metadata.json`, `validation.json`,
`summary.json`, and `raw/rep_*.json` (Exp1/2) or `raw/load_*_rep_*.json`
(Exp3) plus `calibration.json` (Exp3). Raw reps carry `"runtime": "sglang"`
and the same metric keys as the vLLM path, so
`analysis/exp1_analysis.py`, `analysis/exp2_analysis.py`, and
`analysis/exp4_synthesis.py` accept `results/sglang/exp{1,2,3}` inputs
unchanged. SGLang raw reps intentionally omit the vLLM GPU-memory
`gpu_analysis` object — restore evidence comes from authoritative SGLang
metrics instead.

## 8. Experiment 3 load path

### Concurrent load-back and window-start residency

For Exp3, `cpu_hit` classifies where a prefix resides at the beginning of
the isolated load window, not where it happens to reside when SGLang later
admits the request to a prefill batch. HiCache may restore several queued
requests asynchronously. Consequently, the first request can increment
`prefill_host_hit_tokens`, while later requests increment
`prefill_device_hit_tokens` after their H-to-D copies have already
completed.

The authoritative window aggregation is therefore:

1. `total_hit = prefill_device_hit_tokens + prefill_host_hit_tokens`;
2. `host_origin = max(prefill_host_hit_tokens,
   min(load_back_tokens_total, total_hit))`;
3. `device_origin = total_hit - host_origin`.

The cap is required because Qwen3.5 hybrid cache restoration can transfer a
small number of auxiliary Mamba state slots in addition to full-KV tokens.
Missing counters remain unsupported; they are never treated as zero.

This behavior was isolated on A100 with the same 12-family 32K workload:
job `1292807` (ceiling 4) recorded 81,920 hit tokens and 81,930 H-to-D
load-back tokens, while admission-time counters were 16,384 host plus
65,536 device. Job `1292863` (true ceiling 1) passed with host dominance,
confirming that the prefixes were correctly resident in L2 and that the
20/80 split was an admission-time artifact.

The final accepted smoke is job `1292918` on `smtg5001` with Qwen3.5-9B,
32K context, 50% prefix, 12 families, HiCache ratio 3,
`direct/page_first_direct`, and concurrency ceiling 4. It completed `0:0`;
validation, all seven calibration probes, and all seven formal points passed
the dominance gate. Results are under
`results/sglang/exp3/qwen/32k-50pct-cpu_hit/run-20260812T140012Z-job1292918/`.

Exp3 drives **concurrent HTTP requests** (`sglang_hicache/load_driver.py`
with a thread pool sized to the concurrency ceiling), never a synchronous
in-process engine. Per-request timing separates:

```text
TTFT = queueing + service time
queueing_ms = t_start - t_arrival
service_ms  = t_first_token - t_start   (t_first_token == t_complete for max_new_tokens=1)
```

**Prefix pool instead of one shared prefix.**  A single warmed shared
prefix would make the first request a host restore and every later
concurrent request a GPU hit — that is not a CPU-resident load point.
Exp3 therefore builds a deterministic **prefix pool**
(`sglang_hicache/prefix_pool.py`):

- `build_prefix_families` creates `prefix_pool_size` distinct prefix
  families (deterministic, seeded placement in the corpus);
- `PrefixPool.prompt_for`/`prompts` schedule families deterministically
  through the load window (round-robin cycle), so concurrent requests
  touch distinct prefixes;
- `prepare_gpu_hit_pool` warms **all** family prefixes into GPU L1 and
  records whether the bounded pool fits the observed L1 capacity
  (`fits_l1`; the pool size must be pinned so it fits for `gpu_hit`);
- `prepare_cpu_hit_pool` warms **all** family prefixes, then applies L1
  pressure (distinct filler KV) **after** warming so the formal load
  window contains many independently restorable host prefixes;
- every load-point rep records the per-request device/host tier breakdown
  (`tier_breakdown` in the raw rep and summary) and the aggregate number
  and ratio of device vs host hits.

**Dominance gate.**  A load point may only be reported under the
requested residency when that tier dominates by a documented threshold
(`hit_dominance_threshold`, default 0.8, pinned in metadata and sbatch as
`HIT_DOMINANCE_THRESHOLD`).  A mostly-GPU load point requested as
`cpu_hit` is labelled `unsupported` (`residency_dominance_ok=false` +
`residency_dominance_label="unsupported"` in the summary/raw JSON) and is
**never** silently called `cpu_hit`.  `prefix_pool_size` and the threshold
are pinned in `metadata.json` and the sbatch files.

Calibration estimates sustainable capacity (achieved/offered ≥ 0.85);
the formal sweep uses 7 normalized points (0.25x–1.30x). `FROZEN_RATES`
lets control conditions reuse the primary calibration.

## 9. L3 (storage) extension path

L3 storage is **out of scope** for Exp1-3. The same runner can later
enable it without structural changes:

1. server: add `--hicache-storage-backend <backend>` (e.g. `file`) and,
   when needed, `--hicache-storage-prefetch-policy wait_complete`;
2. server: optionally `--hicache-size <GB>` instead of `--hicache-ratio`;
3. client/validation: the response already carries
   `cached_tokens_details.storage` / `storage_backend`, and metrics
   already include `sglang:prefetch_*`, `sglang:backuped_tokens_total`,
   `sglang:prefetch_bandwidth`. Extend
   `evaluate_cpu_hit_evidence` with a storage-tier branch and record the
   backend in `engine_config_dict()`.

The relevant metric names and response fields already exist in the pinned
commit, so enabling L3 is a configuration + validation change, not a
rewrite.

## 10. Slurm execution

`code/sglang_hicache/sbatch/` provides complete `ylh-` sbatch files:

- `run_exp1_sglang.sbatch` / `run_exp2_sglang.sbatch` /
  `run_exp3_sglang.sbatch`;
- `submit_exp{1,2,3}_sglang.sh` (full sweep, and `smoke` phase with small
  settings);
- `--account=humx_lab`, A100 (`i56m512A100`) or L40 (`i80m512l40`)
  partitions, logs under `/share01/hpc/humxlab_intern/yanglihan/logs/`,
  `source "$HOME/yanglihan/env.sh"`, no `--wrap`;
- a pinned `SGLANG_COMMIT` is exported for provenance;
- **the dedicated `envs/sglang-hicache-cu129-torch211` environment is required** (or an explicit
  `SGLANG_ENV_DIR`/`PYTHON_BIN`); a missing dedicated env is a hard
  failure — there is **no fallback to the qwen python** (wrong dependency
  tree) — and a preflight `import sglang` check gives a clean error
  pointing to the env plan;
- **every job writes its own run-tagged output directory** via `RUN_TAG`
  (UTC timestamp + SLURM job id, user-overridable) so repeated runs never
  overwrite raw files.

Before server launch, each actual job verifies its allocated GPU with a BF16
matmul and forces CUDA JIT discovery to the prefix-local nvcc/g++ toolchain.
This turns a Slurm-visible but unhealthy GPU into an immediate, attributable
preflight failure and avoids the Rocky 8 system GCC 8.5 C++20 failure. A
`DRY_RUN=1` job checks the model path, environment markers, pinned commit,
imports, compiler executables, and log path without starting a server.

Note: the `#SBATCH -p` / `--gres` directives are fixed; the
`PARTITION`/`GPU` environment variables are honored **only** by the submit
scripts (they pass `-p`/`--gres` to `sbatch`).

### Cross-A100 handoff

The canonical prefix and checkpoints are shared under the user `/share01`
tree, so a new A100 node reuses them unchanged; no environment installation or
copy is permitted. The `DRY_RUN=1` mode checks the selected prefix, installed
commit, JIT markers, imports and paths, but it does not prove GPU health. A
node handoff must therefore follow dry run with one run-tagged actual smoke,
which performs the BF16 preflight before server launch.

Gemma 32K requires `--gres=gpu:2` on one A100 node with `TP_SIZE=2` and
`MEM_FRACTION=0.75`. If that smoke's `validation.json` is not
`all_passed=true`, stop and preserve the result/log directory; do not submit a
formal sweep. Passing a node smoke only establishes environmental readiness.
The Gemma cache-hit output-consistency and TP=2 Exp3 aggregation blockers
remain separate prerequisites; their exact rerun scope is in
`05-current-status.md`. The complete operator checklist and commands live in
the global `dl-stack/docs/02-环境配置.md`.

Hang policy: the server lifecycle waits for readiness with a bounded
timeout and shuts down gracefully on normal completion; a real hang is
handled by `scancel` at the job level (Slurm cgroup cleanup).

### Serialized formal Exp1/2 submission

`code/sglang_hicache/submit_exp1_exp2_serial.sh` submits the complete Qwen
Exp1 matrix (4 context lengths × 3 modes) followed by Exp2 (0% recompute and
4 non-zero prefix ratios × 3 modes). `submit_gemma_exp1_exp2_serial.sh` uses
the same serial dependency structure for Gemma, explicitly setting TP=2 and
`MEM_FRACTION=0.75` for the 32K workload. All jobs use HiCache ratio 3,
`direct` I/O, and `page_first_direct` layout. `ROOT_DEPENDENCY` may contain
multiple prerequisite gates, for example:

```bash
ROOT_DEPENDENCY=afterok:<recompute-gate-job>:<gpu-hit-gate-job> \
NODELIST=smtg5001 bash code/sglang_hicache/submit_exp1_exp2_serial.sh
```

The first Exp1 job waits for both gates; every later job depends on the
previous job with `afterok`. Thus all jobs can be queued in one operation
without running measurements concurrently on the node. The submitter records
the exact IDs and dependencies in
`results/sglang/exp12-pipelines/<pipeline-tag>/jobs.txt` and cancels only the
jobs created by its own invocation if submission fails part-way through.

## 11. Status

Implementation: **complete, unit-tested (pure-Python), and A100-validated**.
The canonical environment passed import, dependency, BF16, C++20, and CUDA
C++20 validation; all three runner dry runs passed.

**Qwen formal status.** Exp1 has 12/12 valid points and Exp2 has 13/13 valid
points. Exp3 primary and both frozen-rate controls, including strict gates,
passed; the primary seven-point sweep passed the residency-dominance gate.

**Gemma status and boundary.** A 32K workload requires 2×A100 TP=2 with
`MEM_FRACTION=0.75`; the single-A100 capacity is about 29,248 tokens. The
Gemma tokenizer's unrepresentable sentinel `model_max_length` causes public
`/v1/tokenize` to return 500, so the workload builder retains the public call
as its primary path and lazily falls back to the local checkpoint tokenizer
only for that failure. The TP=2 32K single-request smoke passed validation.
However, formal Gemma cache-hit points fail `prefix_consistency` (observed
tokens `805` versus `236779`), and TP=2 Exp3 load points lack aggregate
per-tier evidence and are therefore `unsupported`. Neither set is reportable.
The current result ledger and exact rerun scope are in `05-current-status.md`.

Failed attempts remain diagnostic evidence and are not measurements; no
SGLang number is reportable unless its `validation.json` is PASS (and Exp3
also passes its residency-dominance gate).
