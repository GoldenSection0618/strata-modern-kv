# strata-modern-kv

Re-evaluating Strata's hierarchical context caching and scheduling mechanisms on modern hybrid LLMs and current GPU platforms.

## Goal

This repository studies whether the bottlenecks and optimizations identified by Strata remain important for modern LLM serving systems, especially when the model no longer uses a uniform dense-attention KV cache.

The project is not intended to reproduce every original figure mechanically. It preserves the main causal questions from Strata and redesigns the evaluation around modern models, hardware, workloads, and cache/state representations.

## Core questions

1. How much KV/recurrent-state pressure remains on modern hybrid LLMs?
2. When does a CPU-GPU hierarchical cache still provide meaningful value?
3. How do cache granularity and I/O efficiency interact?
4. Under which workloads do locality-aware scheduling and stall-hiding mechanisms remain useful?
5. Do these mechanisms translate into end-to-end serving gains without short-context regressions?
6. Which conclusions remain stable across models and GPU platforms?

## Planned evaluation

The evaluation is organized into six parts:

1. **Modern KV / state bottleneck profiling**
2. **Hierarchical cache value**
3. **Page granularity and GPU-assisted I/O**
4. **Cache locality and scheduler behavior**
5. **End-to-end serving**
6. **Model and hardware generalization**

See [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md) for the current experiment-group design.

The detailed design for the first group is under [experiments/modern-kv-state-bottleneck/](experiments/modern-kv-state-bottleneck/).

## Model and hardware baseline

Primary models:

- `Qwen/Qwen3.5-9B`
- `google/gemma-4-12B` or the matching instruction-tuned checkpoint selected by the serving setup

Primary GPU platforms:

- NVIDIA A100 40GB
- NVIDIA L40 48GB

Qwen3.5 combines Gated DeltaNet recurrent/linear-attention layers with full attention, while Gemma 4 combines sliding-window and global attention. Their cache/state objects therefore must not be treated as interchangeable ordinary KV caches.

The full model × hardware cross-product is reserved for representative generalization configurations rather than repeating every experiment four times.

Volatile architecture and runtime assumptions are recorded in [docs/TECHNICAL_BASELINE.md](docs/TECHNICAL_BASELINE.md). Exact checkpoint revisions, software versions, and cache policies must be pinned in the metadata of every reported experiment.

## Repository policy

The repository preserves a traceable experimental history. Published history on `main` must not be rewritten to hide earlier mistakes, and generated artifacts or model weights should not be committed directly.

See [docs/REPOSITORY_RULES.md](docs/REPOSITORY_RULES.md).

## Status

Work in progress (last updated 2026-08-09).

Current state of the Modern KV / State Bottleneck Profiling group:

- **Design**: all four experiment designs specified (`docs/EXPERIMENT_PLAN.md`, `experiments/modern-kv-state-bottleneck/docs/00-06`).
- **Code**: Experiment 1-3 implemented on two execution paths:
  - **vLLM 0.26.0 (legacy/reference)** — `experiments/modern-kv-state-bottleneck/code/run_exp{1,2,3}.py`;
  - **SGLang / HiCache (explicit path)** — `experiments/modern-kv-state-bottleneck/code/sglang_hicache/` (public HTTP + Prometheus boundaries only; installed commit `4ad990ba7d75bb9f948f5f6bd8d79a66b5d3fd63`).
  Experiment 4 is design + analysis-code only (synthesis of 1-3).
- **Runtime fixes (vLLM 0.26.0 / A100 40GB)**:
  - `VLLM_USE_FLASHINFER_SAMPLER=0` — flashinfer sampler JIT fails on this CUDA/CUB (cub 300302 `BlockAdjacentDifference` lacks `FlagHeads`).
  - `VLLM_WORKER_MULTIPROC_METHOD=spawn` — without it vLLM races between fork/spawn detection and crashes with "Cannot re-initialize CUDA in forked subprocess".
- **Measured results (Qwen3.5-9B, A100, vLLM path)**: recompute baseline complete — exp1: 4K/8K/16K/32K; exp2: 0%/25%/50%/75%/87.5% prefix ratio (10 reps each, median/P90 TTFT in `results/exp{1,2}/`).
- **Known blocker (vLLM path)**: `gpu_hit` / `cpu_hit` conditions produce no data. `VLLMStatsCollector` cannot reach `KVCacheManager`/prefix-cache counters in the vLLM 0.26 V1 engine (engine internals live in the EngineCore subprocess), so the validation gate always reports `queries=0, hits=0` and measurements are aborted. Recompute mode is unaffected (cache-hit checks skipped).
- **SGLang / HiCache path**: implementation and canonical CUDA 12.9 / Torch 2.11 environment are validated. Qwen formal data are complete: Exp1 12/12, Exp2 13/13, and Exp3 primary + controls pass. Gemma runs at 32K require 2×A100 TP=2: Exp1 currently has 10/12 valid points, Exp2 5/13, and Exp3 has only a valid single-request smoke; failed cache-hit and unsupported TP=2 load points are preserved but not reported. See `experiments/modern-kv-state-bottleneck/docs/05-current-status.md`.

Remaining work: resolve Gemma's 32K cache-hit output-consistency failure and TP=2 Exp3 tier-metric aggregation, rerun only the affected Gemma points, then run Experiment 4 cross-model synthesis. The vLLM hit-mode stats blocker remains legacy-path work.
