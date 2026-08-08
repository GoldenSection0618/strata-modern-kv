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

Work in progress. The experiment design for Modern KV / State Bottleneck Profiling is now specified; implementation, runtime validation, and measured results will be added incrementally.
