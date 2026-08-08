# strata-modern-kv

Reproducing and re-evaluating Strata’s hierarchical KV cache and scheduling mechanisms on modern hybrid-attention LLMs and current GPU platforms.

## Goal

This repository studies whether the bottlenecks and optimizations identified by Strata remain important for modern LLM serving systems, especially under hybrid-attention architectures and newer GPU platforms.

The project is not intended to reproduce every original figure mechanically. Instead, it preserves the main causal questions from Strata and redesigns the evaluation around modern models, hardware, and workloads.

## Core questions

1. How much KV/cache-state pressure remains on modern hybrid-attention LLMs?
2. When does a CPU–GPU hierarchical cache still provide meaningful value?
3. How do cache granularity and I/O efficiency interact?
4. Under which workloads do locality-aware scheduling and stall-hiding mechanisms remain useful?
5. Do these mechanisms translate into end-to-end serving gains without short-context regressions?
6. Which conclusions generalize across different model architectures and GPU platforms?

## Planned evaluation

The evaluation is organized into six parts:

1. **Modern KV / state bottleneck profiling**
2. **Hierarchical cache value**
3. **Page granularity and GPU-assisted I/O**
4. **Cache locality and scheduler behavior**
5. **End-to-end serving**
6. **Model and hardware generalization**

See [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md) for the current experimental design.

## Initial model and hardware matrix

Representative modern models:

- Qwen3.5-9B
- Gemma 4 12B

Representative GPU platforms:

- NVIDIA A100 40GB
- NVIDIA L40 48GB

The full cross-product is reserved for representative generalization experiments rather than repeating every experiment on every configuration.

## Repository policy

The repository is intended to preserve a traceable experimental history. Published history on protected branches must not be rewritten, and generated artifacts or model weights should not be committed directly.

See [docs/REPOSITORY_RULES.md](docs/REPOSITORY_RULES.md).

## Status

Work in progress. Experimental code, configurations, and results will be added incrementally as the evaluation environment is stabilized.
