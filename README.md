# strata-modern-kv

Re-evaluating Strata's hierarchical context caching, I/O, and cache-aware scheduling mechanisms on modern hybrid LLMs and the GPU platforms available to this project.

## Goal

This repository studies whether the bottlenecks and optimizations identified by Strata remain important for modern LLM serving systems, especially when models no longer use a uniform dense-attention KV cache.

The project does not reproduce every original figure mechanically. It preserves the main causal questions from Strata and redesigns the evaluation around modern models, available hardware, workload structure, cache/state representations, and current serving-runtime constraints.

## Core questions

1. How much KV/recurrent-state pressure remains on modern hybrid LLMs?
2. When does a CPU-GPU hierarchical cache still provide meaningful value?
3. How do cache granularity and I/O efficiency interact?
4. Under which cache-distance, arrival-pressure, and same-context-overlap conditions do Strata-style scheduler mechanisms remain useful?
5. Do the validated mechanisms translate into end-to-end serving gains without short-context regressions?
6. Which conclusions remain stable across models and GPU platforms?

See [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md) for the project-level evaluation design.

## Planned evaluation

The evaluation is organized into six groups:

1. **Modern KV / state bottleneck profiling**
2. **Hierarchical cache value**
3. **Page granularity and GPU-assisted I/O**
4. **Cache locality and scheduler behavior**
5. **End-to-end serving**
6. **Model and hardware generalization**

Detailed designs currently available:

- [Modern KV / state bottleneck profiling](experiments/modern-kv-state-bottleneck/)
- [Hierarchical cache value evaluation](experiments/hierarchical-cache-value/)
- [Page granularity and GPU-assisted I/O](experiments/page-granularity-gpu-assisted-io/)
- [Cache locality and scheduler behavior](experiments/cache-locality-scheduler-behavior/)

## Model and hardware baseline

Primary model families:

- `Qwen/Qwen3.5-9B`
- `google/gemma-4-12B-it`

Primary GPU platform:

- NVIDIA A100 40GB

Representative hardware-generalization platform:

- NVIDIA L40 48GB

Qwen3.5 combines Gated DeltaNet recurrent/linear-attention layers with full-attention layers. Gemma 4 12B Unified combines local sliding-window attention with full global attention and unified Keys/Values in global layers. Their cache/state objects must not be treated as interchangeable ordinary dense-attention KV caches.

The full model × hardware cross-product is reserved for representative generalization configurations rather than repeating every experiment four times. Earlier A100 cross-model results are reused when the configuration is identical.

Volatile architecture, runtime, hardware, cluster-software, granularity, and scheduler-semantics assumptions are maintained in [`docs/TECHNICAL_BASELINE.md`](docs/TECHNICAL_BASELINE.md). Exact checkpoint revisions, runtime versions/commits, resolved defaults, cache policies, scheduler mechanisms, and capability status must be pinned in every reported experiment.

A configured offload path is not automatically considered a valid full hierarchy. Hybrid-model experiments must verify that every state group needed to skip the claimed recomputation is correctly restored.

Likewise, a current runtime option with a scheduler-related name is not automatically equivalent to Strata's delay-hit deferral, balanced batching, or bubble filling. Scheduler attribution requires semantic validation of the actual implementation.

## Runtime discipline

Runtime selection follows the mechanism being evaluated rather than assuming one engine is suitable for every experiment.

The Page Granularity and GPU-Assisted I/O group uses SGLang HiCache as the preferred mechanism candidate because it exposes explicit cache-page and direct/kernel I/O controls. This path remains conditional on establishing a non-Docker CUDA-12-compatible build for the current A100 cluster and passing the experiment-specific runtime/state validation gate.

The scheduler group uses Strata §4.3 as its mechanism reference. It separates short-distance/high-overlap delay-hit behavior from longer-distance host-loading pressure instead of assuming that all scheduler pathologies monotonically worsen as locality decreases.

## Repository policy

The repository preserves a traceable experimental history. Published history on `main` must not be rewritten to hide earlier mistakes, and generated artifacts or model weights should not be committed directly.

See [`docs/REPOSITORY_RULES.md`](docs/REPOSITORY_RULES.md).

## Status

Work in progress.

Detailed experiment designs are currently specified for the first four groups:

- Modern KV / State Bottleneck Profiling;
- Hierarchical Cache Value Evaluation;
- Page Granularity and GPU-Assisted I/O;
- Cache Locality and Scheduler Behavior.

Implementation, pinned-runtime validation, and measured results will be added incrementally. End-to-End Serving and Model/Hardware Generalization remain at the project-plan level and should receive experiment-specific designs before implementation begins.
