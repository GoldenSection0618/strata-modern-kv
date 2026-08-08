# Hierarchical Cache Experiments: Common Conventions

This document defines the shared validity rules for Experiments 1–4 in `hierarchical-cache-value`.

## 1. Execution roles

The default primary sweep target is **Qwen3.5-9B on A100 40GB** because it exercises both full-attention KV and Gated DeltaNet recurrent state.

This choice is conditional on the runtime validation gate in [`../../../docs/TECHNICAL_BASELINE.md`](../../../docs/TECHNICAL_BASELINE.md). If the pinned runtime cannot verify full CPU offload/restore for every Qwen3.5 state group, Qwen3.5 must not be used to claim full hierarchical-cache behavior.

In that case, Gemma 4 12B becomes the primary full-hierarchy target and Qwen3.5 is reported as `partial` or `unsupported` until the missing path is validated.

Experiments 1–3 perform the detailed sweeps on one validated primary model. Experiment 4 performs a small matched validation on the second model. The separate project-level Model and Hardware Generalization group reuses these A100 results and adds representative L40 runs rather than repeating the same A100 experiments.

## 2. Workload scope

All experiments in this group use **text-only** serving requests.

Unless an experiment explicitly changes a variable, the following remain fixed within a comparison:

- model revision and runtime commit;
- precision and cache dtype;
- context-length distribution;
- output-length distribution;
- request count and offered load;
- request ordering;
- prefix lengths;
- scheduler policy;
- GPU cache budget;
- CPU offload capacity and policy for hierarchical runs.

The workload trace must be versioned or deterministically reproducible from a recorded seed/configuration.

## 3. Cache architecture definitions

### GPU-only

Prefix caching remains enabled with the same cache/block/state policy used by the paired hierarchical run, but no CPU offload tier is enabled.

Reusable state may remain in the GPU cache. Once reusable state is evicted from GPU, it is unavailable to later requests and the missing prefix state must be recomputed.

### GPU + CPU hierarchical

The GPU budget, prefix-cache policy, block/state checkpointing mode, cache dtype, workload, and scheduler policy are identical to the paired GPU-only run. The intended architecture difference is that a validated CPU offload tier is enabled.

Reusable state evicted from GPU may be retained in the CPU tier and restored on a later hit.

The CPU tier must be large enough that CPU-capacity pressure is not an uncontrolled variable in Experiments 1–3. If CPU eviction occurs, it must be measured and the run must be labeled accordingly.

The CPU offloading backend and its configuration remain pinned across all hierarchical runs that are compared with each other.

## 4. Hierarchy validity gate

A run is valid as a **full hierarchical-cache** result only when all state required to skip the claimed prefix computation is restored correctly.

Validation requires:

- numerical consistency with recomputation;
- observable GPU/CPU residency behavior;
- per-state-group restore coverage where the runtime exposes multiple groups;
- identical prefix-cache/block/cache-dtype/scheduler settings across paired GPU-only and hierarchical runs;
- a pinned CPU offloading backend across hierarchical runs;
- no silent fallback from restore to recomputation;
- no active-request preemption caused by the cache-budget sweep.

For Qwen3.5, full-hierarchy validation includes both attention KV and Gated DeltaNet recurrent state. For Gemma 4, it includes the retained local/sliding-window and global-attention cache groups used by the pinned runtime.

A partial state restore is reported as `partial hierarchy`; it is not merged into the full-hierarchy curve.

## 5. Cache pressure

`GPU cache pressure` refers to pressure on reusable cache/state after maintaining enough capacity for the fixed active-request workload.

Before Experiment 2, a calibration run estimates the cache requirement needed to execute the chosen request load without scheduler preemption. Pressure points are then selected by reducing the remaining capacity available to the reusable working set.

Every pressure point must be validated using observed GPU hit/eviction behavior. A configured memory fraction alone is not sufficient evidence that a point is Low, Medium, or High pressure.

Runs with OOM, scheduler preemption, or a changed effective concurrency are invalid for the main pressure curve unless that behavior is explicitly analyzed separately.

## 6. Prefix reuse

Experiment 3 isolates **reuse opportunity**, not locality.

The primary reuse variable is the fraction of eligible requests that revisit an existing prefix, preferably reported both request-weighted and token/state-volume-weighted.

Across reuse levels, the experiment keeps prefix length, total requests, input/output token distribution, request-rate condition, and the placement/reuse-distance pattern of eligible revisit slots fixed. Lower-reuse traces replace some revisits with matched unique prefixes at the same positions.

Hotspot concentration, cache distance, and request reordering are not changed in Experiment 3. Those belong to the separate cache-locality/scheduler experiment group.

## 7. Core measurements

At minimum, every paired run records:

- GPU cache hit volume;
- CPU cache hit volume when hierarchy is enabled;
- GPU eviction volume;
- recomputed token/state volume or an equivalent verified computation measure;
- CPU-GPU transfer volume and transfer activity;
- non-overlapped restore stall when measurable;
- TTFT distribution;
- steady-state throughput;
- active-request preemption count;
- actual offered and achieved request rate.

Hit counts should be supplemented with token/state volume when possible because a request-level hit can represent very different amounts of saved computation.

## 8. Statistical execution

Each configuration uses multiple independent repetitions after non-measured initialization.

Paired GPU-only and hierarchical configurations use the same workload trace. Their execution order should be alternated or randomized so machine drift does not systematically favor one architecture.

Reported results preserve raw measurements and include a center statistic plus variability. Tail-latency claims require enough samples to support the reported percentile.

## 9. Interpretation rule

The intended evidence chain is:

```text
reuse opportunity
      +
GPU capacity pressure
      ↓
GPU miss / eviction of reusable state
      ↓
validated CPU-tier hit
      ↓
avoided recomputation
      ↓
CPU-GPU transfer and non-overlapped stall
      ↓
TTFT / throughput change
```

If the chain breaks at any stage, the conclusion must identify that stage rather than treating a lack of end-to-end gain as a generic failure of hierarchical caching.
