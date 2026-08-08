# Experiment Plan

## 1. Evaluation objective

The objective is to re-evaluate the main systems claims behind Strata under modern hybrid LLMs and the GPU platforms available to this project.

The experiment suite is organized around causal questions rather than one-to-one figure reproduction. Each group must be broad enough to support a credible conclusion while avoiding repeated measurements that answer the same question.

The project uses `KV/state` as an umbrella term. Attention KV, sliding-window/local KV, and recurrent/linear-attention state are separated whenever the runtime exposes them. Current model, runtime, hardware, and granularity facts are tracked in [TECHNICAL_BASELINE.md](TECHNICAL_BASELINE.md).

## 2. Experiment groups

### 2.1 Modern KV / state bottleneck profiling

**Question:** Does the bottleneck studied by Strata still exist on modern hybrid models, and under which workload conditions does it become important?

The group contains four experiments:

1. context-length scaling;
2. shared-prefix-ratio scaling;
3. request-arrival-rate scaling;
4. cross-model synthesis and matched validation.

The first three experiments use one primary independent variable at a time. Active concurrency in the load experiment is observed as an outcome of arrival pressure rather than swept independently.

Primary observations include:

- cache/state footprint, separated by state type when possible;
- prefill or service computation time;
- CPU-GPU transfer volume and transfer activity;
- non-overlapped I/O stall;
- queueing delay under load;
- TTFT and throughput.

The main hierarchical-cache condition is a verified **CPU-resident hit**. A **recompute baseline** is used to quantify saved computation, and a **GPU-resident hit control** is used when practical to estimate the lower bound for reuse without CPU-GPU restore cost.

Run this group on both Qwen3.5-9B and Gemma 4 12B. Detailed designs and common measurement conventions are maintained under `experiments/modern-kv-state-bottleneck/`.

This group establishes whether the rest of the Strata-style optimization space is still practically relevant.

---

### 2.2 Hierarchical cache value

**Question:** When reusable GPU cache capacity is constrained, is extending reusable cache/state into CPU memory still worthwhile?

This group compares two system architectures under matched GPU cache budgets:

- **GPU-only**: reusable state is lost after GPU eviction and must be recomputed on a later revisit;
- **GPU + CPU hierarchical cache**: reusable state may survive GPU eviction in the validated CPU tier and be restored on a later revisit.

GPU hits, CPU hits, and recomputation are measured as outcomes. They are not treated as three independent forced-residency configurations in this group.

The group contains four experiments:

1. **Baseline Benefit**: GPU-only vs hierarchical cache under cold-cache and warm-cache conditions;
2. **GPU Cache Pressure Scaling**: vary only reusable-cache capacity pressure while keeping workload reuse fixed;
3. **Prefix Reuse Scaling**: vary only prefix revisit/reuse opportunity while keeping cache pressure and locality structure fixed;
4. **Cross-Model Validation**: run a small matched validation on the second model rather than repeating the full sweeps.

Experiments 1–3 use one validated primary model on A100 40GB. The default candidate is Qwen3.5-9B because it exercises both attention KV and Gated DeltaNet recurrent state, but this is conditional on the full hybrid-state offload/restore gate in [TECHNICAL_BASELINE.md](TECHNICAL_BASELINE.md). A partial Qwen3.5 offload path must not be reported as full hierarchical caching.

Experiment 4 validates representative conclusions on the second model on A100. The later Model and Hardware Generalization group reuses these A100 results and adds representative L40 runs instead of duplicating the same A100 work.

Primary observations include:

- GPU cache hit and eviction behavior;
- CPU cache-hit contribution;
- recomputation avoided by CPU-tier reuse;
- CPU-GPU transfer volume and non-overlapped restore stall;
- TTFT and throughput;
- active-request preemption, which must remain absent in the main reusable-cache pressure curve.

Detailed designs and shared validity rules are maintained under `experiments/hierarchical-cache-value/`.

---

### 2.3 Page granularity and GPU-assisted I/O

**Question:** Does fine-grained caching still improve effective reuse at the cost of fragmented I/O, and can GPU-assisted I/O recover transfer efficiency without imposing an excessive GPU compute cost?

This group contains four experiments:

1. **Page Size vs. Cache Reuse**: sweep only supported page sizes under a fixed attention backend and quantify effective reuse / page-boundary loss;
2. **Page Size vs. I/O Efficiency**: first isolate page/transfer fragmentation with controlled logical bytes, then validate whether the effect enters the serving critical path;
3. **GPU-Assisted I/O Compensation**: at representative page-size operating points, compare the same logical restore workload under standard-copy and GPU-assisted I/O;
4. **GPU Compute Cost and Net Benefit**: measure prefill/decode interference from GPU-assisted I/O and determine whether I/O stall reduction survives as an end-to-end benefit.

The primary mechanism path is SGLang HiCache because it exposes an explicit `page_size` together with `direct` and `kernel` CPU-GPU I/O backends. The exact runtime build, attention backend, host-memory layout, write policy, and hybrid-state support must pass the group validity gate before serving-level results are reported.

Primary observations include:

**Reuse layer**

- effective reused tokens;
- reuse efficiency relative to the logically reusable prefix;
- cache hit / occupancy / eviction supporting counters.

**I/O layer**

- actual CPU→GPU restore bytes;
- observed transfer/operation granularity;
- sustained host→GPU bandwidth;
- bandwidth utilization relative to a matched reference;
- restore duration and non-overlapped I/O stall.

**GPU / serving layer**

- prefill throughput and execution time;
- decode throughput and per-token latency;
- GPU-assisted I/O overlap / interference evidence;
- TTFT, request completion time, and overall throughput.

Configured page size must not be used as a substitute for observed transfer size. CPU→GPU restore traffic must also be separated from GPU→CPU backup/write-back traffic.

If an alternative runtime decouples prefix-match granularity from physical cache/transfer granularity, Experiments 1 and 2 must treat those as separate variables rather than pretending they share one page-size axis.

Detailed designs and shared conventions are maintained under `experiments/page-granularity-gpu-assisted-io/`.

---

### 2.4 Cache locality and scheduler behavior

**Question:** Under which workload structures do Strata-style control-plane optimizations still matter?

Control two workload dimensions:

- request arrival pressure;
- cache distance / context locality.

Representative workload patterns include:

- minimal locality;
- shuffled locality;
- large reuse distance;
- high concurrency on the same context.

Compare scheduler stages progressively:

1. baseline scheduling;
2. delay-hit mitigation;
3. balanced batching;
4. bubble filling / stall hiding;
5. complete scheduler.

Observe:

- delay hits;
- redundant prefill;
- queueing delay;
- non-overlapped I/O stall;
- TTFT;
- throughput.

This group should explain not only whether the scheduler helps, but which mechanism addresses which bottleneck.

---

### 2.5 End-to-end serving

**Question:** Do the individual mechanisms combine into meaningful serving gains without introducing regressions?

Use three workload families.

#### A. Long-context reuse

Targets Strata's primary reuse-oriented serving scenario.

#### B. Short-context

Checks whether the optimizations introduce regressions when hierarchical caching is less important.

#### C. Mixed workload

Combines:

- long shared contexts;
- ordinary short requests;
- different output lengths;
- different cache-locality patterns.

Compare the system progressively:

1. baseline;
2. hierarchical cache;
3. I/O optimizations;
4. scheduler optimizations;
5. full configuration.

Primary metrics:

- throughput;
- P50 / P90 / P99 TTFT;
- request completion time;
- GPU utilization.

This is the final system-level validation. Earlier microbenchmarks and ablations should explain the effects observed here.

---

### 2.6 Model and hardware generalization

**Question:** Which conclusions remain stable across modern model architectures and the two available GPU platforms?

Representative matrix:

| Model | A100 40GB | L40 48GB |
|---|---:|---:|
| Qwen3.5-9B | representative results reused / validated | representative validation |
| Gemma 4 12B | representative results reused / validated | representative validation |

The full 2 × 2 matrix is used only for representative configurations selected from the first five experiment groups. It is not necessary to repeat the complete experiment suite four times.

A100 results already produced by earlier cross-model validation should be reused when the configuration is identical. Group 6 should add only the missing matched runs needed to complete the model × hardware comparison.

The model dimension establishes cross-model robustness and relates measured cache/state behavior to observed performance. Differences between two models must not be interpreted as proof that attention architecture alone is the causal factor.

The hardware dimension evaluates whether conclusions depend strongly on memory capacity, CPU-GPU transfer characteristics, memory bandwidth, or GPU compute behavior.

## 3. Experimental logic

The six groups form a dependency chain:

```text
Bottleneck still exists?
        ↓
Is hierarchy itself useful?
        ↓
Where does I/O inefficiency come from, and can it be repaired economically?
        ↓
When does scheduling help?
        ↓
Do the pieces improve end-to-end serving?
        ↓
Do the conclusions generalize?
```

If an earlier premise is no longer true on modern models, later results must be interpreted as conditional engineering gains rather than evidence that the original bottleneck remains broadly important.

## 4. Reproducibility principles

For every reported experiment:

- record exact model identifier and revision;
- record hardware, topology, driver, CUDA/runtime, serving-engine version or commit, and relevant feature flags;
- record workload definition and token-length convention;
- record cache-residency mode and cache/state policy;
- explicitly record all granularity controls rather than a generic `page size` label when the runtime exposes multiple granularities;
- explicitly pin runtime defaults that can affect results, including I/O backend, host layout, write policy, and attention backend where relevant;
- keep raw measurements separate from processed plots;
- repeat measurements when variance is non-negligible;
- preserve failed or negative results when they affect interpretation;
- distinguish measured facts from architectural explanations;
- avoid causal claims that are not isolated by the experiment design;
- record runtime capability failures as `unsupported` or `partial` rather than silently substituting a different mechanism.

## 5. Scope discipline

The project should not reproduce every Strata figure simply for completeness. A historical figure should be reproduced when it is necessary to validate a mechanism, establish a baseline, or connect the modern evaluation to the original paper.

Conversely, experiments should not be compressed so aggressively that a major causal link, regression check, or generalization claim is left unsupported.
