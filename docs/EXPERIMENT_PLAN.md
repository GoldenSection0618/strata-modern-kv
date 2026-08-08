# Experiment Plan

## 1. Evaluation objective

The objective is to re-evaluate the main systems claims behind Strata under modern hybrid-attention LLMs and current GPU platforms.

The experiment suite is designed around causal questions rather than one-to-one figure reproduction. Each group should be broad enough to support a credible conclusion, while avoiding repeated measurements that answer the same question.

## 2. Experiment groups

### 2.1 Modern KV / state bottleneck profiling

**Question:** Does the bottleneck studied by Strata still exist on modern hybrid-attention models, and how severe is it?

Vary representative workload pressure along three dimensions:

- context length;
- shared-prefix length;
- concurrency / request rate.

Observe:

- cache or recurrent-state memory growth;
- prefill latency;
- CPU–GPU traffic;
- I/O stall;
- TTFT.

Run this group on both Qwen3.5-9B and Gemma 4 12B.

This group establishes whether the rest of the Strata-style optimization space is still practically relevant.

---

### 2.2 Hierarchical cache value

**Question:** When GPU memory is constrained, is extending cache/state storage into CPU memory still worthwhile?

Compare:

- GPU-only cache;
- GPU + CPU hierarchical cache.

Evaluate under different combinations of:

- warm and cold cache states;
- GPU cache-capacity pressure;
- prefix-reuse intensity.

Observe:

- GPU and CPU cache hit behavior;
- recomputation;
- CPU–GPU traffic;
- TTFT;
- throughput.

This group evaluates the value of the cache hierarchy itself, independently from later I/O and scheduler optimizations.

---

### 2.3 Page granularity and GPU-assisted I/O

**Question:** Does fine-grained caching still improve reuse at the cost of fragmented I/O, and can GPU-assisted I/O recover transfer efficiency without sacrificing reuse?

Control two primary factors:

- cache page granularity;
- I/O mechanism.

Measure three layers of behavior:

**Cache layer**

- cache hit rate;
- effectively reused tokens or state.

**I/O layer**

- sustained host-to-GPU bandwidth;
- bandwidth utilization.

**GPU layer**

- prefill throughput;
- decode throughput;
- interference between I/O work and model computation.

The goal is to reconstruct the full causal chain from cache granularity to reuse, transfer efficiency, and final compute impact.

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
- I/O stall;
- TTFT;
- throughput.

This group should explain not only whether the scheduler helps, but which mechanism addresses which bottleneck.

---

### 2.5 End-to-end serving

**Question:** Do the individual mechanisms combine into meaningful serving gains without introducing regressions?

Use three workload families:

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

**Question:** Which conclusions remain stable across modern model architectures and GPU platforms?

Representative matrix:

| Model | A100 40GB | L40 48GB |
|---|---:|---:|
| Qwen3.5-9B | ✓ | ✓ |
| Gemma 4 12B | ✓ | ✓ |

The full matrix is used only for representative configurations selected from the first five experiment groups. It is not necessary to repeat the complete experiment suite four times.

The model dimension is intended to establish cross-model robustness and to relate actual cache/state behavior to observed performance. Differences between two models must not be interpreted as proof that attention architecture alone is the causal factor.

The hardware dimension evaluates whether conclusions depend strongly on memory capacity, bandwidth, transfer characteristics, or GPU compute behavior.

## 3. Experimental logic

The six groups form a dependency chain:

```text
Bottleneck still exists?
        ↓
Is hierarchy itself useful?
        ↓
Where does I/O inefficiency come from?
        ↓
When does scheduling help?
        ↓
Do the pieces improve end-to-end serving?
        ↓
Do the conclusions generalize?
```

If an earlier premise is no longer true on modern models, later results should be interpreted as conditional engineering gains rather than evidence that the original bottleneck remains broadly important.

## 4. Reproducibility principles

For every reported experiment:

- record model, hardware, software version, workload definition, and configuration;
- keep raw measurements separate from processed plots;
- repeat measurements when variance is non-negligible;
- preserve failed or negative results when they affect interpretation;
- distinguish measured facts from architectural explanations;
- avoid causal claims that are not isolated by the experiment design.

## 5. Scope discipline

The project should not reproduce every Strata figure simply for completeness. A historical figure should be reproduced when it is necessary to validate a mechanism, establish a baseline, or connect the modern evaluation to the original paper.

Conversely, experiments should not be compressed so aggressively that a major causal link, regression check, or generalization claim is left unsupported.
