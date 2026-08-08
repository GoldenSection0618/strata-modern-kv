# Technical Baseline

> Last verified: 2026-08-09

This document records volatile model-architecture, runtime, and hardware facts that affect experiment validity. Exact checkpoint revisions, runtime commits, drivers, and cache configuration used by a reported run must still be recorded in that run's metadata.

## 1. Model baseline

### Qwen3.5-9B

Primary checkpoint family: `Qwen/Qwen3.5-9B`.

The language backbone has 32 layers arranged as eight repetitions of three Gated DeltaNet layers followed by one Gated Attention layer. The official model card reports a native context length of 262,144 tokens, with longer-context extension available separately.

For this project, Qwen3.5 is therefore treated as a hybrid full-attention / recurrent-state model. Its serving state is not equivalent to the ordinary per-token KV cache of a uniform dense-attention model.

The hierarchical-cache experiments use text-only requests. Multimodal inputs are excluded so vision-side state and preprocessing do not become additional experimental variables.

### Gemma 4 12B Unified

Primary serving checkpoint: `google/gemma-4-12B-it`. The corresponding base family is `google/gemma-4-12B`.

The official model card reports 48 layers, a 1,024-token local sliding window, and a 256K context length. The model interleaves local sliding-window attention with full global attention. The global layers also use unified Keys and Values, so the cache layout must not be assumed to match a conventional dense-attention K/V pair.

vLLM registers the 12B Unified checkpoint as `Gemma4UnifiedForConditionalGeneration`.

The hierarchical-cache experiments also use text-only requests for Gemma 4.

## 2. Hardware baseline

Primary measurement platform:

- NVIDIA A100 40GB.

Generalization platform:

- NVIDIA L40 48GB.

The A100 and L40 are the hardware available to this project, not claims about the newest GPU generation in 2026. NVIDIA documents the A100 in 40GB configurations and the L40 with 48GB GDDR6 ECC memory.

The first five experiment groups use A100 as the default platform unless a design explicitly says otherwise. L40 is reserved mainly for representative hardware-generalization runs so the complete experiment suite is not duplicated across both GPUs.

## 3. vLLM runtime baseline

Current vLLM documentation and source API pages expose support for both model families used here:

- Qwen3.5 is implemented by `Qwen3_5ForConditionalGeneration` and marked as a hybrid model;
- Gemma 4 Unified is implemented by `Gemma4UnifiedForConditionalGeneration`;
- Qwen3.5 Gated DeltaNet uses vLLM's generic Mamba-style state-management interfaces;
- vLLM exposes prefix caching for hybrid/Mamba-style models, but its Mamba prefix-caching support is explicitly described as experimental;
- vLLM exposes native CPU KV-cache offloading through the `OffloadingConnector`, which stores completed cache blocks in host memory and promotes offloaded hits back to GPU.

These facts establish that the required mechanisms exist in the runtime. They do **not** establish that a particular pinned vLLM build correctly offloads and restores every state group of Qwen3.5 or Gemma 4 under the exact configuration used in this project.

## 4. Mandatory runtime validation gate

Before any hierarchical-cache result is considered valid, the pinned runtime must pass all of the following checks.

1. The exact checkpoint revision and runtime architecture must match the intended model.
2. Prefix-cache reuse must produce numerically consistent outputs with full recomputation for the tested text-only workload.
3. GPU-resident and CPU-resident hits must be observable from runtime counters, events, or instrumentation rather than inferred from configuration alone.
4. CPU-resident restore must cover every cache/state group required to skip the claimed recomputation.
5. For Qwen3.5, validation must explicitly cover both full-attention KV and Gated DeltaNet recurrent/state-cache data. A path that offloads only the attention KV is a **partial hierarchy**, not a valid full Qwen3.5 hierarchical-cache result.
6. For Gemma 4, validation must cover the local/sliding-window and global-attention cache groups actually retained by the pinned runtime.
7. Paired GPU-only and hierarchical runs must keep prefix-cache policy, block/state checkpointing mode, cache dtype, scheduler policy, and GPU cache budget fixed. The intended architecture difference is CPU-tier enablement. When a CPU offloading backend is used, its implementation and configuration must remain pinned across all hierarchical runs being compared.
8. A cache-pressure experiment must distinguish reusable-prefix eviction from active-request preemption. Runs that change active-request feasibility or trigger scheduler preemption are not valid measurements of reusable-cache pressure unless preemption is explicitly the studied variable.

If any required state group cannot be verified, the affected result must be labeled `unsupported` or `partial` rather than interpreted as a model property.

## 5. Terminology

`KV/state` is an umbrella term in this repository.

- `attention KV` refers to state retained by attention layers;
- `local/sliding-window KV` refers to the bounded attention state retained by local attention;
- `recurrent state` refers to state retained by linear-attention or recurrent layers such as Qwen3.5 Gated DeltaNet;
- `cache/state footprint` refers to runtime-observed memory allocated or resident for these structures;
- `full hierarchical hit` means that every state group needed to skip the corresponding prefix computation is restored from the hierarchy;
- `partial hierarchical hit` means only a subset of those state groups is restored.

Whenever the runtime exposes a breakdown, state groups must be reported separately before an aggregate number is presented.

## 6. Interpretation boundaries

The project must not assume that cache/state footprint grows linearly with context length for every state group.

Sliding-window attention can bound local KV retention. Recurrent-state layers can use checkpoint/state layouts whose scaling depends on runtime policy. Runtime allocation, block alignment, padding, checkpoint granularity, and cache dtype can all affect measured memory.

Cross-model differences may be associated with different cache/state behavior, but two-model comparisons do not isolate attention architecture as the sole causal factor.

CPU offloading must not be described as beneficial merely because it produces hits. A valid systems conclusion requires relating reuse to avoided recomputation, CPU-GPU transfer activity, non-overlapped stall, TTFT, and throughput.

## 7. Primary references

- Qwen3.5 model card: https://huggingface.co/Qwen/Qwen3.5-9B
- Gemma 4 12B model card: https://huggingface.co/google/gemma-4-12B
- Gemma 4 12B IT model card: https://huggingface.co/google/gemma-4-12B-it
- vLLM supported models: https://docs.vllm.ai/en/latest/models/supported_models/
- vLLM Qwen3.5 implementation: https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/qwen3_5/
- vLLM Gemma 4 Unified implementation: https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/gemma4_unified/
- vLLM Mamba-prefix-caching interface: https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/interfaces/
- vLLM KV offloading guide: https://docs.vllm.ai/en/latest/features/kv_offloading_usage/
- vLLM offloading scheduler: https://docs.vllm.ai/en/latest/api/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler/
- NVIDIA L40 specifications: https://www.nvidia.com/en-us/data-center/l40/
