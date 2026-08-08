# Technical Baseline

> Last verified: 2026-08-09

This document records volatile model-architecture, runtime, and hardware facts that affect experiment validity. Exact checkpoint revisions, runtime commits, drivers, CUDA versions, cache configuration, and feature status used by a reported run must still be recorded in that run's metadata.

## 1. Model baseline

### Qwen3.5-9B

Primary checkpoint family: `Qwen/Qwen3.5-9B`.

The language backbone has 32 layers arranged as eight repetitions of three Gated DeltaNet layers followed by one Gated Attention layer. The official model card reports a native context length of 262,144 tokens, with longer-context extension available separately.

For this project, Qwen3.5 is treated as a hybrid full-attention / recurrent-state model. Its serving state is not equivalent to the ordinary per-token KV cache of a uniform dense-attention model.

Experiments use text-only requests unless an experiment explicitly states otherwise. Multimodal inputs are excluded so vision-side state and preprocessing do not become additional variables.

### Gemma 4 12B Unified

Primary serving checkpoint: `google/gemma-4-12B-it`. The corresponding base family is `google/gemma-4-12B`.

The official model card reports 48 layers, a 1,024-token local sliding window, and a 256K context length. The model interleaves local sliding-window attention with full global attention. Global layers use unified Keys and Values, so the cache layout must not be assumed to match a conventional dense-attention K/V pair.

Experiments also use text-only requests for Gemma 4 unless explicitly stated otherwise.

## 2. Hardware baseline

Primary measurement platform:

- NVIDIA A100 40GB.

Representative hardware-generalization platform:

- NVIDIA L40 48GB.

The A100 and L40 are the hardware available to this project, not claims about the newest GPU generation in 2026. The first five experiment groups use A100 as the default platform unless a design explicitly says otherwise. L40 is reserved mainly for representative hardware-generalization runs.

Every I/O experiment must record the actual GPU form factor, PCIe / host topology, NUMA placement, CPU model, pinned-memory policy, driver, and CUDA runtime. Model name plus nominal GPU memory size is not sufficient metadata for bandwidth comparisons.

## 3. Runtime roles

This project does not assume one serving runtime is optimal for every experiment group. Runtime selection follows the mechanism that must be measured.

### 3.1 vLLM

vLLM is a candidate runtime for modern-model serving, prefix caching, state profiling, and CPU cache offloading where its pinned build passes the experiment-specific validation gate.

Current vLLM documentation exposes support for the model families used here and provides hybrid/Mamba-style cache management. Its CPU `OffloadingConnector` stores completed blocks in pinned host memory and transfers GPU↔CPU data with asynchronous DMA.

A critical granularity detail is that current vLLM can decouple prefix matching from physical cache storage. `prefix_match_unit` defines the finest token boundary for a prefix-cache hit and can be finer than a physical KV cache block. It controls matching granularity, not how often states are stored. Hybrid models also expose separate Mamba cache block/page parameters.

Therefore, a vLLM experiment must not use the generic term `page size` unless the document identifies which concrete runtime granularity is being changed.

### 3.2 SGLang HiCache

SGLang HiCache is the primary mechanism candidate for the **Page Granularity and GPU-Assisted I/O** group because current SGLang exposes the controls needed to reproduce that causal chain directly:

- `--page-size`: number of tokens grouped into a KV cache page/block and used for storage/retrieval granularity;
- `--hicache-io-backend direct`: standard CUDA memory-copy path;
- `--hicache-io-backend kernel`: GPU-assisted I/O kernel path;
- `--hicache-mem-layout`: host-memory layout;
- `--hicache-write-policy`: host-tier backup/write-back policy.

Current SGLang documentation also notes that page-size support depends on the attention backend. Some backends only support fixed or restricted native page sizes. A page-size sweep must therefore keep the attention backend fixed and use only values supported by that same backend.

SGLang HiCache and hybrid-model support continue to evolve. Model-serving support alone is not evidence that hierarchical cache restore is correct for every state group. The exact pinned SGLang build must pass the full-state validation gate before serving-level results are interpreted.

Do not rely on HiCache defaults. `page_size`, I/O backend, host layout, write policy, cache size, and scheduler/overlap settings must be explicitly pinned because defaults and supported paths can change across releases.

## 4. Mandatory runtime validation gates

### 4.1 General gate

Before a runtime contributes reported cache-reuse or offload results:

1. The exact checkpoint revision and runtime model implementation must match the intended model.
2. Prefix reuse must produce numerically consistent outputs with full recomputation for the tested text-only workload.
3. GPU-resident and CPU-resident hits must be observable from runtime counters, events, or instrumentation rather than inferred from configuration alone.
4. CPU-resident restore must cover every cache/state group required to skip the claimed recomputation.
5. Cache policy, cache dtype, scheduler policy, and GPU cache budget must remain fixed across paired comparisons unless explicitly studied.
6. A cache-pressure experiment must distinguish reusable-prefix eviction from active-request preemption.

If any required state group cannot be verified, the affected result is labeled `unsupported` or `partial` rather than interpreted as a model property.

### 4.2 Qwen3.5 gate

Validation must explicitly cover both:

- full-attention KV;
- Gated DeltaNet recurrent/state-cache data.

A path that saves or restores only attention KV is a **partial hierarchy**, not a valid full Qwen3.5 hierarchical-cache result.

### 4.3 Gemma 4 gate

Validation must cover the local/sliding-window and global-attention state groups actually retained by the pinned runtime.

### 4.4 Page-granularity / GPU-I/O gate

For the page-granularity experiment group:

1. All compared page sizes must be supported by the same attention backend.
2. `direct` and `kernel` I/O must restore identical logical state and produce consistent model outputs.
3. `hicache_mem_layout` and `hicache_write_policy` must be explicitly fixed across backend comparisons.
4. Actual CPU→GPU payload bytes and transfer behavior must be observable. Configured page size is not a substitute for measured transfer granularity.
5. CPU→GPU restore traffic and GPU→CPU backup/write-back traffic must be accounted separately.
6. Hybrid-state tracking/checkpoint parameters must remain fixed or their required relationship to page size must be explicitly documented.

## 5. Granularity terminology

`KV/state` is an umbrella term in this repository.

- `attention KV`: state retained by attention layers;
- `local/sliding-window KV`: bounded attention state retained by local attention;
- `recurrent state`: state retained by linear-attention or recurrent layers such as Qwen3.5 Gated DeltaNet;
- `configured page size`: the tokens/page control of the selected runtime, when such a unified control exists;
- `prefix-match granularity`: the finest boundary at which prefix reuse can be recognized;
- `physical cache block size`: the runtime storage/allocation unit;
- `offload / transfer granularity`: the logical or physical unit submitted to the host-transfer path;
- `actual transfer size`: the payload observed from the runtime/profiler after batching, coalescing, or kernel processing;
- `cache/state footprint`: runtime-observed memory allocated or resident for these structures;
- `full hierarchical hit`: every state group needed to skip the corresponding prefix computation is restored;
- `partial hierarchical hit`: only a subset of those state groups is restored.

Whenever the runtime exposes a breakdown, state groups and granularity controls must be reported separately before an aggregate number is presented.

## 6. Interpretation boundaries

The project must not assume that cache/state footprint grows linearly with context length for every state group.

Sliding-window attention can bound local KV retention. Recurrent-state layers can use checkpoint/state layouts whose scaling depends on runtime policy. Runtime allocation, block alignment, padding, checkpoint granularity, and cache dtype can all affect measured memory.

Cross-model differences may be associated with different cache/state behavior, but two-model comparisons do not isolate attention architecture as the sole causal factor.

CPU offloading must not be described as beneficial merely because it produces hits. A valid systems conclusion requires relating reuse to avoided recomputation, CPU-GPU transfer activity, non-overlapped stall, TTFT, and throughput.

Likewise, GPU-assisted I/O must not be described as beneficial merely because raw bandwidth rises. The final claim requires accounting for GPU computation interference and end-to-end serving performance.

## 7. Primary references

### Models

- Qwen3.5 model card: https://huggingface.co/Qwen/Qwen3.5-9B
- Gemma 4 12B model card: https://huggingface.co/google/gemma-4-12B
- Gemma 4 12B IT model card: https://huggingface.co/google/gemma-4-12B-it

### vLLM

- vLLM cache configuration: https://docs.vllm.ai/en/latest/api/vllm/config/cache/
- vLLM KV offloading guide: https://docs.vllm.ai/en/latest/features/kv_offloading_usage/
- vLLM supported models: https://docs.vllm.ai/en/latest/models/supported_models/

### SGLang

- SGLang HiCache design: https://github.com/sgl-project/sglang/blob/main/docs_new/docs/advanced_features/hicache_design.mdx
- SGLang server arguments: https://github.com/sgl-project/sglang/blob/main/docs_new/docs/advanced_features/server_arguments.mdx
- SGLang attention-backend page-size notes: https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/attention_backend.md

### Hardware

- NVIDIA A100: https://www.nvidia.com/en-us/data-center/a100/
- NVIDIA L40: https://www.nvidia.com/en-us/data-center/l40/
