# Technical Baseline

> Last verified: 2026-08-09

This document records volatile model-architecture and serving-runtime facts that affect the experiment design. Exact checkpoint revisions and runtime commits used by an experiment must still be recorded in the experiment metadata.

## 1. Model baseline

### Qwen3.5-9B

Primary checkpoint family: `Qwen/Qwen3.5-9B`.

The language backbone has 32 layers arranged as eight repetitions of three Gated DeltaNet layers followed by one Gated Attention layer. The native context length is 262,144 tokens.

For this project, Qwen3.5 must therefore be treated as a hybrid attention/recurrent-state model. Its runtime state is not equivalent to the ordinary per-token KV cache of a dense full-attention model.

### Gemma 4 12B

Primary checkpoint family: `google/gemma-4-12B` / instruction-tuned variant when required by the serving setup.

The 12B model has 48 layers and uses hybrid local sliding-window attention and full global attention. Its sliding window is 1,024 tokens and its context length is 256K tokens.

For this project, Gemma 4 must therefore be treated as a hybrid attention model whose local and global attention layers have different cache-retention behavior.

## 2. Runtime baseline

The current vLLM model registry supports Qwen3.5 and Gemma 4. Qwen3.5 is handled as a hybrid model, and its Gated DeltaNet recurrent state is mapped through the runtime's Mamba-style state-cache abstraction.

vLLM also provides prefix caching and CPU KV/state offloading. The hybrid/Mamba prefix-caching path is still described by vLLM as experimental, so successful model loading alone is not sufficient evidence that the exact cache-reuse path used in this project is correct.

Before collecting reported measurements, each pinned runtime must pass a validation gate that confirms:

1. the exact checkpoint revision and model architecture loaded by the runtime;
2. prefix-cache hits produce numerically consistent outputs with recomputation;
3. CPU-resident hits actually restore all required cache/state groups;
4. transfer and cache-hit counters correspond to the intended request;
5. the selected cache policy and block/state checkpointing mode remain fixed across compared runs.

If any of these conditions fail, the affected result must be labeled as unsupported by that runtime rather than interpreted as a model property.

## 3. Terminology

`KV/state` is an umbrella term in this repository.

- `attention KV` refers to key/value cache retained by attention layers;
- `recurrent state` refers to state retained by linear-attention or recurrent layers such as Qwen3.5 Gated DeltaNet;
- `cache/state footprint` refers to the runtime-observed memory allocated or resident for these structures.

Whenever the runtime exposes the breakdown, attention KV and recurrent state must be reported separately before presenting an aggregate number.

## 4. Interpretation boundaries

The project must not assume that cache/state footprint grows linearly with context length for every cache group.

Sliding-window attention can bound local KV retention, while recurrent-state layers may retain checkpoint/state structures whose scaling depends on the serving runtime and prefix-cache policy. Runtime allocation, block alignment, padding, and checkpoint granularity can also affect measured memory.

Cross-model differences may be associated with different cache/state behavior, but two-model comparisons do not isolate attention architecture as the sole causal factor.

## 5. Primary references

- Qwen3.5 model card: https://huggingface.co/Qwen/Qwen3.5-9B
- Gemma 4 12B model card: https://huggingface.co/google/gemma-4-12B
- vLLM supported models: https://docs.vllm.ai/en/latest/models/supported_models/
- vLLM hybrid KV cache manager: https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/
- vLLM KV offloading guide: https://docs.vllm.ai/en/latest/features/kv_offloading_usage/
