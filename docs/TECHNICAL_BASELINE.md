# Technical Baseline

> Last verified: 2026-08-18

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

The experiment suite supports two serving runtimes with a shared workload
and output-schema layer.  They are not interchangeable evidence sources.

### vLLM (legacy / reference path)

The current vLLM model registry supports Qwen3.5 and Gemma 4. Qwen3.5 is handled as a hybrid model, and its Gated DeltaNet recurrent state is mapped through the runtime's Mamba-style state-cache abstraction.

vLLM also provides prefix caching and CPU KV/state offloading. The hybrid/Mamba prefix-caching path is still described by vLLM as experimental, so successful model loading alone is not sufficient evidence that the exact cache-reuse path used in this project is correct.

vLLM runtime facts pinned for the already-collected recompute results:

- version: vLLM 0.26.0, environment `~/yanglihan/dl-stack/envs/qwen`;
- `VLLM_USE_FLASHINFER_SAMPLER=0` (flashinfer sampler JIT fails on this CUDA/CUB);
- `VLLM_WORKER_MULTIPROC_METHOD=spawn`;
- Qwen `max_num_seqs=16` (Mamba cache block budget).

### SGLang / HiCache (explicit path for Exp1-3)

SGLang is driven only through its public server and HTTP/metrics boundaries (see `experiments/modern-kv-state-bottleneck/docs/06-sglang-execution-path.md`).

- installed commit: `4ad990ba7d75bb9f948f5f6bd8d79a66b5d3fd63`
  (`0.5.6.post3.dev8468+g4ad990ba7`), selected immediately before the
  upstream Torch 2.13 migration while retaining the required HiCache server
  args, layouts, metrics, and `benchmark/hicache/` implementation;
- read-only reference clone: `~/yanglihan/dl-stack/projects/sglang`, currently
  at `7120f3ee13de565cc737e0598110e7f7603c4e9f`; this is a code reference,
  not the installed runtime provenance;
- environment: `~/yanglihan/dl-stack/envs/sglang-hicache-cu129-torch211`
  with Torch `2.11.0+cu129`, `sglang-kernel 0.4.5+cu129`,
  `sgl-deep-gemm 0.1.5.post1+cu129`, user-prefix `cuda-nvcc 12.9.86`, and
  `g++ 12.4.0` (see `code/envs/sglang/SGLANG_ENV_PLAN.md`);
- invalid evidence-only prefixes: `envs/sglang` (CUDA 13) and
  `envs/sglang-cu129` (partial incompatible attempts); runners never select
  them implicitly;
- server: `python -m sglang.launch_server` with `--enable-metrics`;
- residency mapping:
  - `recompute` = `--disable-radix-cache`;
  - `gpu_hit` = radix cache on, HiCache off;
  - `cpu_hit` = `--enable-hierarchical-cache` with pinned `--hicache-*` flags (defaults: `--hicache-io-backend kernel`, `--hicache-mem-layout page_first`, `--hicache-write-policy write_through`, `--page-size 64`, `--hicache-ratio 2.0`).

The formal Qwen path uses one A100 with `--hicache-ratio 3`, `direct` I/O,
and `page_first_direct` host layout. For the 32K Gemma workload, a single
A100 exposes only about 29,248 usable tokens; the validated server launch is
therefore TP=2 on two A100s with `MEM_FRACTION=0.75`. This is a runtime
feasibility fact, not a cross-model performance comparison: TP must remain in
each run's provenance and any TP=2 metric without complete tier evidence is
unsupported.

SGLang-specific evidence fields (per-request `meta_info` and Prometheus metrics) are documented in `experiments/modern-kv-state-bottleneck/docs/06-sglang-execution-path.md` §4-§5. A missing public metric is recorded as `null`/unsupported, never as a silent zero.

### Shared validation gate

Before collecting reported measurements, each pinned runtime must pass a validation gate that confirms:

1. the exact checkpoint revision and model architecture loaded by the runtime;
2. prefix-cache hits produce numerically consistent outputs with recomputation;
3. CPU-resident hits actually restore all required cache/state groups;
4. transfer and cache-hit counters correspond to the intended request;
5. the selected cache policy and block/state checkpointing mode remain fixed across compared runs.

If any of these conditions fail, the affected result must be labeled as unsupported by that runtime rather than interpreted as a model property.

For the SGLang path the gate additionally verifies (via `GET /server_info`) that the resolved server flags match the pinned configuration.

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
