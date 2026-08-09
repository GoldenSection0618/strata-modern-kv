# Technical Baseline

> Last verified: 2026-08-09

This document records volatile model-architecture, runtime, hardware, and project-environment facts that affect experiment validity. Environment layout, package versions, compatibility settings, and compute-node loading rules are defined in [`ENVIRONMENT_REQUIREMENTS.md`](ENVIRONMENT_REQUIREMENTS.md). Every reported run must still record its exact checkpoint revision, runtime commit/version, driver, CUDA/runtime, cache configuration, scheduler configuration, and capability status.

## 1. Model baseline

### Qwen3.5-9B

Primary checkpoint family: `Qwen/Qwen3.5-9B`.

The official model card reports:

- 9B parameters;
- 32 language-model layers;
- hidden layout `8 × (3 × Gated DeltaNet + 1 × Gated Attention)`;
- native context length 262,144 tokens;
- separately supported extension up to 1,010,000 tokens.

The checkpoint config identifies the repeating layer types as `linear_attention` and `full_attention`.

For this project, Qwen3.5 is therefore treated as a hybrid full-attention / recurrent-state model. Its serving state is not equivalent to the ordinary per-token KV cache of a uniform dense-attention transformer.

Experiments use text-only requests unless explicitly stated otherwise.

### Gemma 4 12B Unified

Primary serving checkpoint: `google/gemma-4-12B-it`. Corresponding base family: `google/gemma-4-12B`.

The official model card reports:

- 11.95B parameters;
- 48 layers;
- 1,024-token local sliding window;
- 256K context length;
- hybrid attention interleaving local sliding-window attention and full global attention;
- unified Keys and Values in global layers.

The cache/state layout must therefore not be assumed to match a conventional uniform dense-attention K/V pair.

Experiments use text-only requests unless explicitly stated otherwise.

## 2. Hardware baseline

Primary measurement platform:

- NVIDIA A100 40GB.

Representative hardware-generalization platform:

- NVIDIA L40 48GB.

NVIDIA documents the L40 as a PCIe Gen4 Ada Lovelace GPU with 48 GB GDDR6 ECC memory. A100 is available in 40 GB and 80 GB variants; this project specifically uses the available 40 GB platform.

These are the platforms available to this project, not claims about the newest GPU generation.

The first five experiment groups use A100 as the default platform unless a design explicitly says otherwise. L40 is mainly reserved for representative hardware-generalization runs.

Every I/O-sensitive run records the actual GPU form factor, PCIe/host topology, NUMA placement, CPU model, host-memory policy, driver, and CUDA runtime. GPU model plus nominal memory size is insufficient metadata for bandwidth comparison.

## 3. Cluster software and environment requirements

The A100 cluster baseline has the following constraints:

- NVIDIA driver `525.60.13`;
- `nvidia-smi` reports `CUDA Version: 12.0`;
- Docker deployment is not allowed.

These are project-environment constraints and must not be generalized to other clusters.

The `CUDA Version` field printed by `nvidia-smi` is a driver capability indicator. It must not be interpreted as the locally installed CUDA toolkit version or as the CUDA target of PyTorch, vLLM, SGLang, or another user-space binary.

The project therefore records the actual PyTorch CUDA build, serving-runtime build source, and native-extension dependencies for every formal environment and run.

### 3.1 Runtime isolation

The required serving environments are isolated under:

```text
~/yanglihan/dl-stack/envs/
├── qwen
├── gemma4
└── sglang
```

- `qwen` and `gemma4` are vLLM environments.
- `sglang` is dedicated to SGLang / HiCache and uses the project-pinned source revision.
- SGLang dependencies must not be installed into the two vLLM environments.
- SGLang experiments must not fall back to a vLLM prefix.

Exact loading and recovery rules are defined in [`ENVIRONMENT_REQUIREMENTS.md`](ENVIRONMENT_REQUIREMENTS.md).

### 3.2 vLLM package baseline

Both vLLM environments use the same required software stack:

| Component | Required version |
|---|---|
| Python | `3.12.11` |
| PyTorch | `2.11.0+cu129` |
| vLLM | `0.26.0+cu129` |
| Triton | `3.6.0` |
| Transformers | `5.14.1` |

vLLM must use the explicit CUDA-12.9 release wheel:

```text
vllm-0.26.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
```

The un-suffixed/default vLLM 0.26.0 CUDA-13 binary is not the project runtime path on this cluster. Adding a CUDA-12.9 PyTorch index does not change the CUDA target of the vLLM native wheel itself, so the selected vLLM artifact must be recorded explicitly.

The vLLM native library must resolve CUDA 12 runtime libraries rather than `libcudart.so.13`.

Both vLLM environments also use the fixed compatibility setting:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

This disables the FlashInfer sampling path that triggers the observed CUB `FlagHeads` JIT compilation failure on the project A100 software stack and falls back to the PyTorch sampler. This setting is held fixed across paired vLLM comparisons unless sampler implementation itself becomes an experimental variable.

### 3.3 libstdc++ compatibility

Rocky 8.6 system `libstdc++.so.6` is insufficient when the required Python/runtime stack needs `CXXABI_1.3.15`.

Each relevant conda environment therefore requires `libstdcxx-ng`, and the environment library directory must be visible through `LD_LIBRARY_PATH` before Python starts.

A `sitecustomize.py` change to `LD_LIBRARY_PATH` is not an acceptable substitute because it executes after process startup and cannot reliably alter already-resolved dynamic-linker dependencies.

### 3.4 SGLang package baseline

SGLang / HiCache must use the independent `envs/sglang` prefix and the project-pinned source revision. Current upstream CUDA/package defaults must not be copied into the experiment environment without checking them against the pinned project revision and the cluster constraints.

Environment conformance is separate from mechanism capability. Before SGLang contributes a mechanism-specific result, the relevant experiment must validate the exact checkpoint, native kernels, attention backend, page-size support, HiCache effective configuration, full required state restore, `direct` / `kernel` I/O path where applicable, and numerical consistency.

### 3.5 Target-checkpoint runtime support

Generic model-family support is not sufficient for this project because the experiments depend on exact hybrid-state behavior.

For Qwen3.5, official SGLang cookbook documentation lists `Qwen/Qwen3.5-9B`. This establishes an upstream support reference, but the project still validates the exact pinned runtime and the state mechanisms required by each experiment.

For Gemma 4, generic Gemma 4 runtime support cannot be used as evidence that the exact `google/gemma-4-12B-it` checkpoint and its required cache/state mechanisms work correctly.

A vLLM v0.21.0 issue reported an initialization failure for the exact Gemma 4 12B checkpoint. That historical issue does not establish the behavior of vLLM 0.26.0+cu129, so the target checkpoint must be validated on the project runtime rather than inferred from the historical issue.

Consequently both target checkpoints must pass an explicit model gate on each runtime/platform combination before mechanism results are accepted. For Gemma 4 12B in particular, successful plain generation is a prerequisite before cache/hierarchy conclusions are attempted.

## 4. Runtime roles

The project does not assume one serving engine is optimal for every experiment group.

Runtime selection follows the mechanism being measured and the exact build used by the experiment.

### 4.1 vLLM

Current vLLM cache configuration exposes hybrid/Mamba-style cache controls, including separate Mamba cache/block settings and `prefix_match_unit`.

`prefix_match_unit` is the finest token boundary at which prefix-cache hashes/hits can land. Current documentation explicitly allows it to be finer than physical KV cache blocks when divisibility requirements hold. It controls matching granularity, not how often states are stored.

Current vLLM also exposes group-aware KV-cache capacity and Mamba cache controls. A hybrid-model experiment must therefore record the resolved physical/cache-group layout rather than infer capacity from a single ordinary KV-block count.

The current `OffloadingConnector` extends prefix caching into pinned CPU memory and can also participate in configurations with additional secondary tiers. Only the CPU primary tier has direct GPU access in that design. Completed GPU blocks can be copied into host memory and promoted back on demand; GPU↔CPU transfers use asynchronous DMA (`cudaMemcpyAsync`).

Therefore a vLLM experiment must keep distinct:

- prefix-match granularity;
- physical attention block size;
- recurrent/Mamba-style cache block or checkpoint granularity;
- offload/transfer behavior;
- observed actual transfer size.

A generic `page size` label is not sufficient when these differ.

### 4.2 SGLang HiCache

SGLang HiCache is the preferred mechanism path for the Page Granularity and GPU-Assisted I/O group.

Current upstream server arguments expose, among others:

- `--enable-hierarchical-cache`;
- general cache page-size controls such as `--page-size` where supported by the selected backend;
- `--hicache-io-backend` with `direct` and `kernel` GPU paths;
- `--hicache-mem-layout`;
- `--hicache-write-policy`;
- host-cache size/ratio controls.

The current HiCache design documentation describes `direct` as standard CUDA-copy I/O and `kernel` as GPU-assisted I/O.

Defaults and supported option sets have changed across SGLang revisions. The project therefore pins resolved values explicitly and does not rely on current/default documentation alone. The code can also override a requested layout at runtime for some backend combinations, so command-line intent and effective configuration must both be recorded.

Attention-backend support for page size and hybrid-state support must be validated on the exact selected commit. Model launch support is not equivalent to HiCache support for every state group required by that model.

### 4.3 SGLang hybrid-state HiCache status

Current SGLang code contains a hybrid cache assembly path and recent Qwen3.5 reports show separate hierarchical attention-KV and Mamba/recurrent host pools being allocated. Full hybrid-state HiCache is therefore an implemented path, but experiment-level correctness and resource semantics must be validated on the exact project model and pinned build.

Recent upstream reports on larger Qwen3.5 hybrid models describe:

- a `HiMambaRadixCache` transfer-path crash on first inference in an April 2026 main-branch environment;
- `--hicache-size` allocating the requested amount separately to the KV host pool and the Mamba host pool in a v0.5.13 report, causing substantial host-memory over-allocation relative to the recurrent device state;
- long stalls during Mamba eviction under large-context traffic in a v0.5.14 report.

These reports use different Qwen3.5 variants and different hardware from this project's Qwen3.5-9B on A100/L40. They are capability warnings, not proof that the project's exact combination fails.

For this project:

- full Qwen3.5 hierarchy is claimed only after attention KV and recurrent/Gated-DeltaNet state both restore correctly on the exact pinned build;
- actual GPU and CPU allocation must be recorded separately by observable state group, rather than inferred from a single aggregate `hicache-ratio` or `hicache-size` setting;
- a configured `--hicache-size` value is not assumed to mean the same total host budget across hybrid state groups; the resolved per-group allocation is authoritative;
- recurrent-state restore, eviction, transfer bytes, stall, and numerical consistency must be observed independently from attention-KV behavior;
- if attention KV works but the recurrent-state hierarchy is unavailable or unreliable, the run is `partial` or `unsupported`, not a full hierarchical-cache result.

## 5. Strata scheduler reference semantics

Scheduler experiments use the Strata paper as the semantic reference, not whatever current upstream scheduler happens to call a similar feature.

Strata §4.3 defines three stages:

1. **Delay-hit deferral**. Requests matching a context whose initial miss is still unresolved are deferred until the cache becomes ready, reducing redundant computation.
2. **Balanced batch formation**. After delay-hit candidates are handled, the scheduler uses load and compute requirements to avoid severely loading-bound prefill batches and preferentially exploit bundle hits when suitable.
3. **Bubble filling / stall hiding**. If a batch remains loading-bound, useful work is inserted into the loading bubble. The paper uses decoding work as the main example and notes that prefill work can also be used in P-D-disaggregated settings.

The paper uses a 100-token delay-hit threshold and a load/compute threshold of 100 in its own testbed. These are **historical implementation parameters**, not universal values. This project must calibrate or explicitly justify any threshold used on modern models/hardware.

Current upstream SGLang/vLLM scheduler behavior must not be assumed semantically identical to these three Strata stages. Scheduler attribution requires explicit mechanism-equivalence validation or a project implementation with independently controlled switches.

## 6. Mandatory runtime validation gates

### 6.1 General cache/state gate

Before a runtime contributes reported cache-reuse/offload results:

1. Required native extensions import and load without unresolved CUDA dependencies.
2. Exact checkpoint revision and runtime model implementation are recorded.
3. Prefix reuse produces numerically consistent outputs with full recomputation for the tested text-only workload.
4. GPU-resident and CPU-resident hits are observable from counters/events/instrumentation rather than inferred from configuration alone.
5. CPU-resident restore covers every cache/state group required to skip the claimed recomputation.
6. Cache policy, cache dtype, scheduler policy and GPU cache budget remain fixed across paired comparisons unless explicitly studied.
7. Reusable-prefix eviction is distinguished from active-request preemption.
8. When the runtime allocates separate cache/state groups, configured budgets and resolved GPU/CPU allocations are recorded per group when observable.

If required state or native runtime behavior cannot be verified, the affected result is labeled `partial` or `unsupported`.

### 6.2 Qwen3.5 gate

Validation must cover both:

- full-attention KV;
- Gated DeltaNet recurrent/state-cache data.

For both state families, the project records actual device/host allocation, restore/eviction behavior, transferred bytes and numerical correctness when the runtime exposes them.

Restoring only attention KV is a partial hierarchy, not a full Qwen3.5 hierarchical-cache result.

### 6.3 Gemma 4 gate

Validation must first confirm that the exact `google/gemma-4-12B-it` checkpoint executes correctly on the pinned runtime.

Cache/state validation must then cover the local/sliding-window and global-attention state groups actually retained by that runtime.

Generic Gemma 4 model-family support or successful execution of another Gemma 4 variant does not satisfy this gate.

### 6.4 Scheduler gate

Before scheduler component attribution:

1. Delay-hit events must be observable as unresolved same-context misses or an equivalent verified state transition.
2. Delay-hit mitigation must be shown to defer those requests rather than merely alter generic priority.
3. Balanced batching must expose or reconstruct the load/compute decision and loading-bound classification.
4. Bundle-hit behavior must be observable or explicitly marked unsupported.
5. Bubble filling must expose residual loading intervals and the useful work inserted into them.
6. Component toggles must be semantically independent before leave-one-out ablation is used.
7. If component isolation is impossible, progressive ablation plus instrumentation is used and the unsupported attribution is stated explicitly.

### 6.5 Page-granularity / GPU-I/O gate

For page-granularity experiments:

1. Compared page sizes use the same supported attention backend.
2. `direct` and `kernel` restore identical logical state and produce consistent outputs.
3. Host layout and write policy remain fixed across backend comparisons.
4. Actual CPU→GPU payload bytes / transfer behavior are observable.
5. CPU→GPU restore and GPU→CPU backup/write-back are accounted separately.
6. Hybrid-state tracking/checkpoint parameters remain fixed or their page-size dependency is documented.
7. Cross-page-size serving comparisons control page-size-dependent attention-kernel performance, preferably with matched GPU-resident-hit controls.

## 7. Granularity terminology

`KV/state` is an umbrella term.

- `attention KV`: state retained by attention layers;
- `local/sliding-window KV`: bounded state retained by local attention;
- `recurrent state`: state retained by linear-attention or recurrent layers such as Qwen3.5 Gated DeltaNet;
- `configured page size`: runtime tokens/page control when such a unified control exists;
- `prefix-match granularity`: finest boundary at which prefix reuse can be recognized;
- `physical cache block size`: runtime storage/allocation unit;
- `offload / transfer granularity`: unit submitted to host-transfer path;
- `actual transfer size`: payload observed after batching/coalescing/kernel processing;
- `configured host-cache budget`: requested host-memory control before runtime state-group expansion or override;
- `resolved state-group allocation`: actual GPU/CPU memory allocated to each observable state group after runtime processing;
- `cache resolve time`: interval from an unresolved miss being accepted until matching context becomes safely reusable;
- `full hierarchical hit`: every state group needed to skip the corresponding prefix computation is restored;
- `partial hierarchical hit`: only a subset is restored.

State groups and distinct granularity controls are reported separately whenever the runtime exposes them.

## 8. Interpretation boundaries

The project does not assume cache/state footprint grows linearly with context length for every state group.

Sliding-window attention can bound local KV retention. Recurrent-state layouts can scale according to checkpoint/runtime policy. Allocation alignment, padding, cache dtype and checkpoint granularity can affect observed memory.

Cross-model differences may correlate with cache/state behavior but do not isolate attention architecture as the sole cause.

CPU offloading is not beneficial merely because it produces hits. A valid systems conclusion requires avoided recomputation, transfer activity, non-overlapped stall and end-to-end metrics.

GPU-assisted I/O is not beneficial merely because bandwidth rises. GPU compute interference and end-to-end serving performance must be included.

Environment conformance does not establish mechanism capability. A runtime feature must be validated on the exact pinned build before it contributes a mechanism-specific result.

Generic model-family support is likewise insufficient. Exact target-checkpoint support and the mechanism-specific state path must be separately verified.

For hybrid runtimes, configured cache size is not automatically equivalent to actual total memory consumed. Resolved allocation by state group is part of the measured system state.

## 9. Primary references

### Original system

- Strata, OSDI 2026: https://www.usenix.org/conference/osdi26/presentation/xie-zhiqiang

### Models

- Qwen3.5-9B model card: https://huggingface.co/Qwen/Qwen3.5-9B
- Qwen3.5-9B config: https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json
- Gemma 4 12B model card: https://huggingface.co/google/gemma-4-12B
- Gemma 4 12B IT model card: https://huggingface.co/google/gemma-4-12B-it

### vLLM

- Current cache configuration: https://docs.vllm.ai/en/latest/api/vllm/config/cache/
- Current KV offloading guide: https://docs.vllm.ai/en/latest/features/kv_offloading_usage/
- vLLM 0.26.0 release: https://github.com/vllm-project/vllm/releases/tag/v0.26.0
- vLLM 0.26.0 CUDA-12.9 wheel index: https://wheels.vllm.ai/0.26.0/cu129/vllm/
- Gemma 4 12B v0.21.0 support issue retained as a validation warning: https://github.com/vllm-project/vllm/issues/44494

### SGLang

- Current SGLang documentation: https://docs.sglang.io/
- Qwen3.5 cookbook: https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.5
- Gemma 4 cookbook: https://docs.sglang.io/cookbook/autoregressive/Google/Gemma4
- Current HiCache design: https://github.com/sgl-project/sglang/blob/main/docs_new/docs/advanced_features/hicache_design.mdx
- Current server arguments: https://github.com/sgl-project/sglang/blob/main/docs_new/docs/advanced_features/server_arguments.mdx
- Current hybrid-cache assembly code: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/unified_radix_cache.py
- Qwen3.5 hybrid HiCache transfer-path warning: https://github.com/sgl-project/sglang/issues/24121
- Qwen3.5 hybrid `hicache-size` allocation warning: https://github.com/sgl-project/sglang/issues/29034
- Qwen3.5 Mamba eviction-stall warning: https://github.com/sgl-project/sglang/issues/30314

### CUDA compatibility

- NVIDIA CUDA Compatibility Guide: https://docs.nvidia.com/deploy/cuda-compatibility/
- CUDA 12.9 Release Notes, driver compatibility table: https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html

### Hardware

- NVIDIA A100: https://www.nvidia.com/en-us/data-center/a100/
- NVIDIA L40: https://www.nvidia.com/en-us/data-center/l40/