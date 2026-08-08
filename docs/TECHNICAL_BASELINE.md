# Technical Baseline

> Last verified: 2026-08-09

This document records volatile model-architecture, runtime, hardware, and project-environment facts that affect experiment validity. Every reported run must still record its exact checkpoint revision, runtime commit/version, driver, CUDA/runtime, cache configuration, scheduler configuration, and capability status.

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

## 3. Current cluster software constraints

The inspected A100 node reports:

- NVIDIA driver `525.60.13`;
- `nvidia-smi` reports `CUDA Version: 12.0`;
- Docker deployment is not allowed.

These are project-environment facts and must not be generalized to other clusters.

The `CUDA Version` field printed by `nvidia-smi` must not be interpreted as the locally installed CUDA toolkit version or as a categorical statement that newer CUDA 12.x user-space binaries cannot run. NVIDIA's CUDA minor-version compatibility policy lists Linux driver `525.60.13` as the minimum driver for the CUDA 12.x family. However, minor-version compatibility is conditional. PTX JIT paths and features that require newer driver support can still fail, and a package may contain native extensions with stricter requirements.

Therefore this project treats driver `525.60.13` as meeting the baseline CUDA 12.x minor-compatibility floor, not as proof that CUDA 12.8/12.9 packages or kernels are executable. Every candidate runtime still has to pass native-extension loading, model execution, and mechanism-specific validation on the actual node.

### 3.1 SGLang packaging status

SGLang v0.5.11 moved the default CUDA stack to CUDA 13.0 across SGLang, `sgl-kernel`, and default images. Later release-line documentation continues to treat CUDA 13 as the default path, while the upstream CUDA migration work retains CUDA 12.9 as a non-default compatibility path.

The current SGLang repository release line has advanced beyond v0.5.11. The transition version is retained here because it marks the packaging change relevant to this cluster, not because v0.5.11 is the recommended project version.

Therefore:

- default CUDA-13 SGLang packages/images are not the project execution path on the current A100 node;
- a non-Docker CUDA-12.x-compatible installation must be explicitly selected and pinned for the chosen release/commit;
- the node meeting NVIDIA's CUDA 12.x minor-compatibility floor does not prove that a CUDA 12.9 SGLang build, selected model, HiCache backend, PTX/native kernels, or target attention backend works correctly;
- current upstream defaults must not be copied into the experiment configuration without resolving them on the exact pinned build.

Before SGLang contributes measured results, validation must cover import, native-kernel loading, server launch, target model execution, hierarchical-cache behavior, and any direct/kernel I/O path used by the experiment.

### 3.2 vLLM packaging status

The project previously attempted `vLLM 0.26.0` from the stable/default package path on this node and observed a native-extension dependency on `libcudart.so.13`. This is retained as a **cluster-specific failed baseline**.

The distinction between vLLM release packaging paths is important:

- the un-suffixed stable v0.26.0 wheel used by the ordinary stable install path is documented as CUDA 13.0-compatible by default;
- v0.26.0 also has an explicit `+cu129` release-wheel variant;
- current vLLM main/nightly documentation uses CUDA 12.9 as the default binary variant and also provides CUDA 13.0 variants.

Therefore adding a CUDA-12.9 PyTorch index to a command is not by itself proof that the installed vLLM native extension is a CUDA-12.9 build. The selected vLLM wheel/build artifact itself must be identified and recorded.

For this project:

- the previously tested vLLM 0.26.0 un-suffixed CUDA-13 binary path is not a validated runtime on the current node;
- the explicit v0.26.0 `+cu129` release wheel, a pinned CUDA-12.9 development wheel, or a source build is a candidate path;
- driver `525.60.13` meeting the CUDA 12.x minor-compatibility floor is necessary context but not sufficient runtime validation;
- the exact vLLM release/commit, wheel/build source, CUDA target, PyTorch build, and native-extension status must be pinned;
- a candidate contributes no experimental result until model execution, cache/state behavior, and required connector or scheduler mechanisms pass validation.

### 3.3 Target-checkpoint runtime support

Generic model-family support is not sufficient for this project because the experiments depend on exact hybrid-state behavior.

For Qwen3.5, current official SGLang cookbook documentation explicitly lists `Qwen/Qwen3.5-9B`. The documented recipe uses the SGLang main branch at that support point. This establishes an upstream model-support candidate, but not proof that the project's pinned non-Docker CUDA-12.x build supports the model or the required hierarchical-state mechanisms.

For Gemma 4, current official SGLang cookbook documentation describes Gemma 4 support but does not list the exact `google/gemma-4-12B-it` Unified checkpoint among its enumerated deployment models. This absence does not prove the checkpoint is unsupported, but generic Gemma 4 support cannot be used as evidence that this exact target works.

A vLLM v0.21.0 issue reported an initialization failure for the exact Gemma 4 12B checkpoint. That historical issue does not establish the status of v0.26.0, but it is sufficient reason not to assume current support without a fresh pinned-version validation.

Consequently both target checkpoints must pass an explicit model gate on each runtime/platform combination before any mechanism experiment begins. For Gemma 4 12B in particular, successful plain generation is a prerequisite before cache/hierarchy conclusions are attempted.

## 4. Runtime roles

The project does not assume one serving engine is optimal for every experiment group.

Runtime selection follows the mechanism being measured and the build that actually passes the current cluster capability gate.

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

SGLang HiCache remains the preferred mechanism candidate for the Page Granularity and GPU-Assisted I/O group, subject to the cluster build gate.

Current upstream server arguments expose, among others:

- `--enable-hierarchical-cache`;
- general cache page-size controls such as `--page-size` where supported by the selected backend;
- `--hicache-io-backend` with `direct` and `kernel` GPU paths;
- `--hicache-mem-layout`;
- `--hicache-write-policy`;
- host-cache size/ratio controls.

The current HiCache design documentation describes `direct` as standard CUDA-copy I/O and `kernel` as GPU-assisted I/O.

Defaults and supported option sets have changed across SGLang revisions. The project therefore pins resolved values explicitly and does not rely on current/default documentation alone.

Attention-backend support for page size and hybrid-state support must be validated on the exact selected commit. Model launch support is not equivalent to HiCache support for every state group required by that model.

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

If required state or native runtime behavior cannot be verified, the affected result is labeled `partial` or `unsupported`.

### 6.2 Qwen3.5 gate

Validation must cover both:

- full-attention KV;
- Gated DeltaNet recurrent/state-cache data.

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

A runtime feature documented upstream but not executable under this project's pinned cluster environment is a candidate capability, not an available experimental capability.

Generic model-family support is likewise a candidate capability. Exact target-checkpoint support and the mechanism-specific state path must be separately verified.

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
- Current main-branch CUDA installation source: https://github.com/vllm-project/vllm/blob/main/docs/getting_started/installation/gpu.cuda.inc.md
- vLLM 0.26.0 release: https://github.com/vllm-project/vllm/releases/tag/v0.26.0
- Official vLLM-Omni installation note documenting the vLLM 0.26.0 un-suffixed CUDA-13 wheel: https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/installation/gpu/
- Gemma 4 12B v0.21.0 support issue retained as a validation warning: https://github.com/vllm-project/vllm/issues/44494

### SGLang

- Current SGLang documentation: https://docs.sglang.io/
- Qwen3.5 cookbook: https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.5
- Gemma 4 cookbook: https://docs.sglang.io/cookbook/autoregressive/Google/Gemma4
- Current HiCache design: https://github.com/sgl-project/sglang/blob/main/docs_new/docs/advanced_features/hicache_design.mdx
- Current server arguments: https://github.com/sgl-project/sglang/blob/main/docs_new/docs/advanced_features/server_arguments.mdx
- v0.5.11 CUDA-13 migration release: https://github.com/sgl-project/sglang/releases/tag/v0.5.11
- CUDA migration tracker: https://github.com/sgl-project/sglang/issues/21498

### CUDA compatibility

- NVIDIA CUDA Compatibility Guide: https://docs.nvidia.com/deploy/cuda-compatibility/
- CUDA 12.9 Release Notes, driver compatibility table: https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html

### Hardware

- NVIDIA A100: https://www.nvidia.com/en-us/data-center/a100/
- NVIDIA L40: https://www.nvidia.com/en-us/data-center/l40/
