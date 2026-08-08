# Common Conventions: Page Granularity and GPU-Assisted I/O

## 1. Purpose

本文档定义 Experiments 1–4 共享的 runtime、granularity、I/O backend、measurement 和 validity 口径。四个实验只有在这些口径一致时才能组成一条可解释的因果链。

## 2. Primary runtime path

本实验组优先使用 **SGLang HiCache** 作为机制实现路径，因为当前 HiCache 直接暴露以下与 Strata 机制对应的控制项：

- `--page-size`：KV cache storage / retrieval granularity；
- `--hicache-io-backend direct`：标准 CUDA memory-copy 路径；
- `--hicache-io-backend kernel`：GPU-assisted I/O kernel 路径；
- `--hicache-mem-layout`：host cache memory layout；
- `--hicache-write-policy`：GPU→CPU backup / write-back policy。

实验必须记录并固定精确的 SGLang version / commit。不得依赖 runtime 默认值，因为默认 backend、layout 或其他行为可能随版本变化。

如果最终使用其他 serving runtime，只有在该 runtime 能提供等价且可验证的控制变量时，才能沿用本文档中的实验名称。否则必须重新定义变量，不能把不同语义的 knob 都称为 `page size`。

## 3. Granularity terminology

### 3.1 Configured page size

在 SGLang 主路径中，`page size` 专指 `--page-size`，单位为 tokens/page。

它决定 KV cache block 的 token granularity，并影响 prefix cache 能够匹配的完整 page 边界。正式结果必须记录 configured page size，而不是只记录诸如 small / medium / large 的标签。

### 3.2 Hybrid-state granularity

对于 Qwen3.5 等包含 recurrent / linear-attention state 的模型，attention page 与 recurrent-state tracking/checkpoint granularity 不是天然等价概念。

任何影响 recurrent state 保存或恢复位置的参数都必须单独记录并保持可比。若该参数与 `page size` 存在整除、对齐或实现约束，page-size sweep 只能使用同时满足这些约束的配置。

### 3.3 Actual transfer granularity

`configured page size` 不能直接替代 `actual transfer size`。

实际 CPU→GPU I/O 可能受到 host memory layout、transfer batching、coalescing、kernel implementation 和 runtime scheduling 的影响。因此 Experiments 2–4 必须通过 instrumentation / profiler 记录实际 transfer behavior，而不是根据 page size 推断 fragmentation。

### 3.4 Alternative-runtime warning

某些现代 runtime 会将 prefix-match granularity 与物理 cache block size 解耦。例如，vLLM 的 `prefix_match_unit` 可以比物理 KV cache block 更细，并且只控制 prefix matching，不控制 state storage frequency。

如果采用这类 runtime：

- Experiment 1 的独立变量应写成 **reuse-matching granularity**；
- Experiment 2 的独立变量应写成 **physical / transfer granularity**；
- 两者不得继续假设共享同一个 `page-size axis`。

## 4. Runtime capability gate

正式实验前，目标 model × runtime × attention backend 必须通过以下 gate。

1. 目标 checkpoint 能够稳定启动并完成固定 text-only inference trace。
2. Prefix reuse 的输出与 full recomputation 数值一致。
3. CPU-resident restore 能够被 runtime counter、trace 或 instrumentation 实际观测。
4. 对 hybrid model，所有为了跳过对应 prefix computation 而必需的 state groups 都能正确保存和恢复。
5. Qwen3.5 的验证必须覆盖 full-attention KV 与 Gated DeltaNet recurrent/state data。
6. Gemma 4 的验证必须覆盖 pinned runtime 实际保留的 local/sliding-window 与 global-attention state groups。
7. `direct` 与 `kernel` backend 在相同 cache hit 上产生一致的模型输出和 reuse accounting。
8. 所有 page-size candidate 都由同一个 attention backend 支持。不得为了扩大 page-size sweep 而在不同 page size 之间切换 attention backend。
9. HiCache host layout、write policy、GPU cache budget 和 scheduler policy 在需要隔离 page size 或 I/O backend 的比较中保持固定。

无法通过 gate 的配置标记为 `unsupported` 或 `partial`，不能通过 silently fallback、替换 backend 或只测试部分 state 来制造完整结果。

## 5. Fixed runtime controls

除实验明确改变的变量外，每个 paired comparison 至少显式固定并记录：

- model identifier 与 revision；
- SGLang/runtime commit；
- attention backend；
- precision 与 cache dtype；
- `page_size`；
- hybrid/recurrent-state tracking parameters；
- GPU cache budget；
- CPU HiCache size；
- `hicache_io_backend`；
- `hicache_mem_layout`；
- `hicache_write_policy`；
- scheduler / overlap configuration；
- request trace 与 random seed；
- driver、CUDA 与 hardware topology。

比较 page size 时，除 `page_size` 及其不可避免的派生 allocation behavior 外，上述控制项保持不变。

比较 I/O backend 时，`page_size`、layout、write policy、logical bytes、cache state 与 workload 保持不变。

## 6. Workload and cache-state rules

### 6.1 Prefix alignment

Experiment 1 的共享 prefix cut points 必须包含与候选 page size 不完全对齐的情况。若所有 prefix length 都恰好是所有 page size 的共同倍数，实验将人为消除 page-boundary reuse loss。

### 6.2 Cache capacity

不同 page-size 配置以相同 **memory budget** 为主要容量控制口径，而不是简单固定 page 数量。

每个 run 同时记录 runtime 实际可用 token/state capacity 和 allocator padding。若 page size 改变导致实际可用容量变化，该变化必须作为 measured consequence 报告，而不是隐藏在配置中。

### 6.3 Read and write traffic

Serving-level I/O analysis必须区分：

- CPU→GPU restore traffic；
- GPU→CPU backup / write-back traffic。

Experiment 2–4 以 CPU→GPU restore 为主要研究对象，但后台 write traffic 不能混入 restore bandwidth 或 stall 而不加区分。

## 7. Metric definitions

### 7.1 Effective reused tokens

实际避免重新执行 prefix prefill 的 token 数量。

### 7.2 Reuse efficiency

```text
reuse efficiency = effective reused tokens / logically reusable prefix tokens
```

该指标比 page-level hit count 更直接地描述 page boundary 对可利用 reuse 的影响。

### 7.3 Sustained host→GPU bandwidth

使用目标 CPU→GPU restore 的实际 payload bytes 与对应 transfer interval 计算。必须说明 measurement window 和 overlap accounting。

### 7.4 Bandwidth utilization

```text
bandwidth utilization = observed sustained bandwidth / matched reference bandwidth
```

Reference bandwidth 必须在同一 hardware、同一 host-memory condition、同一 transfer direction 下测量。

### 7.5 Non-overlapped I/O stall

只统计进入 serving critical path、没有被 computation overlap 隐藏的 restore stall。Raw transfer duration 不得直接当作 non-overlapped stall。

### 7.6 Compute interference

GPU-assisted I/O 对模型计算的影响必须由 prefill / decode execution time、throughput 和 profiler evidence 联合支持。单独的 GPU utilization 上升不是 interference 证据。

## 8. Experiment dependency

```text
Experiment 1
page size → effective reuse
        ↓
Experiment 2
page size → observed transfer fragmentation → I/O efficiency / stall
        ↓
Experiment 3
same page/workload → direct vs kernel I/O compensation
        ↓
Experiment 4
kernel I/O benefit → GPU compute interference → end-to-end net benefit
```

Experiment 2 不能仅凭 Experiment 1 的 page-size labels 推断 I/O fragmentation。Experiment 3 不能仅凭 bandwidth recovery 宣称端到端收益。Experiment 4 必须使用实际 end-to-end measurement 判断净收益。

## 9. Interpretation boundary

本实验组最终需要确定的是 operating region，而不是证明某个机制始终更优。

如果目标 hybrid model 无法通过 HiCache full-state validation，controlled I/O microbenchmark 仍可用于验证 transfer mechanism，但 serving-level 结果必须标记为 `partial` / `unsupported`，不能外推为该 hybrid model 的完整系统结论。
