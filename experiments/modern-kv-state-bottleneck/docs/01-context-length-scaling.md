# Experiment 1: Context Length Scaling

## 1. 实验目标

本实验用于评估现代 hybrid 模型在长上下文场景下的 cache/state 压力，并判断 Strata 所关注的 state restore 与 CPU-GPU I/O 问题是否会随着 context length 增长而变得重要。

实验主要回答三个问题：

1. context length 增长时，不同 cache/state group 的实际内存 footprint 如何变化；
2. context length 增长时，prefill/recompute cost 与 CPU-resident state restore cost 分别如何变化；
3. TTFT 的增长主要由 computation、non-overlapped I/O stall，还是其他开销推动。

本实验不预设 cache/state footprint 必须随 context 线性增长。Qwen3.5 的 recurrent state 与 Gemma 4 的 sliding-window/global KV 具有不同的 retention behavior，实际 scaling 以 runtime measurement 为准。

## 2. 实验对象

实验分别使用 Qwen3.5-9B 与 Gemma 4 12B。

两种模型保持各自原生 attention/cache/state 机制，不人为统一成普通 dense-attention KV cache。

主实验统一在 A100 40GB 上完成。L40 48GB 留到后续硬件泛化实验中用于代表性配置复验。

正式实验前必须通过仓库 `docs/TECHNICAL_BASELINE.md` 所定义的 runtime validation gate。

## 3. Context length 设置

本实验定义 `1K = 1024 tokens`。

主实验使用以下 context length：

| Label | Exact tokens |
|---|---:|
| 4K | 4,096 |
| 8K | 8,192 |
| 16K | 16,384 |
| 32K | 32,768 |

如果两个模型在 32K 条件下仍能在相同实验约束下稳定运行，则增加 65,536 tokens 作为额外压力测试点。

64K 不作为必须完成的主实验点，不为了加入该点而改变 cache policy、precision、并发或其他核心条件。

## 4. Shared-prefix 条件

主实验固定 shared-prefix ratio 为 50%。

| Context length | Shared prefix | Unique suffix |
|---|---:|---:|
| 4K | 2,048 | 2,048 |
| 8K | 4,096 | 4,096 |
| 16K | 8,192 | 8,192 |
| 32K | 16,384 | 16,384 |

固定 reuse ratio 而不是固定绝对 prefix length，可以避免 context length 增长时 reuse proportion 同时变化。

Shared-prefix ratio 的独立变化留到 Experiment 2。

## 5. Cache-residency 条件

主实验条件为 **CPU-resident hit**。

在正式请求开始前，共享 prefix 的可复用 cache/state 已经存在于 CPU/offload tier。请求执行时需要恢复对应 state，并对 unique suffix 执行正常计算。

每个 context point 同时保留两个轻量控制条件：

### Recompute baseline

不恢复已有 prefix state，完整计算 context。该条件量化没有 reuse 时的计算成本。

### GPU-resident hit control

在 runtime 支持且能够稳定控制 residency 时，使相同 reusable prefix 已驻留 GPU。该条件用于估计不包含 CPU-GPU restore cost 的 reuse 下界。

如果 runtime 无法可靠实现某个控制条件，应明确记录为 unavailable，而不是用未验证的行为替代。

## 6. 请求与负载条件

本实验使用 text-only workload。

并发与 request rate 保持在低且固定的区域，使 queueing 对 TTFT 的贡献可忽略或单独测量。

Experiment 1 不进行 concurrency 或 request-rate sweep。其核心自变量只有 context length。

所有请求使用相同且较短的 output length，避免 decode workload 成为主要变量。

## 7. 核心测量指标

所有指标遵循 `00-measurement-conventions.md`。

### 7.1 Cache/state footprint

记录 runtime-observed cache/state memory。

能够分项时分别报告：

- full/global-attention KV；
- sliding-window/local-attention KV；
- recurrent/linear-attention state；
- allocator padding 或其他 cache-management overhead。

形成：

> Context length → cache/state footprint by state type

重点观察 scaling shape 与 slope，不预设其为线性关系。

### 7.2 Computation cost

记录 recompute baseline 的 prefill/service computation，以及 CPU-resident hit 下 unique suffix 的 computation cost。

形成：

> Context length → recomputation cost / residual computation cost

该结果用于区分长 context 带来的计算压力与 state restore 压力。

### 7.3 CPU-GPU transfer

记录 CPU-resident hit 的：

- transfer volume；
- transfer activity/duration；
- achieved transfer bandwidth。

原始 transfer duration 允许与 computation 重叠，因此只作为资源指标，不直接与 compute time 相加。

### 7.4 Non-overlapped I/O stall

记录执行路径真正等待 cache/state restore 的 non-overlapped stall。

低负载条件下主要使用：

```text
service stall ratio = I/O stall / service time
```

由于 queueing 应接近零，TTFT stall contribution 可作为辅助展示。

### 7.5 TTFT

记录 client-observed TTFT，并结合 recompute、GPU-resident hit 与 CPU-resident hit 三种条件解释其变化。

实验最终需要回答 TTFT 为什么随 context length 变化，而不是只给出一条总 latency 曲线。

## 8. 实验执行方式

每一个配置先执行 warm-up，再进行多次独立重复测量。

正式结果至少报告 median、P90 与波动范围。

Qwen3.5 与 Gemma 4 使用相同的 exact token length、reuse ratio、output length 和低负载条件。由于 tokenizer 不同，跨模型比较匹配 token 数量和 workload structure，不要求 raw text 完全一致。

不同 context point 采用交替或随机顺序执行。每个测量点恢复到规定的 cache-residency 初始状态。

实验元数据必须记录 model revision、runtime version/commit、precision、cache policy 与所有影响 hybrid-state caching 的配置。

## 9. 最终结果组织

实验形成四组核心结果：

1. **Context length → cache/state footprint by state type**；
2. **Context length → computation cost**；
3. **Context length → transfer volume/bandwidth 与 non-overlapped I/O stall**；
4. **Context length → TTFT，比较 recompute、CPU-resident hit，并在可用时加入 GPU-resident hit control**。

结果同时展示 Qwen3.5 与 Gemma 4，但跨模型重点是 scaling trend 与 bottleneck composition，不是简单比较谁绝对更快。

## 10. 结果判断逻辑

### 情况 A：State restore pressure 随 context 增长明显增强

如果 CPU-resident hit 的 transfer demand 与 non-overlapped I/O stall 增长，并且 service stall ratio 上升，则说明 Strata 所针对的 loading bottleneck 在现代模型上仍然存在，并在 long-context workload 下加剧。

### 情况 B：主要增长来自 computation

如果 state transfer 的绝对成本增长，但 non-overlapped stall 占 service time 的比例稳定或下降，同时 recompute/prefill computation 成为主要增长项，则说明 hierarchical state management 仍有成本，但不是主要 scaling bottleneck。

### 情况 C：State footprint 并不随 context 持续增长

如果某类 state 因 sliding window、recurrent representation 或 runtime policy 呈现 bounded 或 stepwise behavior，应将其作为模型实际 state behavior 报告，而不能把它解释成测量失败。

### 情况 D：两个模型表现显著不同

如果 Qwen3.5 与 Gemma 4 的 footprint、restore cost 或 stall scaling 明显不同，则结论限定为 bottleneck behavior 具有 model-dependent 特征。

本实验不把模型差异直接归因于 attention architecture。

## 11. 实验边界

本实验只系统改变 context length。

Shared-prefix ratio、request rate、cache locality、page size 与 scheduler strategy 均不在本实验中 sweep。

统一术语与 timing accounting 见 [00-measurement-conventions.md](00-measurement-conventions.md)。
