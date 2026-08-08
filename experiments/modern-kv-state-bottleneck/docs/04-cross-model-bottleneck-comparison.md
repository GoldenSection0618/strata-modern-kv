# Experiment 4: Cross-Model Bottleneck Comparison

## 1. 实验定位

Experiment 4 不重新完整执行 Experiments 1-3，而是复用前三组结果，并补充少量严格 matched validation。

本实验用于回答：

> Strata 所关注的 KV/state bottleneck 在 Qwen3.5-9B 与 Gemma 4 12B 上是否都存在，严重程度和触发条件是否稳定，以及这些差异与实际 cache/state behavior 有什么关联。

本实验只支持跨模型稳定性与关联性结论，不把两种模型之间的差异直接归因于 attention architecture。

## 2. 数据来源

Experiment 4 主要复用：

### Experiment 1

- context length；
- state-type footprint；
- computation cost；
- CPU-GPU transfer；
- non-overlapped I/O stall；
- TTFT。

### Experiment 2

- shared-prefix ratio；
- reusable state footprint；
- avoided recomputation；
- CPU restore penalty；
- Net Reuse Benefit；
- reuse speedup。

### Experiment 3

- absolute and normalized request rate；
- achieved throughput；
- active concurrency；
- transfer bandwidth；
- queueing；
- service stall ratio；
- P50/P90/P99 TTFT。

因此 Experiment 4 属于 cross-model synthesis + matched validation，而不是第四套完整 workload sweep。

## 3. 比较口径

Experiment 4 明确区分两种比较，避免把“相同 workload”和“相同 relative load”混为一谈。

### 3.1 Absolute matched comparison

用于比较模型在完全相同系统 workload 下的状态与 latency behavior。

两个模型保持一致的：

- exact context token count；
- shared-prefix ratio；
- output token count；
- cache-residency mode；
- offered request rate；
- measurement procedure。

Absolute matched points 只选择两个模型都能稳定承载的低到中等 load，不强行要求某个相同 requests/s 同时接近两个模型各自的 saturation。

### 3.2 Capacity-normalized comparison

用于比较两个模型接近各自 serving limit 时的 bottleneck evolution。

定义：

```text
Normalized Load = Offered Request Rate / Measured Sustainable Capacity
```

在相似 normalized load 下，两个模型的 absolute requests/s 可以不同。因此这类结果只能解释“相对于各自 capacity 的压力”，不能称为 identical absolute workload。

## 4. Matched validation points

Absolute matched validation 建议选择三个主要点：

| Configuration | Context | Prefix reuse | Load |
|---|---:|---:|---|
| Short / light | 8,192 | 50% | common low load |
| Long / light | 32,768 | 50% | common low load |
| Long / high reuse | 32,768 | 75% | common low load |

Capacity-normalized validation 使用 Experiment 3 的 load curve，重点比较约半负载、中高负载、接近 saturation 的区域。

如果 32K 不是两个模型都能稳定运行的公共点，则使用在最终 runtime/configuration 下验证过的最大公共 context point，并在所有 cross-model matched tests 中保持一致。

## 5. Token 与内容匹配

由于 Qwen3.5 与 Gemma 4 tokenizer 不同，cross-model benchmark 以 exact token count 和 workload structure 为主要控制量。

不要求同一 raw text 在两个 tokenizer 下恰好得到相同 token 数。可以从同一语料分布生成或截取 model-specific token sequences，但必须保证：

- context length 相同；
- shared-prefix ratio 相同；
- 请求组结构相同；
- output length 约束相同。

这避免 tokenizer 差异破坏系统层面的 length matching。

## 6. State Footprint Comparison

相同 context point 下首先比较 runtime-observed cache/state footprint。

能够分项时分别报告：

- full/global-attention KV；
- sliding-window/local-attention KV；
- recurrent/linear-attention state；
- allocator/padding overhead。

主要比较：

1. absolute state footprint per request；
2. context 增长时的 measured footprint slope；
3. 各 state type 在 aggregate footprint 中的比例。

不把 `state bytes per token` 作为唯一归一化指标。对于 sliding-window 或 recurrent-state cache，该指标可能呈现 bounded、stepwise 或 runtime-policy-dependent behavior。

如果需要使用 effective bytes/token，只作为 descriptive normalization，并与原始 footprint 曲线同时报告。

## 7. State Movement Comparison

在相同 CPU-resident reuse workload 下比较：

- restored state bytes per request；
- transfer bandwidth；
- transfer activity/duration；
- non-overlapped I/O stall；
- service stall ratio。

可以计算：

```text
Transferred Bytes per Reused Token = Restored Bytes / Reused Tokens
```

但该指标必须结合 state-type breakdown 解释，不能假设两个模型每个 reused token 对应同类状态。

如果某模型 transferred bytes 更少，但 service stall ratio 并未同比下降，则说明性能差异不能由 state volume 单独解释。

## 8. Reuse Efficiency Comparison

使用 Experiment 2 的统一 residency definitions。

定义：

```text
Reuse Speedup = TTFT_recompute / TTFT_CPU-hit
Net Reuse Benefit = TTFT_recompute - TTFT_CPU-hit
CPU Restore Penalty = TTFT_CPU-hit - TTFT_GPU-hit
```

其中 GPU-resident hit 只在两种模型都能可靠执行时用于严格 matched comparison。

进一步比较：

```text
Reuse Realization Ratio = Net Reuse Benefit / Avoidable Recompute Time
```

该指标用于表示理论上可避免的 recomputation 有多少真正转化为 end-to-end TTFT 改善。

如果分母非常小或 measurement uncertainty 与分母同量级，则不报告该 ratio，避免产生不稳定的放大结果。

## 9. Load Sensitivity Comparison

使用 Experiment 3 同时展示 absolute load 与 normalized load。

### Absolute view

展示两个模型在公共低、中负载区域的真实 requests/s、TTFT、state traffic 与 achieved throughput。

该视图回答：

> 相同 workload 下两个模型产生怎样不同的系统压力。

### Normalized view

比较约 50%、75%、90% capacity 以及 saturation 附近的：

- service stall ratio；
- queueing delay；
- P50/P90/P99 TTFT；
- achieved throughput；
- transfer bandwidth utilization。

该视图回答：

> 当两个模型接近各自 capacity 时，哪类 bottleneck 更早主导。

## 10. Bottleneck Composition

对 representative matched points 使用统一的 TTFT accounting：

```text
TTFT = queueing + service time
service time = compute-path time + non-overlapped I/O stall + other service overhead
```

Raw transfer duration 不作为额外 additive component，避免与 overlapped computation 重复计时。

结果按统一表格组织：

| Workload | Model | Queueing | Compute path | I/O stall | Other |
|---|---|---:|---:|---:|---:|
| Short/light | Qwen3.5 | ... | ... | ... | ... |
| Short/light | Gemma 4 | ... | ... | ... | ... |
| Long/light | Qwen3.5 | ... | ... | ... | ... |
| Long/light | Gemma 4 | ... | ... | ... | ... |
| Long/high-reuse | Qwen3.5 | ... | ... | ... | ... |
| Long/high-reuse | Gemma 4 | ... | ... | ... | ... |

## 11. Bottleneck transition 的定义

如果前三组曲线存在明显 transition，可为两个模型标记 approximate bottleneck region。

Transition criterion 必须在最终 comparative analysis 前固定，并对两个模型使用同一规则。可以使用 service stall ratio、throughput saturation 与 queue growth 的组合判据，也可以采用预先定义的数据驱动 changepoint 方法。

不得在观察完两种模型结果后分别挑选不同阈值，以制造更强的架构差异。

最终比较：

- context 增长到什么区域后 state pressure 明显；
- reuse 增长到什么区域后 CPU restore penalty 明显；
- normalized load 增长到什么区域后 I/O stall 或 queueing 快速放大。

## 12. 最终结果组织

Experiment 4 建议形成四组核心结果。

### 图 1：Context Length → State Footprint by Type

展示 absolute footprint 与必要的 normalized slope，避免只用单一 bytes/token 指标。

### 图 2：Shared Prefix → Reuse Benefit vs Restore Cost

比较 Reuse Speedup、Net Reuse Benefit、CPU Restore Penalty 与 service stall ratio。

### 图 3：Absolute / Normalized Load → Bottleneck Growth

同时保留真实 requests/s 与 normalized load 两种视图。

### 图 4：Representative Workloads → TTFT Composition

展示 queueing、compute path、non-overlapped I/O stall 与 other overhead。

该图作为本实验组的核心 cross-model summary。

## 13. 结果判断逻辑

### 情况 A：两种模型趋势一致

两个模型虽然 state representation 不同，但都表现出 context/reuse/load 增长后 restore pressure 和 I/O stall 上升。

结论是 Strata 所研究的问题至少在两种现代 hybrid 模型上具有跨模型稳定性。

### 情况 B：问题存在，但严重程度不同

两个模型都存在 state-loading cost，但 bottleneck 出现的 context、reuse 或 normalized-load 区域明显不同。

结论限定为问题仍存在，但严重程度和触发条件 model-dependent。

### 情况 C：只有一个模型出现明显 state/I/O bottleneck

结论是 Strata motivation 不能被视为现代 hybrid LLM 的统一问题，其适用范围依赖模型与 runtime 的实际 state behavior。

### 情况 D：两个模型均以 computation 为主

如果两个模型在合理 long-context、reuse 和 serving load 下始终没有明显 non-overlapped state stall，则说明当前模型/runtime/hardware 条件下原 Strata motivation 已显著弱化。

后续 I/O 优化即使在 microbenchmark 上有效，也必须重新证明其 end-to-end 价值。

## 14. 与后续泛化实验的边界

Experiment 4 只在 A100 主平台上完成跨模型 synthesis。

后续 Model × Hardware Generalization 再选择代表性 operating points，在 A100 40GB 与 L40 48GB 上交叉验证。

Experiment 4 不承担“attention architecture 导致差异”的因果证明，也不重复完整 hardware matrix。

统一 measurement definitions 见 [00-measurement-conventions.md](00-measurement-conventions.md)。
