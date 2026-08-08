# Experiment 4: Cross-Model Bottleneck Comparison

## 1. 实验定位

实验四不再完整重复实验一至实验三，而是对前三组结果进行统一归一化比较，并补充少量严格对齐的 matched configuration。

本实验用于回答 Strata 所关注的 KV / state bottleneck 在两种现代 hybrid 模型上是否具有稳定性，以及瓶颈的严重程度和出现条件是否存在明显模型差异。

本实验只支持跨模型稳定性和状态行为与性能之间的关联结论，不直接将差异归因于 attention architecture。

## 2. 实验目标

实验主要回答以下四个问题。

1. 两个模型在相同 workload 下产生的 cache/state pressure 是否处于相同量级。
2. context length、prefix reuse 和 serving load 增长时，两种模型的瓶颈是否沿相同方向演化。
3. 两个模型分别在什么条件下从 computation-dominated 转变为 state/I/O-dominated。
4. Strata 所针对的问题是两个模型共同存在的问题，还是明显依赖具体模型。

## 3. 数据来源

实验四主要复用实验一至实验三的结果。

实验一提供以下数据：

- context length；
- cache/state memory；
- prefill latency；
- CPU-GPU transfer；
- I/O stall；
- TTFT。

实验二提供以下数据：

- shared-prefix ratio；
- reusable state size；
- recomputation saving；
- state loading cost；
- warm/cold TTFT；
- reuse speedup。

实验三提供以下数据：

- request rate；
- throughput；
- active concurrency；
- transfer bandwidth；
- queueing；
- I/O stall；
- P50/P90/P99 TTFT。

因此，实验四属于 cross-model synthesis 与 matched validation，而不是第四套完整 workload sweep。

## 4. Matched Configuration

从前三组实验中选择少量具有代表性的配置，对两个模型重新进行严格对齐测试。

建议设置四类 operating points。

| Configuration | Context | Prefix reuse | Load |
|---|---:|---:|---|
| Short / light | 8K | 50% | Low |
| Long / light | 32K | 50% | Low |
| Long / high reuse | 32K | 75% | Low |
| Long / loaded | 32K | 50% | Near saturation |

如果某模型无法稳定支持某个绝对配置，则使用两个模型都能够稳定运行的最大公共配置。

实验四优先保证模型之间 workload 完全一致，而不是追求更极端的 context length。

## 5. 控制条件

所有 matched comparison 使用同一块 A100 40GB。

以下条件保持一致：

- context token 数量；
- shared-prefix ratio；
- unique suffix 长度；
- output length；
- request pattern；
- cache condition；
- warm-up procedure；
- measurement window。

模型使用各自原生的 cache/state representation，不为了让两个模型具有相同 state size 而人为修改模型。

本实验需要观察相同 serving workload 在不同现代模型上自然产生怎样不同的系统压力。

## 6. State Footprint Comparison

首先比较相同 context length 下两个模型产生的 cache/state footprint。

主要观察绝对 State Size，并计算：

```text
State Bytes per Context Token = State Size / Context Tokens
```

分别比较 8K、16K、32K。

这一部分用于判断两个模型对 context growth 的状态存储成本是否存在显著差异。

比较重点是 scaling slope，而不是单个 context point 的绝对显存。

## 7. State Movement Comparison

在相同 shared-prefix workload 下比较以下指标：

- reusable state size；
- CPU-GPU transfer volume；
- transfer time；
- I/O stall。

计算：

```text
Transfer Bytes per Reused Token = Transferred State Bytes / Reused Tokens
```

以及：

```text
I/O Stall Ratio = I/O Stall / TTFT
```

这一部分用于判断相同数量的 reusable context 在两个模型上是否产生类似的数据移动压力。

如果某模型需要搬运更少的 state，但 I/O stall 并没有同比减少，则说明瓶颈不能单纯由 state size 解释。

## 8. Reuse Efficiency Comparison

使用实验二中的 warm-cache 与 cold-recomputation 结果。

定义：

```text
Reuse Speedup = TTFT_cold / TTFT_warm
```

同时计算：

```text
Saved Prefill Time = T_cold,prefill - T_warm,prefill
```

以及：

```text
Reuse Efficiency = Net TTFT Saving / Saved Prefill Time
```

其中：

```text
Net TTFT Saving = TTFT_cold - TTFT_warm
```

该指标用于表示理论上通过 prefix reuse 节省的 computation，有多少真正转化为了 end-to-end latency improvement。

如果两个模型都减少了大量 prefill，但其中一个模型因为 state loading 成本较高，只兑现了较少的 TTFT 收益，则该差异作为实验四的重要观察结果。

## 9. Load Sensitivity Comparison

使用实验三结果比较两个模型进入 saturation 的过程。

由于两个模型基础 throughput 不同，不直接使用相同 requests/s 比较完整曲线。

定义：

```text
Normalized Load = Offered Request Rate / Measured Sustainable Capacity
```

比较不同 normalized load 区间，例如：

- 25% capacity；
- 50% capacity；
- 75% capacity；
- 90% capacity；
- saturation 附近。

重点观察：

- I/O stall ratio；
- queueing delay；
- TTFT；
- achieved throughput；
- transfer bandwidth utilization。

这一部分回答在接近自身 serving capacity 的程度相同时，哪个模型更早受到 state movement 限制。

## 10. Bottleneck Composition

选择四个 matched operating points，对 TTFT 进行统一组成分析。

至少分为：

```text
TTFT = Queueing + Computation + State/I/O + Other
```

Qwen3.5 和 Gemma 4 使用相同的分解定义。

结果按统一表格组织。

| Workload | Model | Computation | State/I/O | Queueing | Other |
|---|---|---:|---:|---:|---:|
| Short/light | Qwen3.5 | ... | ... | ... | ... |
| Short/light | Gemma 4 | ... | ... | ... | ... |
| Long/light | Qwen3.5 | ... | ... | ... | ... |
| Long/light | Gemma 4 | ... | ... | ... | ... |
| Long/loaded | Qwen3.5 | ... | ... | ... | ... |
| Long/loaded | Gemma 4 | ... | ... | ... | ... |

这一部分直接展示同一种 workload 下，两种现代模型真正由什么因素限制。

## 11. Bottleneck Transition Point

如果前三组实验能够观察到清晰趋势，则进一步确定两个模型的近似 bottleneck transition point。

可以使用统一 operational criterion，例如 state/I/O ratio 超过预先定义的比例，或者 I/O stall 出现明显非线性增长时，将该区域标记为 state pressure 明显区域。

不强行预设唯一理论阈值，而是根据实际曲线选择可解释且统一的 operational threshold。

两个模型必须使用相同判据。

最终比较以下 transition behavior：

- 多长 context 开始明显出现 state pressure；
- 多高 reuse 开始受到 loading cost 限制；
- 多高 normalized load 开始出现 I/O amplification。

最终形成两个模型的 bottleneck region map。

## 12. 最终结果组织

实验四建议形成四组核心结果。

### 图 1：Context Length → Normalized State Footprint

比较 state bytes / context token，用于展示两种模型的状态规模差异。

### 图 2：Shared Prefix → Reuse Benefit and I/O Cost

同时比较 reuse speedup 和 I/O stall ratio，用于展示不同模型能否有效兑现 prefix reuse 的计算收益。

### 图 3：Normalized Load → Bottleneck Growth

比较 TTFT、I/O stall ratio 和 queueing，用于展示两个模型进入 saturation 的方式。

### 图 4：Bottleneck Composition

选取代表性 workload，直接展示 computation、state/I/O、queueing 和 other 的组成。

该图作为实验四的核心总结图。

## 13. 结果判断逻辑

### 情况 A：两种模型趋势一致

两种模型虽然 state footprint 不同，但都表现为 context 增长导致 state pressure 上升、prefix reuse 受到 loading cost 限制、高负载进一步放大 I/O stall。

此时可以得出结论：Strata 所研究的问题并没有随着现代 hybrid 模型消失，并且至少在两种不同现代模型上具有跨模型稳定性。

### 情况 B：问题存在，但严重程度不同

两个模型都存在 state loading cost，但瓶颈出现的 context、reuse 或 load threshold 明显不同。

此时结论限定为问题仍然存在，但其严重程度和触发条件明显 model-dependent。

### 情况 C：只有一个模型明显存在问题

一个模型出现明显 state/I/O bottleneck，另一个模型始终主要由 computation 限制。

此时结论限定为 Strata motivation 不能被视为现代 hybrid LLM 的统一问题，其适用性依赖模型的具体 state behavior。

### 情况 D：两个模型都不再受 state movement 限制

如果两个模型在合理 long-context、reuse 和 serving load 下始终表现为 computation dominated，则说明在当前现代模型和硬件条件下，原 Strata 所强调的 KV/state loading bottleneck 已显著弱化。

这种结果意味着后续优化即使在 microbenchmark 中有效，其 end-to-end 价值仍需要重新评估。

## 14. 与前三组实验的关系

四个实验形成如下结构：

```text
Context Scaling
    ↓
Reuse Scaling
    ↓
Load Scaling
    ↓
Cross-Model Synthesis
```

实验四不再重新执行完整的前三组 sweep，而是利用前三组数据，并用少量严格 matched 的配置消除跨模型比较中的不一致。

最终回答的问题是：现代 hybrid LLM 上，Strata 所描述的 KV/state bottleneck 仍然存在到什么程度，以及该结论在不同现代模型之间是否稳定。

这一设计同时避免与后续更完整的模型与硬件泛化实验重复。