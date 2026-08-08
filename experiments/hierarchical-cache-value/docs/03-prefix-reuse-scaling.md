# Experiment 3: Prefix Reuse Scaling

## 1. 实验目标

本实验用于研究 prefix reuse opportunity 如何影响 hierarchical cache 的实际收益，并确定在已有 GPU cache pressure 时，什么程度的 prefix revisit 才值得把被驱逐状态保存在 CPU。

实验固定模型、GPU cache budget、cache locality structure、request ordering、serving load、context/output distribution 和 scheduler policy，只系统改变 **prefix revisit fraction**。

本实验主要回答三个问题：

1. prefix revisit 增加时，CPU tier 能够捕获多少新增 reusable state；
2. 更高 reuse 是否稳定减少 recomputation；
3. hierarchical cache 是否存在清晰的 reuse value-onset region，以及高 reuse 下收益是否因 GPU residency 或 restore traffic 而趋于平台。

所有配置遵循 [`00-common-conventions.md`](00-common-conventions.md)。

## 2. 实验对象

本实验在 Experiments 1–3 的 validated primary model 上执行完整 sweep，默认平台为 A100 40GB。

本实验不在两个模型上重复完整 reuse sweep。Experiment 4 只在第二个模型上复验少量代表性 reuse 配置。

## 3. 核心自变量

本实验唯一系统 sweep 的变量是 **prefix revisit fraction**。

请求 trace 预先定义一组 eligible revisit slots。高 reuse trace 在这些位置重新访问此前出现过的 prefix。低 reuse trace 则把其中一部分 revisit 替换为长度、位置和 suffix 结构匹配的 unique prefix。

建议主实验至少使用四个有明显间隔的 reuse level，例如：

- 0% 或接近 0% revisit，作为 negative control；
- low revisit；
- medium revisit；
- high revisit。

具体比例在 workload calibration 后冻结，并写入版本控制配置。正式结果使用实际 request-weighted 和 token/state-volume-weighted reuse 指标，而不是只使用 `Low/Medium/High` 标签。

## 4. 为什么不再同时改变 hotspot concentration

旧设计同时通过减少 prefix-group 数量、提高单个 prefix 访问次数来增加 reuse。这会同时改变：

- reuse opportunity；
- hotspot concentration；
- reuse distance / cache locality。

这样无法判断 hierarchy 收益究竟来自更多 reuse，还是来自更好的 locality。

因此本实验只改变 revisit 是否发生。Prefix placement、eligible revisit slots 和 reuse-distance pattern 保持固定。

Hotspot concentration、cache distance、shuffle 和 request reordering 留到独立的 Cache Locality and Scheduler Behavior 实验组。

## 5. GPU Cache Pressure 条件

本实验固定 GPU cache budget。

该 budget 从 Experiment 2 的结果中预先选择一个代表性的 **value-onset 或 moderately high pressure** point。选择标准是：

- GPU-only 已存在稳定 reusable-state eviction；
- active-request preemption 为 0；
- hierarchy 存在使用 CPU tier 的机会；
- restore traffic 尚未完全主导服务性能。

该 budget 在 Experiment 3 的所有 reuse levels 中保持不变。

## 6. Workload 设计

所有 reuse levels 保持：

- 相同总请求数；
- 相同 prefix length distribution；
- 相同 input/output token distribution；
- 相同 eligible revisit slot 位置；
- 相同 request ordering；
- 相同 offered load 和 concurrency target；
- 相同 prefix-group size template；
- 相同 scheduler policy。

低 reuse trace 通过把部分 revisit prefix 替换为 matched unique prefix 构造，不通过重新排序请求或集中到更少热点 prefix 构造。

每个请求记录 prefix identifier、是否为 revisit、可复用 prefix token 数以及 reuse distance。这样可以验证不同 trace 除 reuse opportunity 外没有出现显著 locality drift。

## 7. 对照配置

每个 reuse level 严格配对运行：

- **GPU-only**；
- **GPU + CPU hierarchical cache**。

两种 architecture 使用相同 GPU cache budget、CPU-independent request trace 和 serving load。

## 8. Cache initial state

主实验使用 **warm-cache steady-state**。

每个 reuse level 先执行与该 trace 配套的 cache-population phase，再验证实际 cache occupancy/residency，随后开始正式测量。

不同 reuse level 之间重新初始化 cache。

## 9. Validity conditions

主结果 run 必须满足：

- full-hierarchy restore 已通过验证；
- active-request preemption 为 0；
- CPU tier 不发生未控制 capacity eviction；
- actual prefix length、request size、offered load 和 reuse-distance distribution 与目标匹配；
- 除 revisit fraction 外，没有通过 request reordering 或热点集中改变 locality；
- CPU restore failure 不静默回退并被计为 CPU hit。

## 10. 实验执行过程

从最低 reuse level 开始逐步提高 revisit fraction。

每个 level 分别运行 GPU-only 与 hierarchical。配对 architecture 的执行顺序交替或随机化。

每轮执行固定 warm-up，然后运行足够长的正式 trace，使 GPU hit、CPU hit、recomputation、restore 和 throughput 进入稳定状态。

每个配置进行多次独立重复实验。

## 11. 核心测量指标

### 11.1 Actual reuse

至少记录：

- revisit request fraction；
- reusable prefix token/state volume；
- unique prefix 数量；
- reuse-distance distribution。

该结果用于验证本实验真正改变的是 reuse opportunity，而不是 locality。

### 11.2 GPU cache hit / eviction

记录 GPU hit 和 eviction volume，区分新增 reuse 是被 GPU 直接捕获，还是需要 CPU tier 才能保留。

### 11.3 CPU cache hit

Hierarchical 配置记录 CPU hit volume，并尽可能按 state group 分项。

### 11.4 Recomputation

记录 GPU-only 与 hierarchical 中的 recomputation，并计算 CPU-tier reuse 避免的 recomputation。

### 11.5 CPU-GPU traffic / stall

记录 CPU restore traffic、transfer activity 和能够测量时的 non-overlapped restore stall。

### 11.6 Serving performance

记录：

- median / P90 / P99 TTFT；
- steady-state throughput；
- achieved request rate；
- active-request preemption count。

## 12. 派生指标

### Reuse capture ratio

```text
reuse capture ratio
=
effective reused token/state volume
/
total reusable opportunity
```

### CPU-tier contribution

```text
CPU-tier contribution
=
CPU hit volume
/
(GPU hit volume + CPU hit volume)
```

### Recomputation reduction

```text
recomputation reduction
=
recompute_GPU-only - recompute_hierarchical
```

### Relative TTFT improvement

```text
relative TTFT improvement
=
(TTFT_GPU-only - TTFT_hierarchical)
/
TTFT_GPU-only
```

### Throughput gain

```text
throughput gain
=
throughput_hierarchical / throughput_GPU-only - 1
```

所有派生指标必须保留绝对值和 raw measurements。

## 13. 结果组织

实验至少形成五组结果：

1. **Configured revisit fraction → actual reuse + reuse-distance validation**；
2. **Prefix reuse → GPU hit / CPU hit / eviction**；
3. **Prefix reuse → recomputation**，比较 GPU-only 与 hierarchical；
4. **Prefix reuse → recomputation reduction + restore traffic/stall**；
5. **Prefix reuse → TTFT / throughput benefit**。

最终得到：

> **Prefix reuse opportunity → hierarchical cache value**

关系曲线。

## 14. 结果判断逻辑

### 区域 A：Low reuse

大量 eligible requests 使用 unique prefix，CPU tier 很少再次命中。GPU-only 与 hierarchical 的 recomputation 和性能接近。

这说明保存低复用状态缺乏价值。

### 区域 B：Reuse value onset

随着 revisit 增加，被 GPU 驱逐的 prefix 开始稳定从 CPU 命中。Recomputation reduction 增加，并在 restore cost 足够低时转化为 TTFT / throughput 收益。

这一位置定义 hierarchy 的 **reuse value-onset region**。

### 区域 C：High reuse

收益可能继续扩大，也可能趋于平台。

如果新增 revisit 主要转化为 GPU hit，CPU-tier contribution 会下降或停止增长，说明热点状态已经常驻 GPU。

如果 CPU hit 和 restore traffic 继续增长但端到端收益停止增长，则说明 data movement / stall 开始限制 hierarchy。

### 情况 D：Reuse 增加但 CPU hit 仍低

首先检查新增 revisit 是否被 GPU 直接捕获。如果 GPU hit 同时上升，则 hierarchy 边际价值有限是合理结果。如果 GPU/CPU hit 都未捕获预期 reuse，则优先检查 runtime/cache-policy validity，而不是解释为模型性质。

## 15. 与其他实验的关系

Experiment 2 隔离 GPU capacity pressure，并为本实验选定固定 pressure point。

Experiment 3 只隔离 reuse opportunity，不改变 cache locality。

Cache distance、shuffle、hotspot concentration 和同 context 高并发由后续 Cache Locality and Scheduler Behavior 实验组研究。

Experiment 4 只在第二个模型上复验少量代表性 reuse/pressure point，不重复完整 sweep。

## 16. 实验边界

本实验只系统改变 prefix revisit fraction。

GPU cache budget、request ordering、reuse-distance template、hotspot structure、request rate、concurrency target、context/output distribution、scheduler strategy 和 hardware 均保持固定。
