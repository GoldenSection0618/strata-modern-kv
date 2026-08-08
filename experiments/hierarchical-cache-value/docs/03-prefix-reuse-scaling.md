# Experiment 3: Prefix Reuse Scaling

## 1. 实验目标

本实验用于研究 prefix reuse 程度如何影响 hierarchical cache 的实际收益，并确定在什么复用条件下，将被驱逐的 cache/state 保存在 CPU 中比直接重新计算更划算。

实验固定模型、GPU cache budget、serving load、context distribution 和 cache policy，只系统改变 workload 中的 prefix reuse 程度。

本实验主要回答三个问题：

1. prefix reuse 增加时，hierarchical cache 能够获得多少额外 CPU cache hit；
2. 更高 reuse 是否能够持续降低 recomputation；
3. hierarchical cache 的端到端收益是否存在明确的 reuse threshold，以及收益在高 reuse 区间是否继续增长。

本实验不用于重新研究 GPU cache pressure。GPU cache budget 使用 Experiment 2 中已经验证能够产生稳定 eviction、但尚未进入极端 thrashing 的代表性配置。

## 2. 核心自变量

本实验唯一系统 sweep 的变量是 **prefix reuse level**。

Reuse level 从接近无共享逐步提高到高度共享，使实验覆盖 hierarchical cache 从几乎没有复用机会到大量状态重复访问的完整区间。

建议至少设置四个 reuse level：

| Reuse level | Workload 特征 |
|---|---|
| Low | 大部分请求使用不同 prefix，仅存在少量重复访问 |
| Medium | 一部分请求共享 prefix，存在稳定但有限的重复访问 |
| High | 大量请求共享历史 prefix，可复用状态被频繁再次访问 |
| Very High | workload 由少量热点 prefix 主导，重复访问高度集中 |

具体比例不预先写死。正式实验前根据 workload generator 的实际访问分布确定具体参数，并确保四个 level 在实际测量中形成清晰的 reuse 梯度。

## 3. Reuse 的定义

本实验中的 reuse 不只用“共享 token 比例”表示，而是同时控制两个相关维度：

- 一个 prefix 被多少个请求重复访问；
- 已被访问过的 prefix 在后续请求中重新出现的频率。

实验需要保证 reuse level 的改变主要来自访问重复程度，而不是同时改变平均 context length、请求数量或输出长度。

对于每个请求，需要能够标识其所属 prefix group，并记录该 prefix 在当前 workload trace 中的访问次数。

这样可以从请求轨迹直接计算实际 reuse，而不是只依赖 workload generator 的理论配置。

## 4. Workload 设计

实验使用多个 shared-prefix groups 构造请求集合。

每一个 group 包含一个共享 prefix 和多个不同的 unique suffix 或 query。

不同 reuse level 通过改变 prefix group 的复用次数和热点集中程度构造。

Low reuse 场景使用较多 prefix groups，每个 prefix 只被少量请求访问。

随着 reuse level 提高，prefix group 数量逐渐减少，同一 prefix 被更多请求重复访问。

Very High reuse 场景使用少量热点 prefix，使系统能够观察 hierarchical cache 在高度重复 workload 下的收益上限。

所有 reuse level 保持相同的总请求数量。

平均 input length、平均 output length 和整体 token volume 尽量保持一致。

这样可以避免高 reuse workload 因为输入本身更短或请求更少而天然获得性能优势。

## 5. GPU Cache Pressure 条件

本实验固定 GPU cache budget。

该 budget 从 Experiment 2 中选择一个具有代表性的 pressure point。

选择的配置需要满足以下条件：

- GPU-only 下已经存在稳定 eviction；
- hierarchical cache 存在使用 CPU tier 的机会；
- 系统尚未进入严重 cache thrashing；
- CPU-GPU traffic 尚未完全主导服务性能。

该设计使 Experiment 3 主要研究 reuse 是否值得保存，而不是重新研究显存是否足够。

如果两个模型的实际 cache/state footprint 差异较大，可以分别选择与其 working set 相匹配的 GPU cache budget。

跨模型不要求使用完全相同的绝对显存容量，而要求使用可比较的 cache pressure regime。

## 6. 对照配置

每一个 reuse level 都分别运行两种 cache architecture。

### GPU-only

系统只保留 GPU 中仍然驻留的 cache/state。

状态被 GPU cache 淘汰后不再保留。

后续请求再次访问相同 prefix 时，需要重新计算缺失状态。

### GPU + CPU hierarchical cache

GPU 使用与 GPU-only 完全相同的 cache budget。

被 GPU 驱逐的 reusable state 可以保存在 CPU tier。

后续请求再次访问相同 prefix 时，可以从 CPU 恢复已有状态。

因此每个 reuse level 都进行严格配对比较，使两种 cache architecture 面对完全相同的请求轨迹和 reuse opportunity。

## 7. Cache Initial State

本实验使用 warm-cache steady-state 作为主要测量条件。

正式测量前执行固定的 cache population workload，使 GPU 与 CPU cache 已经形成与当前 reuse level 对应的稳定工作状态。

Warm-up 阶段不计入正式结果。

每个 reuse level 使用与正式 workload 相匹配的 warm-up trace，避免使用统一预热过程导致不同 reuse workload 获得不一致的初始 cache state。

不同 reuse level 之间重新初始化 cache。

## 8. 实验执行过程

每个模型首先固定 GPU cache budget、context distribution、output length、request rate 和并发条件。

随后从 Low reuse 开始逐步提高 reuse level。

每个 reuse level 分别运行 GPU-only 和 hierarchical cache。

两种 cache architecture 使用完全相同的请求集合和请求顺序。

每轮实验先执行 warm-up，然后进入正式测量阶段。

正式阶段持续足够数量的请求，使 cache hit、eviction、CPU restore 和 recomputation 进入稳定状态。

每个配置进行多次独立重复实验。

不同配置的执行顺序可以随机化或交替执行，避免长期机器状态变化系统性偏向某一配置。

## 9. 核心测量指标

### 9.1 Actual Prefix Reuse

记录 workload 实际产生的 prefix reuse。

至少统计：

- unique prefix 数量；
- 每个 prefix 的访问次数；
- reused request 比例；
- reused token 或 reused state volume。

该结果用于确认实验实际形成了预期的 reuse 梯度。

### 9.2 GPU Cache Hit

记录不同 reuse level 下的 GPU cache hit。

Reuse 增加以后，一部分重复请求可能直接命中仍驻留 GPU 的状态，因此 hierarchical cache 的收益不能只根据总 reuse 判断。

GPU hit 用于区分“GPU 已经能够解决的 reuse”与“只有 CPU tier 才能保留的 reuse”。

### 9.3 CPU Cache Hit

Hierarchical 配置记录 CPU cache hit。

该指标是本实验最直接的中间变量。

实验需要观察 prefix reuse 增加后，有多少原本已经从 GPU 淘汰的状态真正通过 CPU tier 被再次利用。

如果 reuse 增加但主要转化为 GPU hit，而 CPU hit 没有明显增加，则说明额外 hierarchy 的边际价值有限。

### 9.4 Recomputation

记录 GPU-only 和 hierarchical 两种配置下的 recomputation。

GPU-only 的 recomputation 代表 GPU cache 无法覆盖重复访问时付出的额外计算成本。

Hierarchical 配置中的 recomputation 用于判断 CPU tier 实际避免了多少重复计算。

重点形成：

> Prefix reuse level → recomputation

并比较 GPU-only 与 hierarchical cache 两条曲线。

### 9.5 CPU-GPU Traffic

记录 hierarchical cache 中 CPU-resident state restore 引入的 CPU-GPU traffic。

随着 reuse 增加，CPU hit 可能增加，因此 transfer demand 也可能增加。

CPU-GPU traffic 需要与 recomputation reduction 配对分析。

如果 transfer 增长明显，但对应减少的 recomputation 很少，则 hierarchy 缺乏效率。

如果 transfer 增长伴随大量昂贵 recomputation 被消除，则 hierarchy 具有实际价值。

### 9.6 TTFT

记录不同 reuse level 下的 TTFT 分布。

GPU-only 场景中，高 reuse 不一定自动降低 TTFT，因为被复用状态可能已经被 GPU 淘汰。

Hierarchical cache 如果能够有效保留这些状态，应随着 reuse 增加获得更明显的 TTFT 改善。

实验至少报告 median、P90 和 P99 TTFT。

Tail latency 需要单独观察，因为 CPU restore 和 recomputation 可能对不同请求产生不均匀影响。

### 9.7 Throughput

记录 steady-state throughput。

如果 hierarchical cache 减少了大量重复 prefill/recomputation，GPU 计算资源可以用于处理更多新请求，因此 throughput 可能改善。

如果 CPU-GPU restore 成为主要瓶颈，则 throughput gain 可能在高 reuse 区域趋于平台。

## 10. 关键派生指标

### Reuse capture ratio

定义 hierarchical cache 实际捕获的可复用机会比例：

```text
reuse capture ratio
=
effective reused state
/
total reusable state opportunity
```

该指标用于区分 workload 理论上存在多少 reuse，与系统实际利用了多少 reuse。

### CPU-tier contribution

定义 CPU tier 对总有效 reuse 的贡献：

```text
CPU-tier contribution
=
CPU cache hit volume
/
(GPU cache hit volume + CPU cache hit volume)
```

该指标用于判断 reuse 增加以后，额外收益主要来自 GPU cache，还是来自 hierarchical CPU tier。

### Recomputation reduction

```text
recomputation reduction
=
recompute_GPU-only - recompute_hierarchical
```

用于量化 CPU tier 替代了多少重复计算。

### Hierarchy TTFT benefit

```text
TTFT benefit
=
TTFT_GPU-only - TTFT_hierarchical
```

正值表示 hierarchical cache 改善 TTFT。

### Throughput gain

```text
throughput gain
=
throughput_hierarchical / throughput_GPU-only - 1
```

用于量化 hierarchy 的整体 serving 收益。

## 11. 结果组织

实验最终至少形成五组结果。

### 结果一：Reuse 梯度验证

展示：

> Configured reuse level → actual prefix reuse

该结果确认 workload generator 实际产生了预期的访问模式。

### 结果二：Cache hierarchy 如何吸收 reuse

展示：

> Prefix reuse → GPU hit / CPU hit

用于判断 reuse 增加以后，新增复用机会最终落在哪一层 cache。

### 结果三：Reuse 对 recomputation 的影响

展示：

> Prefix reuse → recomputation

分别绘制 GPU-only 与 hierarchical cache。

该结果用于直接验证 hierarchy 是否将 workload reuse 转化为计算节省。

### 结果四：Computation 与 transfer trade-off

展示：

> Prefix reuse → recomputation reduction / CPU-GPU traffic

该结果用于判断保存并恢复状态是否比重新计算划算。

### 结果五：End-to-end value curve

展示：

> Prefix reuse → TTFT / throughput

同时比较 GPU-only 与 hierarchical cache。

该结果最终形成：

> **Prefix reuse → hierarchical cache value**

关系曲线。

## 12. 预期分析区域

实验结果可以按照 reuse 程度分为三个区域解释。

### 区域 A：Low reuse

大部分 prefix 不会被再次访问。

即使 CPU cache 保存了被 GPU 驱逐的状态，这些状态后续也很少命中。

CPU hit 较低。

GPU-only 与 hierarchical cache 的 recomputation、TTFT 和 throughput 接近。

该区域说明保存低复用状态缺乏价值。

### 区域 B：Moderate reuse

部分被 GPU 驱逐的 prefix 开始频繁重新出现。

CPU hit 增加。

Hierarchical cache 减少 GPU-only 中的 recomputation。

如果 CPU restore cost 低于对应计算成本，TTFT 和 throughput 开始出现稳定收益。

该区域是本实验最重要的 reuse value onset region。

### 区域 C：High reuse

大量请求访问少量热点 prefix。

Hierarchical cache 可以重复利用 CPU-resident state。

此时可能出现三种结果。

第一种结果是 hierarchy 收益继续扩大，因为大量重复计算被消除。

第二种结果是收益趋于平台，因为热点 prefix 已主要驻留 GPU，新增 reuse 不再增加 CPU-tier contribution。

第三种结果是 CPU restore 流量过高以后，transfer 成为新的限制因素，使收益不再增长。

## 13. 结果判断逻辑

### 情况 A：存在明显 reuse threshold

如果 Low reuse 下 hierarchy 基本无收益，而 Medium 或 High reuse 后 TTFT 和 throughput 明显改善，则可以说明 hierarchical cache 的价值依赖足够高的 prefix reuse。

该结果能够给出 hierarchy 的基本 workload applicability boundary。

### 情况 B：较低 reuse 已经出现收益

如果 Medium 甚至更低 reuse 下 CPU tier 已经稳定减少 recomputation 并改善性能，则说明 hierarchical cache 不需要非常强的热点 workload 才具有价值。

该结果意味着 hierarchy 的适用范围较宽。

### 情况 C：Reuse 很高，但 hierarchy 收益有限

如果实际 prefix reuse 很高，但 CPU hit 较低，则需要进一步判断大量 reuse 是否已经被 GPU cache 捕获。

如果 CPU hit 较高但性能仍无改善，则说明 restore cost 抵消了 recomputation reduction。

这两种情况必须区分，不能统一解释为“hierarchy 无价值”。

### 情况 D：收益在高 reuse 后趋于平台

如果随着 reuse 提高，hierarchy 收益先快速增长后趋于稳定，则需要结合 GPU hit 与 CPU-tier contribution 判断原因。

如果热点 prefix 逐渐稳定驻留 GPU，则说明 GPU cache 已捕获主要 reuse。

如果 CPU traffic 持续增加而 throughput 不再改善，则说明 hierarchy 开始受到 transfer bottleneck 限制。

## 14. 与前两个实验的关系

Experiment 1 回答：

> 在代表性条件下，hierarchical cache 是否存在基础收益。

Experiment 2 回答：

> GPU cache pressure 增加以后，hierarchy 从什么容量压力区间开始具有价值。

Experiment 3 回答：

> 在 GPU cache 已存在容量压力时，workload 需要具有多少 reuse，CPU tier 才值得保存这些被驱逐状态。

三组实验分别控制：

```text
Experiment 1: architecture + initial state
Experiment 2: GPU cache pressure
Experiment 3: prefix reuse
```

这种设计避免多维 sweep，使每一组实验都对应一个明确的主要变量。

## 15. 实验边界

本实验只系统改变 prefix reuse。

GPU cache budget 固定。

Request rate 与 concurrency 固定。

平均 context length 与 output length 固定。

Scheduler strategy 固定。

CPU cache capacity 保持足够，不主动引入 CPU-tier eviction pressure。

本实验不直接将 Qwen3.5 与 Gemma 4 的结果差异归因于 attention architecture。

最终目标是建立一条清晰的：

> **Prefix reuse → CPU-tier reuse → recomputation reduction → transfer cost → TTFT / throughput benefit**

证据链，并据此判断 hierarchical cache 在现代 workload 下需要什么程度的 context reuse 才真正值得使用。
