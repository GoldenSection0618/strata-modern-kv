# Experiment 2: GPU Cache Pressure Scaling

## 1. 实验目标

本实验用于研究 GPU HBM cache pressure 对 hierarchical cache 收益的影响，并确定 CPU tier 的价值是否只在显存极度紧张时出现，还是在常规高负载 serving 条件下已经能够产生稳定收益。

实验固定模型、workload 结构、prefix reuse 程度和请求负载，只系统改变 GPU 可用于 cache/state 的容量预算。

本实验主要回答三个问题：

1. GPU cache 容量降低时，GPU-only 系统的 eviction 和 recomputation 如何变化；
2. hierarchical cache 能否通过 CPU tier 吸收被 GPU 驱逐的可复用状态；
3. hierarchical cache 的端到端收益从什么 cache pressure 区间开始出现，并在压力继续增大后如何变化。

本实验不用于研究 prefix reuse 本身的影响。Prefix reuse ratio 在本实验中保持固定，其独立作用留到 Experiment 3。

## 2. 核心自变量

本实验唯一系统 sweep 的变量是 **GPU cache budget**。

GPU cache budget 从宽松状态逐步降低到明显不足状态，使实验覆盖从“GPU 本身足够容纳主要 working set”到“频繁发生 eviction”的完整区间。

建议至少设置四个压力等级：

| Pressure level | GPU cache 状态 |
|---|---|
| Low | GPU cache 可以容纳绝大部分活跃 working set |
| Medium | GPU cache 接近 working set 大小，开始出现稳定 eviction |
| High | GPU cache 明显小于 working set，频繁发生 eviction |
| Very High | GPU cache 只能保留少量近期状态，recomputation 或 CPU restore 成为常态 |

具体容量数值不预先写死，而是在正式实验前根据两个模型的实际 cache/state footprint 和 workload working set 确定。

各压力点需要形成明确的容量梯度，并保证能够观察到从低 eviction 到高 eviction 的连续变化。

## 3. 对照配置

每一个 GPU cache pressure point 都分别运行两种 cache architecture。

### GPU-only

系统只保留 GPU cache。

当 GPU cache 空间不足时，被淘汰状态直接失效。后续再次访问对应 prefix 时，需要重新计算缺失部分。

### GPU + CPU hierarchical cache

系统保持与 GPU-only 完全相同的 GPU cache budget，同时允许被 GPU 淘汰的可复用状态进入 CPU cache。

后续再次访问相同 prefix 时，系统可以从 CPU tier 恢复状态。

因此每个 pressure point 都形成一组严格配对实验：

```text
GPU cache budget X
├── GPU-only
└── GPU + CPU hierarchical
```

这种设计保证观察到的差异来自 CPU hierarchy，而不是 GPU cache 容量不同。

## 4. Workload 设计

实验使用固定的 shared-prefix workload。

请求由多个 prefix groups 构成，同一 group 内存在稳定的 prefix reuse，不同 group 之间保持独立。

整个请求集合的 active working set 需要大于最低几个 GPU cache budget，使高压力配置能够稳定产生 eviction。

同时，working set 不能远大于所有 GPU cache budget，否则所有实验点都会处于极端 cache thrashing 状态，无法观察收益出现的转折过程。

Experiment 2 使用固定的 prefix reuse 程度。该 reuse 水平应足以让被驱逐状态在后续请求中存在再次访问机会，但不能设计成所有请求都命中同一个 prefix。

请求顺序在所有配置之间完全一致。

Request rate、output length、context distribution 和并发条件保持固定，避免系统负载变化与 cache pressure 同时变化。

## 5. 初始状态

本实验以 **warm-cache steady-state** 作为主实验条件。

正式测量前使用固定 workload 建立 cache working set，使系统进入持续发生 cache allocation、hit、eviction 和 restore 的稳定运行阶段。

Warm-cache 条件更适合本实验，因为 Experiment 2 关注的是容量不足导致的长期 cache competition，而不是系统从空 cache 开始建立状态的过程。

Cold-cache 不进行完整 pressure sweep。

如果需要确认 warm initialization 没有引入异常，可以在代表性 pressure point 上保留少量 cold-cache validation，但不作为本实验主要结果。

## 6. 实验执行过程

每个模型首先确定一个固定 workload working set。

随后从最高 GPU cache budget 开始逐步降低容量。

在每一个 cache budget 下，先运行 GPU-only，再运行 hierarchical cache，并保持完全相同的 workload trace。

每轮实验先执行固定的 warm-up/cache population 阶段，然后进入正式测量阶段。

正式测量持续足够数量的请求，使 cache hit、eviction、recomputation、restore 和 throughput 进入稳定状态。

不同 cache budget 的实验之间重新初始化 cache 状态，避免前一个容量配置影响后一个配置。

所有 pressure point 进行多次独立重复实验。

## 7. 核心测量指标

### 7.1 GPU cache hit

记录不同 GPU cache budget 下的 GPU cache hit。

该指标用于确定 cache pressure 是否确实随着 GPU budget 降低而增强。

理论上，随着 GPU budget 减少，GPU hit 应整体下降，但实际变化形态由 workload locality 和模型 state behavior 决定。

### 7.2 GPU eviction

记录 GPU cache eviction 的数量或对应的有效 state volume。

Eviction 是本实验判断 cache pressure 的直接指标。

实验结果需要验证人为设置的 Low、Medium、High、Very High pressure 是否实际对应逐渐增强的 eviction，而不能只根据配置的显存比例进行命名。

### 7.3 CPU cache hit

Hierarchical 配置记录 CPU cache hit。

重点观察 GPU eviction 增加以后，被驱逐状态中有多少能够在后续请求中重新命中 CPU tier。

如果 eviction 增加但 CPU hit 没有同步增加，则说明被驱逐状态缺乏后续 reuse，此时增加 CPU cache 不一定具有价值。

### 7.4 Recomputation

记录由于 reusable state 不存在而触发的 recomputation。

GPU-only 配置用于观察 cache pressure 增加后重复计算成本如何增长。

Hierarchical 配置用于观察 CPU restore 能够避免其中多少 recomputation。

核心结果形成：

> GPU cache pressure → recomputation volume/cost

并同时比较 GPU-only 与 hierarchical cache。

### 7.5 CPU-GPU traffic

记录 hierarchical cache 的 CPU→GPU restore traffic。

随着 GPU cache budget 降低，更多 reusable state 可能进入 CPU tier，因此 CPU-GPU traffic 预计增加。

该指标必须与 recomputation reduction 一起观察。

CPU traffic 增加本身不是负面结果，关键问题是它所替代的计算成本是否更高。

### 7.6 TTFT

记录每一个 GPU cache pressure point 下的 TTFT 分布。

重点观察 GPU-only 在 cache pressure 增加后是否因为 recomputation 导致 TTFT 恶化，以及 hierarchical cache 是否能够延缓或减轻这种恶化。

结果至少需要比较 median 和 tail latency。

### 7.7 Throughput

记录 steady-state throughput。

Throughput 用于判断 CPU hierarchy 在长期运行状态下是否能够降低重复计算造成的 GPU resource consumption，并提升整体 serving capacity。

## 8. 关键派生指标

为了使不同 cache pressure point 更容易比较，可以定义以下派生量。

### Recomputation reduction

```text
recomputation reduction
=
recompute_GPU-only - recompute_hierarchical
```

用于直接表示 CPU tier 避免了多少重复计算。

### Hierarchy latency benefit

```text
TTFT benefit
=
TTFT_GPU-only - TTFT_hierarchical
```

正值表示 hierarchy 带来 latency 改善。

### Throughput benefit

```text
throughput gain
=
throughput_hierarchical / throughput_GPU-only - 1
```

用于表示 hierarchy 的相对吞吐收益。

这些指标都需要与原始测量值同时保存，不能只保留归一化结果。

## 9. 结果组织

实验最终至少形成四组主要结果。

### 结果一：Cache pressure 是否实际建立

展示：

> GPU cache budget → GPU hit / eviction

该结果验证自变量是否真正改变了系统 cache pressure。

### 结果二：Hierarchy 是否扩大有效 cache capacity

展示：

> GPU cache budget → GPU hit / CPU hit / total effective reuse

用于判断 GPU cache 不足以后，CPU tier 是否实际接管了一部分 reusable working set。

### 结果三：Recomputation 与 traffic trade-off

展示：

> GPU cache budget → recomputation reduction + CPU-GPU traffic

用于判断 hierarchy 是通过什么代价减少重复计算。

### 结果四：End-to-end benefit

展示：

> GPU cache budget → TTFT / throughput

同时比较 GPU-only 与 hierarchical cache。

这一结果用于得到 hierarchy value curve。

## 10. 预期分析重点

本实验最重要的结果不是寻找一个绝对最优 GPU cache size，而是识别 hierarchical cache 收益随 cache pressure 变化的整体形态。

理想情况下可以观察到三个区域。

### 区域 A：GPU cache 充足

GPU hit 较高，eviction 很少。

GPU-only 已经能够保存主要 working set，因此 hierarchical cache 的 CPU hit 很少。

两种系统的 TTFT 和 throughput 接近。

该区域说明 CPU tier 没有明显必要性。

### 区域 B：GPU cache 开始不足

GPU eviction 增加。

GPU-only recomputation 开始明显增长。

Hierarchical cache 可以通过 CPU hit 保存部分被驱逐状态，并降低 recomputation。

如果 CPU restore cost 小于重新计算成本，则 hierarchy 的 TTFT 和 throughput 收益开始出现。

这一阶段是本实验最关键的 **value onset region**。

### 区域 C：GPU cache 严重不足

GPU eviction 与 CPU restore 都非常频繁。

此时可能出现两种结果。

第一种结果是 hierarchy 收益继续扩大，因为 CPU tier 避免了大量 expensive recomputation。

第二种结果是 hierarchy 收益达到平台甚至下降，因为 CPU-GPU traffic 自身成为新的瓶颈。

两种结果都具有明确的系统意义。

## 11. 结果判断逻辑

### 情况 A：只有高 pressure 下 hierarchy 才有收益

如果 Low 和 Medium pressure 下两种方案接近，而 High pressure 后 hierarchical cache 明显改善 TTFT 或 throughput，则可以说明：

> hierarchical cache 主要是 GPU HBM shortage 下的容量扩展机制，其价值取决于 cache working set 是否超过 GPU 容量。

### 情况 B：Medium pressure 已出现稳定收益

如果 GPU cache 尚未进入极端 thrashing 时 hierarchy 已经产生稳定收益，则说明 CPU tier 不只是 OOM 或极端容量不足的兜底方案，而可能在正常高负载 serving 中具有实际价值。

### 情况 C：Cache pressure 增加，但 hierarchy 始终没有性能收益

如果 CPU hit 与 recomputation reduction 存在，但端到端性能没有改善，则说明 CPU-GPU restore overhead 抵消了计算节省。

这种结果意味着 hierarchy 的瓶颈已经从 cache capacity 转移到 data movement efficiency。

### 情况 D：CPU hit 始终很低

如果 GPU eviction 增加，但 CPU hit 仍然很低，则说明当前 workload 被淘汰状态缺乏后续 reuse。

此时 hierarchical cache 无收益的主要原因不是 restore 太慢，而是缺乏值得保存的 reusable state。

## 12. 与 Experiment 1 的关系

Experiment 1 回答：

> 在一个固定且具有代表性的 cache pressure 条件下，hierarchical cache 是否存在基础收益。

Experiment 2 进一步回答：

> 这种收益是否由 GPU HBM pressure 驱动，以及收益从什么压力区间开始出现。

因此 Experiment 2 不重复 cold/warm cache 的完整对照，而以 warm steady-state 为主，系统 sweep GPU cache budget。

Experiment 1 建立“有没有价值”的基础证据，Experiment 2 建立“什么时候因为 GPU cache 不足而变得有价值”的条件边界。

## 13. 实验边界

本实验只系统改变 GPU cache budget。

Prefix reuse ratio 保持固定。

Request rate 和 concurrency 保持固定。

Cache locality pattern 保持固定。

Scheduler strategy 保持固定。

CPU cache capacity 应保证不是本实验中的主要约束，否则实验会同时改变 GPU pressure 和 CPU capacity pressure。

模型之间的结果可以分别报告，但本实验不根据模型差异直接推断 attention architecture 的因果作用。

最终目标是得到一条清晰的：

> **GPU cache pressure → hierarchical cache value**

关系曲线，为后续 prefix reuse 实验和最终系统价值判断提供基础。
