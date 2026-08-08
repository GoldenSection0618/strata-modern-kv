# Experiment 1: Hierarchical Cache Baseline Benefit

## 1. 实验目标

本实验用于建立 hierarchical cache 的基础收益基线。

实验在相同模型、相同请求负载和相同 GPU cache budget 下，对比 **GPU-only cache** 与 **GPU + CPU hierarchical cache** 两种系统配置，并分别考察 cold-cache 与 warm-cache 场景。

本实验主要回答三个问题：

1. 当 GPU HBM 无法长期保存全部可复用 context/state 时，将被驱逐状态保留在 CPU 是否比直接丢弃并重新计算更有收益；
2. hierarchical cache 的收益是否主要来自减少 recomputation；
3. CPU-GPU transfer cost 是否会抵消额外 cache reuse 带来的收益。

本实验不负责确定 hierarchy 在所有 cache pressure 与 reuse ratio 下的完整适用边界。GPU cache pressure 和 prefix reuse 的系统 sweep 留到后续实验完成。

## 2. 对照配置

实验设置两种 cache architecture。

### GPU-only

系统只使用 GPU HBM 保存可复用 cache/state。GPU cache 被淘汰后，对应状态不再保留。后续再次访问相同 prefix 时，需要重新计算已经失效的部分。

### GPU + CPU hierarchical cache

系统同时使用 GPU 与 CPU 保存可复用 cache/state。GPU 中被淘汰但仍具有复用价值的状态可以保留在 CPU tier。后续再次访问相同 prefix 时，可以从 CPU 恢复对应状态，而不是重新计算全部内容。

两种配置使用相同的 GPU cache budget，使实验比较的是 CPU hierarchy 本身带来的增量价值，而不是更大的 GPU cache 容量。

## 3. Cache initial state

实验分别设置 cold-cache 与 warm-cache 两种初始状态。

### Cold-cache

每轮测试开始前清空待测 cache/state，使正式 workload 从无历史复用状态开始运行。

Cold-cache 用于观察 hierarchy 是否能够随着请求执行逐步积累可复用状态，并最终减少后续 recomputation。

### Warm-cache

正式计时前执行固定的 cache population workload，使系统已经积累一定数量的可复用 context/state。

预热阶段不计入正式性能结果。GPU-only 与 hierarchical 两种配置使用完全相同的预热请求与访问顺序。

Warm-cache 用于模拟长期运行的 serving 系统，并判断 CPU tier 是否能够扩大可长期保留的有效 working set。

## 4. 实验矩阵

基础实验形成四种主要配置：

| Cache architecture | Initial state |
|---|---|
| GPU-only | Cold |
| GPU-only | Warm |
| Hierarchical cache | Cold |
| Hierarchical cache | Warm |

除 cache architecture 与 initial state 外，其余条件保持一致。

## 5. Workload 设计

实验使用具有明确 prefix reuse 的请求集合。

请求集合由多个 shared-prefix groups 构成。同一 group 内的请求共享主要输入 context，并具有不同的后续 query 或 generation 部分。不同 group 之间相互独立，避免所有请求集中在单个 prefix 上形成过度理想化的 workload。

本实验固定 prefix reuse 水平和 GPU cache pressure，使系统具有实际 eviction 与 reuse 机会，同时避免在基础实验中引入多维 sweep。

各配置使用完全相同的请求集合和请求顺序，使 GPU-only 与 hierarchical cache 面对相同的 reuse opportunity 与访问轨迹。

输出长度和 serving load 保持固定，避免 decode workload 与 queueing 成为主要干扰变量。

## 6. Cold-cache 实验流程

Cold-cache 实验在每轮正式测试前恢复到空 cache 状态。

第一阶段请求从无历史状态开始执行。随着请求继续到达，GPU cache 逐步建立并产生容量压力。

当可复用状态被 GPU cache 淘汰后，GPU-only 配置直接失去这些状态，hierarchical 配置则允许其中可复用部分进入 CPU tier。

后续再次访问相同 prefix 时，比较两种配置的 cache hit、recomputation、CPU-GPU traffic、TTFT 与 throughput。

Cold-cache 结果同时观察整个运行阶段和随请求进程变化的指标，避免初始 cache population 阶段掩盖后期 hierarchy 收益。

本部分主要判断：

> hierarchical cache 是否能够在系统运行过程中逐步积累复用收益，以及这种收益是否最终反映到用户可见性能上。

## 7. Warm-cache 实验流程

Warm-cache 实验首先执行固定的 cache population 阶段。

预热阶段结束后立即执行正式 workload，并保持已经形成的 cache 状态不被人为清除。

GPU-only 与 hierarchical 配置经历相同的历史访问过程，但由于 cache architecture 不同，两者在正式 workload 开始时实际保留下来的 reusable working set 可以不同。

正式阶段继续使用与 cold-cache 相同结构的 workload，并比较两种 cache architecture 的 cache reuse、recomputation、CPU-GPU traffic、TTFT 与 throughput。

本部分主要判断：

> 当 serving 系统已经积累大量历史 context/state 后，CPU cache 是否能够显著扩大可复用状态的有效容量，并产生稳定的端到端收益。

## 8. 核心观测指标

实验同时从 cache、计算、数据移动和服务性能四个层面记录指标。

### 8.1 Cache reuse

记录 GPU cache hit 与 CPU cache hit。

GPU cache hit 用于确认两种配置在相同 GPU budget 下的基础复用行为。CPU cache hit 用于量化 hierarchical cache 额外保留下来的可复用状态。

### 8.2 Recomputation

记录因 cache miss 导致的 recomputation。

重点比较 hierarchical cache 是否通过 CPU-resident reuse 稳定减少 GPU-only 配置中的重复计算。

### 8.3 CPU-GPU traffic

记录 hierarchical cache 引入的 CPU-GPU 数据移动。

该指标用于衡量减少 recomputation 所付出的额外 transfer cost，并与最终 TTFT 和 throughput 一起分析。

### 8.4 TTFT

记录请求的 TTFT 分布。

TTFT 用于判断 cache reuse 是否真正降低请求在生成首 token 之前的等待成本，而不是只改善内部 cache statistics。

### 8.5 Throughput

记录稳定阶段的系统 throughput。

Throughput 用于判断 hierarchy 在整体 serving 层面是否提升有效处理能力。

## 9. 实验执行方式

每个配置进行多次独立重复测量。

每轮正式实验前恢复到规定的 cold-cache 或 warm-cache 初始状态，避免上一轮运行污染下一轮结果。

各配置使用相同 workload trace、相同请求数量和相同实验结束条件。

正式测量前执行必要的非计时初始化，使模型加载、首次内存分配和首次执行开销不进入结果。

正式结果同时报告中心趋势与运行间波动，不依赖单次运行结果。

每次 run 记录完整 metadata，包括模型、硬件、runtime、cache policy、GPU cache budget、workload 配置、initial state 与 repetition index。

## 10. 结果组织

实验最终至少形成以下结果：

1. GPU-only 与 hierarchical cache 的 GPU/CPU cache hit 对比；
2. GPU-only 与 hierarchical cache 的 recomputation 对比；
3. hierarchical cache 引入的 CPU-GPU traffic；
4. cold-cache 下两种 cache architecture 的 TTFT 与 throughput；
5. warm-cache 下两种 cache architecture 的 TTFT 与 throughput；
6. cold-cache 运行过程中 cache reuse 与性能收益随请求进程的变化。

结果分析必须将 cache reuse、recomputation reduction、transfer cost 与端到端性能放在同一证据链中解释。

## 11. 结果判断逻辑

### 情况 A：Hierarchical cache 产生明确正收益

如果 hierarchical cache 获得稳定 CPU cache hit，同时显著减少 recomputation，并最终改善 TTFT 或 throughput，则说明现代 hybrid 模型在当前 workload 下仍然存在 hierarchical context caching 的基础价值。

### 情况 B：Reuse 存在，但 transfer cost 抵消收益

如果 CPU cache hit 与 recomputation reduction 明显存在，但 TTFT 和 throughput 没有改善，则说明 hierarchy 在语义上具有复用价值，但 CPU-GPU transfer cost 已经抵消计算节省。

### 情况 C：额外 CPU tier 基本没有被利用

如果 CPU cache hit 很低，hierarchical 与 GPU-only 的 recomputation 和端到端性能接近，则说明当前 workload 的主要 working set 已经被 GPU cache 覆盖，额外 CPU hierarchy 缺乏基础收益空间。

### 情况 D：Cold-cache 与 warm-cache 结论不同

如果 cold-cache 收益有限而 warm-cache 收益明显，则说明 hierarchy 的价值依赖历史状态积累和长期 reuse，而不是单次请求路径本身。

如果 warm-cache 仍然没有收益，则后续 GPU cache pressure 与 prefix reuse sweep 需要判断是 workload reuse 不足、GPU cache 已足够，还是 transfer cost 过高。

## 12. 实验边界

本实验只比较 cache architecture 与 cache initial state。

GPU cache pressure 不在本实验中系统 sweep。Prefix reuse ratio 不在本实验中系统 sweep。模型架构差异也不在本实验中做因果归因。

这些变量将在后续实验中独立控制，以判断 hierarchical cache 的收益边界及其跨模型稳定性。
