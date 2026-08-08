# Experiment 1: Hierarchical Cache Baseline Benefit

## 1. 实验目标

本实验用于建立 hierarchical cache 的基础收益基线。

实验在相同模型、相同 workload、相同 GPU cache budget 和相同 serving load 下，对比 **GPU-only** 与 **GPU + CPU hierarchical cache**，并分别考察 cold-cache 与 warm-cache。

本实验主要回答三个问题：

1. 当 GPU 无法长期保留全部 reusable state 时，CPU tier 是否能够减少后续 recomputation；
2. 这种 recomputation reduction 是否能够转化为 TTFT 或 throughput 收益；
3. CPU-GPU restore traffic 和 non-overlapped stall 是否会抵消复用收益。

本实验不负责确定完整的 cache-pressure 或 reuse 适用边界。GPU cache pressure 和 prefix reuse 分别由 Experiments 2 和 3 隔离研究。

所有配置遵循 [`00-common-conventions.md`](00-common-conventions.md)。

## 2. 实验对象与执行平台

Experiments 1–3 使用一个通过 full-hierarchy validation gate 的 primary model，在 A100 40GB 上执行完整 sweep。

默认候选为 Qwen3.5-9B。如果 pinned runtime 无法验证 attention KV 与 Gated DeltaNet recurrent state 的完整 CPU restore，则不得把该路径视为 full hierarchy。此时按 `00-common-conventions.md` 切换到已验证 primary model，并把 Qwen3.5 标记为 `partial` 或 `unsupported`。

本实验只使用 text-only requests。

## 3. 对照配置

### GPU-only

Reusable cache/state 可以保留在 GPU 中。状态被 GPU cache 淘汰后不再可用，后续再次访问对应 prefix 时需要重新计算缺失部分。

### GPU + CPU hierarchical cache

GPU 使用与 GPU-only 完全相同的 cache budget。被 GPU 淘汰的 reusable state 可以进入经过验证的 CPU tier，并在后续命中时恢复到 GPU。

CPU tier 容量保持足够，使 CPU eviction 不成为本实验中的第二个容量变量。

## 4. Cache initial state

### Cold-cache

每轮正式 workload 开始前清空待测 reusable cache/state。

Cold-cache 用于观察 hierarchy 是否能在系统运行过程中逐步积累可复用状态，并在后续 revisit 中减少 recomputation。

### Warm-cache

正式计时前执行固定的 cache-population trace。预热阶段不进入正式性能统计。

GPU-only 与 hierarchical 使用相同的预热请求和访问顺序。正式测量开始前需要记录实际 GPU/CPU residency 和 cache occupancy，不能仅根据执行过 warm-up 就假定 warm state 建立成功。

Warm-cache 用于模拟已经积累历史 context/state 的长期 serving 状态。

## 5. 实验矩阵

| Cache architecture | Initial state |
|---|---|
| GPU-only | Cold |
| GPU-only | Warm |
| Hierarchical | Cold |
| Hierarchical | Warm |

除 cache architecture 与 initial state 外，其余条件保持一致。

## 6. Workload 设计

实验使用由多个 shared-prefix groups 组成的固定请求 trace。

同一 group 内的请求共享固定长度的 prefix，并具有不同的 suffix/query。不同 group 之间相互独立，避免所有请求集中到单个热点 prefix。

本实验选择一个代表性的中等 reuse 和中等偏高 reusable-cache pressure，使系统能够稳定产生 GPU eviction 和后续 revisit，同时避免进入严重 thrashing。

该代表性配置在正式实验前通过 calibration 确认，并在所有四个配置中保持不变。

Request count、request ordering、input/output token distribution、offered load 和 scheduler policy 全部固定。

## 7. Validity conditions

每个正式 run 必须满足以下条件：

- full-hierarchy restore path 已通过数值与 residency 验证；
- GPU-only 与 hierarchical 的 GPU cache budget 完全一致；
- CPU tier 不发生未控制的 capacity eviction；
- 固定 serving load 下不发生 active-request preemption；
- cold/warm 初始状态能够被观测并重复建立；
- CPU restore failure 不得静默退化为 recomputation。

任何不满足条件的 run 均保留 raw result，但标记为 invalid、partial 或 unsupported，不进入主结果汇总。

## 8. Cold-cache 实验流程

每轮首先恢复空 cache 状态，然后执行完整固定 trace。

运行初期两种架构都需要建立状态。随着请求继续到达，GPU cache 开始产生容量竞争。

GPU-only 中，被驱逐的 reusable state 在后续 revisit 时需要重新计算。Hierarchical 配置中，被驱逐状态如果成功保留在 CPU，则通过 restore 避免对应 recomputation。

Cold-cache 结果同时报告全程 aggregate 指标和按请求序列位置划分的阶段性指标，避免初始 population 阶段掩盖后续 hierarchy 收益。

## 9. Warm-cache 实验流程

每轮先执行固定 cache-population trace，并验证 cache occupancy/residency。

预热完成后立即执行正式 workload。两种架构经历相同访问历史，但由于 cache architecture 不同，正式阶段开始时能够保留的 reusable working set 可以不同。

Warm-cache 重点判断 CPU tier 是否扩大了长期可复用 working set，以及这种额外复用是否产生稳定端到端收益。

## 10. 核心观测指标

### Cache behavior

记录：

- GPU cache hit volume；
- GPU eviction volume；
- CPU cache hit volume；
- full-hit / partial-hit status by state group when available。

优先使用 token/state volume，而不是只报告 request-level hit count。

### Computation

记录 GPU miss 导致的 recomputation，并比较 hierarchical 相对 GPU-only 避免的 recomputation。

### Data movement

记录 CPU-GPU restore volume、transfer activity，以及能够测量时的 non-overlapped restore stall。

Raw transfer duration 不与 computation time 直接相加。

### Serving performance

记录：

- TTFT distribution；
- steady-state throughput；
- active-request preemption count；
- achieved request rate。

## 11. 实验执行方式

每个配置进行多次独立重复测量。

每轮实验重新建立规定的 cold/warm 初始状态。GPU-only 与 hierarchical 使用相同 workload trace。

两种 architecture 的执行顺序交替或随机化，避免机器长期状态变化系统性偏向某一配置。

正式测量前完成模型加载、首次分配和必要 runtime initialization。每次 run 保存 model revision、runtime commit、hardware、precision、cache/offload policy、GPU/CPU cache budget、workload identifier、initial state、repetition index 和 validity status。

## 12. 结果组织

实验至少形成：

1. cold-cache 下 GPU-only vs hierarchical 的 GPU/CPU hit、eviction 与 recomputation；
2. warm-cache 下相同对比；
3. CPU-GPU restore traffic 与 non-overlapped stall；
4. cold-cache TTFT / throughput；
5. warm-cache TTFT / throughput；
6. cold-cache 运行过程中 reuse benefit 随请求进程的变化。

结果分析必须保持以下证据链：

```text
GPU eviction of reusable state
        ↓
validated CPU-tier hit
        ↓
avoided recomputation
        ↓
restore traffic / stall
        ↓
TTFT / throughput change
```

## 13. 结果判断逻辑

### 情况 A：Hierarchy 产生明确正收益

CPU hit 稳定存在，recomputation 明显下降，并最终改善 TTFT 或 throughput。该结果说明当前 workload 下存在 hierarchical caching 的基础系统价值。

### 情况 B：Reuse 存在，但 restore cost 抵消收益

CPU hit 和 recomputation reduction 明显存在，但端到端性能没有改善。该结果说明额外容量有复用价值，但当前 data movement cost 抵消了计算节省。

### 情况 C：CPU tier 基本没有被利用

GPU eviction 后很少出现 CPU hit，hierarchical 与 GPU-only 的 recomputation 和性能接近。该结果说明当前代表性 workload 缺乏足够的可复用 working set，不能据此归因于 restore 太慢。

### 情况 D：Cold 与 Warm 结论不同

Warm-cache 明显受益而 cold-cache 收益有限，说明 hierarchy 的价值依赖历史状态积累和长期 revisit。若 warm-cache 仍无收益，则 Experiments 2 和 3 分别定位是 capacity pressure 不足、reuse 不足还是 transfer/stall 过高。

## 14. 实验边界

本实验只系统比较 cache architecture 和 initial state。

GPU cache pressure 不做 sweep。Prefix reuse 不做 sweep。Cache locality、request ordering、scheduler strategy 和 hardware 均保持固定。
