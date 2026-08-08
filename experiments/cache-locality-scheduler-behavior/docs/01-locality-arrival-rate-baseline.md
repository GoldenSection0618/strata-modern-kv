# Experiment 1: Locality × Arrival Rate Baseline Profiling

## 1. 实验目标

本实验建立现代 hybrid LLM serving workload 下的 scheduler 基础性能画像。

实验独立控制 request arrival rate 与 cache locality / reuse distance，观察不同 workload 条件下 cache reuse、delay hit、redundant prefill、queueing、I/O stall、TTFT 与 throughput 的变化。

本实验只使用 baseline scheduler，不引入 delay-hit mitigation、balanced batching、bubble filling / stall hiding 或其他 scheduler optimization。

本实验主要回答两个问题：

1. Strata 所关注的 scheduler pathology 在现代 serving workload 中是否仍然存在；
2. 这些问题主要出现在哪些 locality 与 load 组合下。

实验结果用于选择 Experiment 2 的 representative workloads，并作为后续 scheduler ablation 的统一 baseline。

所有配置遵循 [`00-common-conventions.md`](00-common-conventions.md)。

## 2. 实验变量

本实验设置两个独立自变量。

### 2.1 Request arrival rate

Request arrival rate 划分为四个负载等级。

| Load level | 定义 |
|---|---|
| Low | 系统明显未饱和，请求基本不形成持续排队 |
| Medium | 系统存在稳定并发，但仍保留明显处理余量 |
| High | 系统接近稳定吞吐上限，queueing 与资源竞争开始明显 |
| Overload | offered load 略高于稳定处理能力，系统形成持续排队 |

四个等级不使用跨模型、跨硬件统一的绝对 request rate。

正式实验前先对当前模型、硬件和 serving configuration 进行 capacity calibration，并根据稳定 throughput 与 queueing behavior 确定四个负载点。

Overload 条件用于识别 saturation 对指标的影响，并帮助区分 scheduler pathology 与纯容量不足。Overload 不作为判断 scheduler 优化价值的唯一依据。

### 2.2 Cache locality

实验设置三种基础 locality workload。

| Locality condition | 请求访问关系 | 作用 |
|---|---|---|
| Min distance | 相同或相关 context 尽可能连续出现 | 构造高 locality |
| Shuffle | 同一请求集合按固定随机顺序混合 | 构造中等 locality |
| Max distance | 相同 context 的 revisit 尽可能分散 | 构造低 locality |

三种 workload 使用完全相同的请求集合，只改变 request ordering 与 reuse-distance structure。

Context 内容、请求数量、context length distribution、output length distribution 和理论 reuse opportunity 在三种 locality 条件下保持一致。

这种设计使 locality 变化与 workload 内容变化分离，并避免把“没有可复用内容”误判为“scheduler 没有及时利用已有复用机会”。

## 3. Workload 构造

实验使用由多个 context / shared-prefix groups 组成的固定请求集合。

每个 group 包含多个访问同一 context 或共享同一 reusable prefix 的请求，不同 group 之间保持独立。

请求集合在实验开始前冻结，并生成三种 locality ordering。

Min-distance ordering 将同组请求尽可能集中排列。Shuffle ordering 使用固定 seed 对同一请求集合进行随机化。Max-distance ordering 将同组 revisit 尽可能拉开，同时保持请求集合本身不变。

所有 ordering 均保存稳定 trace identifier，并记录实际 reuse-distance distribution。实验结果同时报告设定的 locality condition 与实际观察到的 reuse distance，避免只依赖 workload 标签进行解释。

## 4. 实验矩阵

主实验采用三种 locality 与四种 arrival-rate level 的完整交叉设计。

| Locality \ Load | Low | Medium | High | Overload |
|---|---:|---:|---:|---:|
| Min distance | ✓ | ✓ | ✓ | ✓ |
| Shuffle | ✓ | ✓ | ✓ | ✓ |
| Max distance | ✓ | ✓ | ✓ | ✓ |

主实验共包含 12 个 workload conditions。

每个 condition 执行多次独立重复测量。正式比较使用重复运行的统计结果，不使用单次 measurement 作为主要结论依据。

## 5. 实验前 calibration

实验首先固定模型、硬件、cache hierarchy、cache capacity、I/O backend 与 baseline scheduler。

系统在代表性 workload 上从低 offered load 开始逐步提高 request arrival rate，并记录 achieved throughput、queueing delay 和系统是否形成持续 backlog。

Calibration 根据系统从明显未饱和、正常并发、接近稳定吞吐上限到持续排队的变化确定 Low、Medium、High 与 Overload 四个 load point。

Calibration 只用于确定实验工作区间，不进入 locality 主结论。

如果不同模型或硬件的稳定处理能力明显不同，则分别完成 calibration，并保持相同的相对负载定义。

## 6. 正式实验流程

每轮实验首先恢复一致的初始 serving 状态，并完成不计入正式统计的一次性 runtime initialization。

实验加载预先冻结的 request trace，并按照目标 locality ordering 与 arrival-rate condition 发送请求。

模型、硬件、cache hierarchy、cache capacity、I/O backend、baseline scheduler、请求集合和 token-length distribution 在 12 个 workload conditions 中保持不变。

每个 run 覆盖足够长的稳定 serving 区间。初始化阶段与结束阶段不进入主性能聚合。

不同 workload condition 的执行顺序交替或随机化，避免机器温度、长期负载和集群状态变化系统性偏向某个 condition。

每次 run 保存 workload identifier、locality condition、reuse-distance summary、offered/achieved request rate、repetition index 和 validity status。

## 7. 核心观测指标

### 7.1 Cache / scheduler behavior

实验记录 realized cache hit / reuse、delay hit 与 redundant prefill。

这组指标用于判断已有 reuse opportunity 是否被 baseline scheduler 正确利用，以及 locality 恶化后是否产生更多重复计算或延迟命中。

### 7.2 Waiting and stall behavior

实验记录 queueing delay 与 I/O stall。

这组指标用于区分请求等待、cache/state movement 和其他 serving stall 对 TTFT 的影响。

### 7.3 User-visible performance

实验记录 TTFT distribution 与 achieved throughput。

TTFT 至少保留中位数与 tail latency，避免平均值掩盖高负载下少量请求的严重等待。

Throughput 使用实际完成请求或等价的稳定 serving 输出定义，并同时记录 offered 与 achieved request rate。

## 8. 分析一：固定 arrival rate 比较 locality

分析在相同 load level 下比较 Min distance、Shuffle 与 Max distance。

该分析检查 locality 变差后 realized cache reuse 是否下降、delay hit 是否增加、redundant prefill 是否增加，以及这些变化是否进一步传导到 queueing、I/O stall、TTFT 和 throughput。

如果 scheduler-level pathology 随 reuse distance 增大而稳定增强，并同步出现端到端性能恶化，则说明 request ordering 与 cache locality 仍然是现代 serving 系统中的有效 control-plane 变量。

如果 locality 变化明显改变 cache behavior，但端到端性能基本不变，则说明当前系统仍具有足够余量吸收这些开销，后续 scheduler optimization 的实际价值需要在更高 load 下判断。

## 9. 分析二：固定 locality 提高 arrival rate

分析在同一 locality condition 下比较 Low、Medium、High 与 Overload。

该分析检查低负载下可以被系统余量吸收的 scheduler pathology 是否随着系统接近饱和而被放大。

重点观察 delay hit、redundant prefill、queueing delay、I/O stall 与 TTFT 是否在 High load 附近出现明显变化，以及 achieved throughput 是否开始偏离 offered load。

Overload 结果单独解释。Overload 中持续 backlog 本身会显著增加 queueing，因此不能把所有性能下降直接归因于 locality 或 scheduler pathology。

## 10. 分析三：检查 locality × load 交互作用

分析重点比较以下四类 workload region：

- high locality + low load；
- low locality + low load；
- high locality + high load；
- low locality + high load。

该分析判断 locality 的影响是否随着系统负载提高而放大。

如果低 locality 在 Low load 下影响有限，但在 High load 下显著增加 delay hit、redundant prefill 或 queueing，并最终恶化 TTFT / throughput，则说明 scheduler optimization 的价值集中在特定 locality × load regime，而不是所有 workload。

如果 locality 与 load 基本呈独立影响，则后续 Experiment 2 分别选择 locality-sensitive 与 load-sensitive representative points，而不强行解释为交互效应。

## 11. 控制条件

除 locality 与 arrival rate 外，其余主要条件保持固定。

固定条件包括：

- model identifier 与 revision；
- hardware platform；
- precision 与 cache dtype；
- cache hierarchy；
- GPU/CPU cache capacity；
- cache/page policy；
- I/O backend；
- baseline scheduler implementation；
- request set；
- context/input length distribution；
- output length distribution；
- theoretical reuse opportunity。

本实验不同时 sweep cache capacity、page granularity、GPU-assisted I/O 或 hierarchical-cache policy。这些变量由其他实验组独立研究。

## 12. 结果组织

Experiment 1 至少形成以下结果：

1. locality × arrival rate 的 delay-hit surface；
2. locality × arrival rate 的 redundant-prefill surface；
3. locality × arrival rate 的 queueing-delay / I/O-stall summary；
4. locality × arrival rate 的 TTFT summary；
5. locality × arrival rate 的 achieved-throughput summary；
6. actual reuse-distance distribution 与设定 locality condition 的一致性检查；
7. representative workload selection table，供 Experiment 2 使用。

结果至少保留一组 scheduler-level pathology 图和一组 user-visible performance 图，使后续机制消融能够回到同一 baseline surface 上解释。

## 13. Representative workload selection

Experiment 2 不重复全部 12 个 workload conditions。

Experiment 1 完成后，根据 baseline pathology 与端到端性能结果选择少量 representative workloads。

Representative points 至少覆盖一个 scheduler pathology 很弱的 control point、一个 locality-sensitive point 和一个在高负载下 pathology 明显的 stress point。

Representative-point selection rule 在运行 Experiment 2 前冻结，并记录对应 Experiment 1 run identifiers。后续不能根据 scheduler optimization 的最终收益反向挑选 baseline workload。

## 14. 结果判断逻辑

### 情况 A：Low locality 在高负载下显著恶化

Delay hit、redundant prefill 或 queueing 随 reuse distance 增大而上升，并在 High load 下明显传导到 TTFT / throughput。

该结果说明现代 workload 中仍存在明确的 scheduler optimization space，Experiment 2 继续验证具体机制贡献。

### 情况 B：Cache behavior 变化明显，但端到端影响有限

Locality 改变了 hit、delay hit 或 redundant prefill，但 TTFT / throughput 基本稳定。

该结果说明 scheduler pathology 存在，但当前系统资源余量或 overlap 能够吸收其成本。后续消融重点检查更高负载或 stall-sensitive workload，而不能直接声称 scheduler 优化具有明显系统收益。

### 情况 C：Locality 对 scheduler-level 与 end-to-end 指标都影响很弱

三种 locality ordering 在多数稳定负载区间下表现接近。

该结果说明 Strata 所强调的 locality-sensitive scheduler problem 在当前模型/runtime/workload 中已经弱化。后续 scheduler 实验缩小范围，并把“收益边界已经收缩”作为有效结论。

### 情况 D：只有 Overload 出现明显差异

Low、Medium 与 High 条件基本稳定，明显恶化只出现在持续 backlog 的 Overload 条件。

该结果不足以证明 locality-aware scheduler 在正常 serving 区域具有价值。Overload 结果主要用于确定系统容量边界，不作为 scheduler 优化必要性的核心证据。

## 15. 实验边界

本实验只定位 baseline scheduler 在 locality × load space 中的 pathology。

本实验不比较不同 scheduler mechanism，不研究 scheduler optimization 的相对收益，也不对 delay-hit mitigation、balanced batching 或 stall hiding 做因果归因。

这些问题由 Experiment 2 及后续实验处理。