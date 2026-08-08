# Experiment 1: Locality × Arrival Rate Baseline Profiling

## 1. 实验目标

本实验建立现代 hybrid LLM serving workload 下的 baseline scheduler 性能画像。

实验独立控制 request arrival rate 与 cache distance / reuse distance，观察不同 workload 条件下 delay hit、redundant prefill、host-loading pressure、queueing、I/O stall、TTFT 与 throughput 的变化。

本实验只使用 baseline scheduler，不引入 delay-hit mitigation、balanced batching、bubble filling / stall hiding 或其他 scheduler optimization。

本实验主要回答：

1. Strata 所关注的 scheduler pathology 在现代 serving workload 中是否仍然存在；
2. delay-hit pressure 与 host-loading pressure 分别集中在哪些 cache-distance × load 区域；
3. 哪些 representative workloads 值得进入 Experiment 2 做机制消融。

所有配置遵循 [`00-common-conventions.md`](00-common-conventions.md)。

## 2. 实验变量

### 2.1 Request arrival rate

Arrival rate 划分为四个相对负载等级。

| Load level | 定义 |
|---|---|
| Low | 系统明显未饱和，请求基本不形成持续排队 |
| Medium | 系统存在稳定并发，但仍保留处理余量 |
| High | 系统接近稳定吞吐上限，但尚未形成持续 overload |
| Overload | offered load 超过稳定处理能力并形成持续 backlog |

四个等级不跨模型、跨硬件复用同一绝对 request rate。正式实验前对当前模型、硬件和固定 serving configuration 单独做 capacity calibration。

Overload 主要用于标定 saturation boundary，不作为 scheduler 优化价值的核心证据。

### 2.2 Cache distance

实验设置三种基础 request ordering。

| Condition | 请求访问关系 | 解释 |
|---|---|---|
| Min distance | 同一 context / prefix group 尽可能连续 | cache distance 最小，locality 最高 |
| Shuffle | 同一 request set 按固定 seed 随机混合 | 普通无序 workload |
| Max distance | 同一 context 的 revisit 尽可能均匀分散 | cache distance 最大，locality 最低 |

三种 workload 使用完全相同的 request set。Context 内容、请求数量、prefix/context length distribution、output length distribution 和 theoretical reuse opportunity 保持一致。

每条 trace 同时保存实际 reuse-distance distribution，最终分析不只依赖 Min/Shuffle/Max 标签。

## 3. 关键因果假设

本实验不假设“locality 越差，所有 scheduler pathology 都越严重”。

Cache distance 会改变至少两类不同现象：

1. **Delay-hit pressure**。短 cache distance 让同一 context 的请求更容易在首次 miss 尚未 resolve 时聚集，因此可能增加 delay hit 和 redundant prefill。
2. **Host-loading pressure**。长 cache distance 增加 reusable state 在再次访问前离开 GPU-ready 状态的机会。当固定 hierarchy 能从 CPU tier 恢复这些状态时，host restore 和 I/O stall 可能增加。

Strata Figure 11 的历史结果正是这两个方向分离。Minimum cache distance 下 delay-hit mitigation 收益最明显；maximum cache distance 下 delay-hit 减弱，而 CPU DRAM cache loading 相关优化更重要。

本实验只把该结果作为 reference hypothesis。现代模型/runtime 可以得到不同结果。

## 4. Workload 构造

实验使用多个 context/shared-prefix groups 构成固定 request set。

每个 group 包含多个共享 reusable context 的请求，同时保留不同 suffix/query 和 output generation。不同 group 之间相互独立。

请求集合在实验开始前冻结，然后生成 Min distance、Shuffle、Max distance 三种 ordering。

每条 trace 保存：

- request identifier；
- context/prefix group；
- arrival timestamp；
- reuse distance；
- input/output token length；
- theoretical reusable volume；
- seed/configuration hash。

## 5. Cache / I/O 底座

为了让 host-loading metrics 有明确含义，本实验使用一个在实验开始前冻结的 cache hierarchy 与 I/O backend。

如果目标模型的 full CPU-resident state restore 已通过 validation gate，则主结果同时研究 delay hit 与 loading-related behavior。

如果只能验证 partial hierarchy，则 host-loading、I/O-stall 与 balanced-batching 相关结果必须标记为 `partial`，不能解释为完整 hybrid-state behavior。

如果当前模型/runtime 完全无法建立可靠 CPU hierarchy，本实验仍可以执行 cold-miss delay-hit profiling，但不对 host-loading operating region 做正式结论。此时 Experiment 2 的 loading-related workload 应切换到已验证的模型/runtime，或标记为 `unsupported`。

## 6. 实验矩阵

主实验采用三种 cache-distance condition 与四种 arrival-rate level 的完整交叉设计。

| Cache distance \ Load | Low | Medium | High | Overload |
|---|---:|---:|---:|---:|
| Min distance | ✓ | ✓ | ✓ | ✓ |
| Shuffle | ✓ | ✓ | ✓ | ✓ |
| Max distance | ✓ | ✓ | ✓ | ✓ |

共 12 个 workload conditions。每个 condition 进行多次独立重复测量。

## 7. 实验前 calibration

首先固定模型、硬件、cache hierarchy、cache capacity、I/O backend 与 baseline scheduler。

在代表性 request set 上逐步提高 offered request rate，记录 achieved throughput、queueing delay 和 backlog status。

根据系统从明显未饱和、稳定并发、接近稳定吞吐上限到持续 backlog 的变化确定 Low、Medium、High 与 Overload。

Calibration 只用于冻结 load points，不进入 locality/cache-distance 主结论。

## 8. 正式实验流程

每轮实验恢复一致的初始 serving 状态，并完成不计入正式统计的一次性 runtime initialization。

系统按冻结的 request trace 和目标 arrival-rate condition 发送请求。

12 个条件中保持以下内容一致：

- model revision；
- hardware；
- precision/cache dtype；
- cache hierarchy 与 capacity；
- I/O backend；
- baseline scheduler；
- request set；
- token-length distribution；
- theoretical reuse opportunity。

不同 condition 的运行顺序交替或随机化。每个 run 保存 calibration identifier、trace identifier、offered/achieved request rate、actual reuse-distance summary、repetition index 与 validity status。

## 9. 核心指标

### 9.1 Delay-hit / reuse behavior

记录：

- delay-hit count / affected request volume；
- redundant prefill / recomputation；
- realized cache reuse；
- same-context overlap when observable；
- cache resolve time when observable。

### 9.2 Host-loading behavior

在 full/partial hierarchy 状态允许时记录：

- GPU-resident hit；
- CPU-resident hit；
- host restore volume；
- duplicate restore activity；
- non-overlapped I/O stall。

### 9.3 User-visible performance

记录：

- queueing delay distribution；
- P50/P90/P99 TTFT；
- achieved throughput；
- TPOT 或等价 decode metric when available。

Offered load 与 achieved load 同时保存，避免在接近 saturation 时误读 throughput。

## 10. 分析一：固定 arrival rate 比较 cache distance

在相同 load level 下比较 Min distance、Shuffle 与 Max distance。

分析不预设所有指标具有同一方向。重点分别判断：

- Min distance 是否形成更高 same-context overlap、delay hit 和 redundant work；
- cache distance 增大后 delay hit 是否下降；
- cache distance 增大后 CPU-resident hit / host restore / I/O stall 是否上升；
- 这些 scheduler-level变化是否传导到 TTFT 和 throughput。

如果实际结果与 Strata reference direction 不一致，则以当前 measurement 为准，并进一步检查现代 cache policy、hybrid-state layout 或 runtime coordination 是否改变了机制。

## 11. 分析二：固定 cache distance 提高 arrival rate

在同一 ordering 下比较 Low、Medium、High 与 Overload。

重点判断 arrival pressure 是否放大已有 pathology。

对于 Min distance，重点观察更高 request rate 是否增加首次 miss resolve window 内的 same-context overlap 与 delay hit。

对于 Shuffle / Max distance，重点观察更高 load 是否把 host restore 与 batch imbalance 放大为 non-overlapped I/O stall。

Overload 单独解释。持续 backlog 导致的 queueing 不能直接归因于 scheduler pathology。

## 12. 分析三：cache distance × load interaction

本实验不把“低 locality + 高 load”预设为唯一最差区域。

需要分别形成至少两张 mechanism-specific surface：

1. `cache distance × load → delay hit / redundant work`；
2. `cache distance × load → host restore / I/O stall`。

然后再检查两类 pathology 如何影响 TTFT / throughput。

这种设计允许出现：

- Min distance + High load 为 delay-hit-dominated；
- Max distance + High load 为 loading-dominated；
- Shuffle + High load 同时存在多种压力；
- 某些现代 runtime 下三者差异都很弱。

## 13. Representative workload selection

Experiment 2 不重复全部 12 个 conditions。

Experiment 1 完成后，按照预先固定的规则选择少量 representative workloads：

- **W0 Control**：稳定区域中 pathology 很弱的 control point；
- **W1 Delay-hit-sensitive**：delay hit / redundant prefill 明显，优先来自短 cache distance 和较高但稳定的 load；
- **W2 Loading-balance-sensitive**：host loading 明显、batch load/compute imbalance 可观察且 delay hit 不占主导；
- **W3 Residual-stall-sensitive**：存在明显 residual I/O stall / GPU bubble 的 workload。

如果 Experiment 1 无法找到满足某一角色的 workload，则不人为制造该角色，直接记录 `not observed`，对应 Experiment 2 component 的适用性结论相应收缩。

Selection rule 在运行任何 optimized scheduler 前冻结，并保存对应 baseline run identifiers。

## 14. 结果输出

Experiment 1 至少形成：

1. 三种 ordering 的实际 reuse-distance distribution；
2. cache distance × arrival rate 的 delay-hit / redundant-work surface；
3. cache distance × arrival rate 的 host-restore / I/O-stall surface，在 hierarchy capability 允许时；
4. queueing-delay summary；
5. P50/P90/P99 TTFT summary；
6. offered vs achieved throughput summary；
7. representative workload selection table。

## 15. 结果判断逻辑

### 情况 A：短 distance 下 delay hit 明显

Min distance 在 Medium/High stable load 下产生更多 delay hit 与 redundant work，并传导到 TTFT 或 throughput。

该结果说明现代 runtime 中仍存在 Strata-style delay-hit optimization space。

### 情况 B：长 distance 主要暴露 host loading

Shuffle/Max distance 的 delay hit 较弱，但 CPU-resident restore 与 non-overlapped I/O stall 增加。

该结果说明 scheduler value 更可能来自 balanced batching / stall hiding，而不是 delay-hit mitigation。

### 情况 C：内部 pathology 存在但端到端影响有限

Delay hit 或 host loading 指标变化明显，但 TTFT / throughput 基本稳定。

该结果说明当前系统余量或 overlap 可以吸收该成本，不能据内部指标直接声称 scheduler 具有显著 serving benefit。

### 情况 D：只有 Overload 出现明显恶化

稳定区域内差异较弱，只有持续 backlog 后出现明显 queueing / TTFT 恶化。

该结果不足以证明正常 serving 区域需要额外 scheduler optimization。

### 情况 E：cache distance 影响整体很弱

三种 ordering 在稳定 load 下的 delay hit、loading stall 与端到端性能都接近。

该结果说明 Strata 的 cache-distance-sensitive scheduler bottleneck 在当前 model/runtime/workload 中已明显弱化。

## 16. 实验边界

本实验只定位 baseline scheduler 的 pathology，不进行 scheduler mechanism attribution。

Delay-hit mitigation、balanced batching 与 stall hiding 的因果贡献由 Experiment 2 处理。Same-context burst 的专门压力测试由 Experiment 3 处理。最终 operating region 由 Experiment 4 汇总。
