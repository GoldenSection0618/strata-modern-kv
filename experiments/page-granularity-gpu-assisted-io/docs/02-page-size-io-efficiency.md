# Experiment 2: Page Size vs. I/O Efficiency

## 1. Objective

本实验定量评估 page granularity 对 CPU→GPU cache/state transfer efficiency 的影响，验证较小 page 是否会因为传输碎片化导致有效带宽下降，并确定这种 I/O degradation 从什么 page-size 区间开始显著。

本实验只研究 `page size → I/O efficiency` 这一段因果关系，不引入 GPU-assisted I/O。GPU-assisted I/O 留到 Experiment 3 单独验证。

本实验最终需要回答：实验一得到的 reuse-efficient page-size region 是否已经开始付出明显的 I/O penalty，以及 page size 从什么范围开始进入 fragmentation-dominated region。

## 2. Research questions

本实验回答以下问题：

1. page size 减小时，单次 transfer 是否变得更加碎片化；
2. fragmented transfer 是否导致 sustained host→GPU bandwidth 下降；
3. bandwidth degradation 是否进一步增加 cache restore / load 的等待时间；
4. 不同 transfer volume 和 workload 下，该趋势是否保持一致；
5. Experiment 1 得到的 reuse-efficient page-size region 中，是否已经出现明显的 I/O penalty；
6. page size 从什么范围开始进入明显的 fragmentation-dominated region。

## 3. Independent variable

实验的主要自变量为 page size。

Page size 使用与 Experiment 1 相同的档位，使两组实验结果能够直接对应。所有正式结果都应保留相同的 page-size identifier，便于后续联合分析。

本实验统一使用普通 CPU→GPU I/O backend，不启用 GPU-assisted/kernel I/O。

## 4. Controlled variables

同一组 page-size sweep 中保持以下条件不变：

- model 与 model revision；
- hardware；
- serving runtime 与 runtime configuration；
- precision；
- CPU memory configuration；
- GPU memory configuration；
- I/O backend；
- scheduler policy；
- cache policy；
- transfer direction；
- request sequence；
- repetition protocol。

除明确作为实验变量的 transfer volume、access pattern 和 page size 外，不允许同时改变其他可能影响 I/O efficiency 的因素。

## 5. Experimental structure

Experiment 2 分为两个层次。

第一层采用 Controlled I/O experiment，固定逻辑总传输数据量，只改变 page granularity，用于建立 page fragmentation 与 bandwidth efficiency 之间的直接因果关系。

第二层采用 Serving-level validation，在实际 cache reuse workload 中验证 Controlled I/O experiment 观察到的 fragmentation effect 是否足以影响 serving critical path。

两层实验分别回答机制是否存在，以及该机制是否具有实际系统影响。

## 6. Experiment 2A: Controlled I/O Efficiency

### 6.1 Purpose

Controlled I/O experiment 固定总传输数据量，只改变数据被划分成多少个 page。

在同一比较组中，总数据量、transfer direction、hardware 和 I/O backend 保持一致。page size 是主要变化因素，因此 observed bandwidth difference 可以主要归因于 transfer granularity 与 fragmentation。

### 6.2 Transfer volume

实验设置多个代表性的总 transfer volume，至少覆盖 small、medium 和 large transfer 三个区间。

每个 transfer volume 都执行完整的 page-size sweep。

该设计用于判断 fragmentation penalty 是在不同 transfer scale 下普遍存在，还是只在大规模 cache restoration 时显著。

具体 transfer-volume 档位在实现阶段根据模型实际 cache/state footprint 与 runtime 可支持范围确定，并在所有 page-size 配置中保持逻辑总传输字节数一致。

### 6.3 Access pattern

Controlled I/O experiment 至少包含两类 access pattern。

#### A. Contiguous transfer

需要传输的 cache/state 在逻辑上连续。

该组作为相对理想的传输基线，用于测量在访问连续的情况下，page granularity 本身对 I/O efficiency 的影响。

#### B. Fragmented page selection

需要传输的数据由多个离散 page 组成，但同一比较组的逻辑总传输数据量保持一致。

该组用于模拟 cache reuse 场景中只恢复部分有效 page 的实际访问形态，并观察小 page 与离散访问叠加后的额外 fragmentation cost。

两类 access pattern 的结果分开报告，不将 contiguous 与 fragmented selection 混合为单一 bandwidth 数值。

## 7. Experiment 2B: Serving-Level Validation

### 7.1 Purpose

Serving-level validation 在真实 cache reuse workload 中验证 Controlled I/O experiment 发现的 fragmentation effect 是否足以影响 serving。

该阶段保留 page size 对实际 cache behavior、transfer volume 和 restore pattern 的自然影响，因此不要求不同 page-size 配置具有完全相同的实际传输数据量。

该阶段回答的是：在实际系统中，小 page 带来的 cache reuse 收益是否同时伴随着可观测且具有实际影响的 I/O penalty。

### 7.2 Workload selection

Serving-level validation 不重复 Experiment 1 的全部 workload sweep。

从 Experiment 1 的结果中选择三个代表性 operating points。

#### Point A: Low reuse benefit

该点选择小 page 对 effective reuse 提升较小的 workload。

该点用于建立负收益边界。如果 page size 缩小已经明显降低 I/O efficiency，而 reuse benefit 很小，则该区域不具有实际系统价值。

#### Point B: Intermediate trade-off

该点选择 page size 缩小时 effective reuse 明显增加，但收益尚未完全饱和的 workload。

该点作为主要 trade-off operating point，用于观察 reuse benefit 与 I/O penalty 是否同时出现。

#### Point C: High reuse / saturation

该点选择小 page 已接近 Experiment 1 最大 reuse benefit 的 workload。

该点用于判断继续缩小 page 是否主要增加 I/O cost，而不再产生明显额外 reuse benefit。

三个 operating points 的选择必须直接依据 Experiment 1 的 processed results，不通过主观指定与结果无关的 workload。

## 8. Cache-state control

每次正式 serving measurement 前将 cache 初始化到规定状态。

同一 workload 的所有 page-size 配置使用相同的 request sequence、prefix structure、cache capacity 和 initial residency condition。

实验必须确认正式测量期间实际发生目标 CPU→GPU cache/state transfer。

没有产生目标 transfer 的 run 不进入 I/O efficiency 分析，也不能将“未发生 I/O”解释为较高的 I/O efficiency。

## 9. Core metrics

### 9.1 Sustained host→GPU bandwidth

Sustained host→GPU bandwidth 是 Experiment 2 的主要性能指标。

该指标使用正式测量窗口内实际 CPU→GPU 数据量与对应 transfer duration 计算，并分别报告 Controlled I/O experiment 与 Serving-level validation 的结果。

主要观察关系为：

```text
page size ↓ → sustained host→GPU bandwidth ?
```

### 9.2 Bandwidth utilization

Bandwidth utilization 将 observed sustained bandwidth 与同一 hardware、同一 backend 下的大块连续传输 reference bandwidth 进行比较。

```text
bandwidth utilization = observed sustained bandwidth / reference bandwidth
```

Reference bandwidth 的测量方法在整个实验中保持一致。

该指标用于比较不同 page granularity 对硬件可用传输能力的利用程度。

### 9.3 Transfer count

实验记录完成一次逻辑 cache/state restoration 所需要的实际 transfer/page operation 数量。

该指标用于验证 page size 缩小是否真实转化为更多 transfer operation，而不是仅在逻辑页面表示上发生变化。

### 9.4 Transfer-size distribution

实验记录实际 transfer size 的分布，而不是仅依据 configured page size 推断 I/O granularity。

该指标用于识别 runtime 是否自动执行 page merging、coalescing 或 batching。

如果 runtime 已经将多个小 page 聚合为较大的实际 transfer，则正式结论必须基于 observed transfer granularity，而不能仅依据 configured page size 宣称存在 fragmented I/O。

### 9.5 I/O duration

实验记录一次 cache/state restoration 的实际 I/O duration。

该指标用于验证 sustained bandwidth degradation 是否进一步转化为更长的 restore latency。

### 9.6 Non-overlapped I/O stall

Serving-level validation 进一步记录无法被 computation overlap 隐藏的 I/O stall。

该指标用于判断 bandwidth degradation 是否真正进入 serving critical path。

较低 bandwidth 如果完全能够被 computation overlap，则不能仅凭 microbenchmark bandwidth loss 声称 serving performance 已受到实际影响。

## 10. Execution procedure

Controlled I/O experiment 对每一个 transfer-volume 与 access-pattern 组合执行完整 page-size sweep。

每个配置先完成统一 warm-up，再进入正式 measurement window。每个配置重复运行多次，并记录 repetition index。

Controlled I/O 的基本实验单元定义为：

```text
fixed total logical bytes × fixed access pattern × page-size sweep
```

Serving-level validation 对每一个 representative operating point 使用与 Experiment 1 一致的 request sequence 和 workload parameters，再执行完整 page-size sweep。

每个 run 同时记录 cache behavior、actual transfer behavior 与 I/O stall，使 page size、reuse 和 I/O 的关系能够在同一次 serving execution 中对应起来。

不同 page-size 配置的运行顺序不应固定为单向递增或递减。正式实现应避免长期系统状态、温度或共享资源变化与 page-size 顺序系统性相关。

## 11. Result organization

Controlled I/O results 与 Serving-level results 分开保存和汇总。

每条 raw measurement 至少记录：

- experiment type；
- page size；
- logical transfer volume；
- actual transferred bytes；
- access pattern；
- transfer count；
- transfer-size statistics；
- transfer duration；
- sustained bandwidth；
- bandwidth utilization；
- workload identifier；
- cache-state validity；
- repetition index；
- runtime 与 hardware metadata。

Serving-level run 额外记录 Experiment 1 中对应的 reuse metrics 与 non-overlapped I/O stall。

## 12. Primary analyses

### Analysis A: Page size vs. sustained bandwidth

绘制 page size 与 sustained host→GPU bandwidth 的关系。

该分析用于判断小 page 是否导致 host→GPU bandwidth 系统性下降，以及下降从什么粒度开始明显出现。

### Analysis B: Page size vs. bandwidth utilization

绘制 page size 与 bandwidth utilization 的关系。

该分析用于判断不同 page granularity 距离同一硬件和 backend 的 reference bandwidth 有多远。

### Analysis C: Fragmentation mechanism

联合分析 page size、transfer count、transfer-size distribution 与 sustained bandwidth。

目标机制链为：

```text
page size ↓
    ↓
actual transfer fragmentation ↑
    ↓
bandwidth utilization ↓
```

如果 configured page size 减小但 actual transfer granularity 没有变小，则不能认为上述机制得到验证。

### Analysis D: Serving-level I/O stall

绘制 page size 与 non-overlapped I/O stall 的关系，并与 sustained bandwidth 变化对应。

该分析用于判断 Controlled I/O experiment 中观察到的 bandwidth degradation 是否进入真实 serving critical path。

## 13. Joint analysis with Experiment 1

Experiment 1 与 Experiment 2 使用相同 page-size axis 进行联合分析。

联合结果至少同时展示：

- effective reused tokens；
- cache utilization efficiency；
- sustained host→GPU bandwidth；
- bandwidth utilization；
- non-overlapped I/O stall。

联合分析的目标是识别 page granularity 的系统 trade-off：

```text
page size ↓
    ├── cache reuse ↑
    └── transfer fragmentation ↑ → I/O efficiency ↓ → I/O stall ↑
```

实验不以寻找单独最高的 cache hit rate 或最高的 bandwidth 为目标，而是寻找 reuse benefit 与 I/O penalty 开始发生明显冲突的区域。

## 14. Validity checks

正式分析前必须确认以下条件成立：

1. Controlled I/O comparison 中不同 page-size 配置的 logical transfer bytes 一致；
2. actual transferred bytes、transfer count 和 transfer-size distribution 能够被实际观测；
3. runtime 的 page merging、coalescing 或 batching 行为已经被识别；
4. bandwidth measurement 不包含明显无关的计算时间；
5. Serving-level validation 中 cache hit 实际触发目标 CPU→GPU restore；
6. non-overlapped I/O stall 与 raw transfer duration 使用不同口径，不直接互相替代；
7. invalid run 保留记录并标记 invalid reason，不进入正式聚合结果。

## 15. Interpretation boundary

本实验可以证明 page size 与实际 transfer fragmentation、bandwidth efficiency 和 serving I/O stall 之间的关联，并通过 Controlled I/O experiment 建立较强的机制证据。

本实验不能证明 GPU-assisted I/O 能够解决该问题，因为该变量尚未引入。

本实验也不能仅凭 configured page size 推断 fragmentation。正式结论必须依赖实际 transfer count 和 transfer-size distribution。

Serving-level bandwidth degradation 只有在同时观察到 non-overlapped I/O stall 增加时，才能进一步解释为实际 serving bottleneck 的增强。

## 16. Expected conclusion structure

Experiment 2 最终应明确回答：page size 减小时，实际 I/O 是否变得更加碎片化，fragmentation 是否造成 sustained bandwidth 与 bandwidth utilization 下降，以及这种下降是否足以进入 serving critical path。

最终应根据 observed behavior 将 page-size space 划分为大致三个区域：

- **I/O-efficient region**：page 较大，actual transfer granularity 较粗，bandwidth utilization 较高；
- **trade-off region**：Experiment 1 的 reuse benefit 与 Experiment 2 的 I/O penalty 同时明显；
- **fragmentation-dominated region**：继续缩小 page 已显著损害 I/O efficiency，并产生可观测 I/O stall。

Experiment 2 的结果与 Experiment 1 的 reuse-efficient region 叠加后，决定 Experiment 3 应重点测试哪些 page-size operating points。

如果 reuse-efficient region 与 I/O-efficient region 存在明显冲突，则 Experiment 3 的核心问题自然变为：GPU-assisted I/O 是否能够恢复这些小 page 配置的传输效率，从而缓解 page granularity 的 reuse-I/O trade-off。
