# Experiment 2: Short-context Serving Regression

## 1. 实验目标

本实验用于验证面向 long-context reuse 设计的 hierarchical cache、I/O optimization 和 scheduler optimization，在普通 short-context serving 场景中是否引入明显的性能回退。

本实验重点回答三个问题：

1. Full Configuration 在缺乏显著 long-context reuse 的情况下，是否增加请求处理开销；
2. 各项优化是否会降低 short-context workload 的 serving capacity，或者恶化 TTFT 与 request completion time；
3. 当系统负载逐渐提高时，优化机制是否会引入额外的 queueing、调度或资源竞争，从而放大 short-context 请求的 tail latency。

本实验不要求优化后的系统优于 Baseline。实验目标是建立一个严格的 regression / equivalence test，而不是寻找 short-context speedup。

所有配置遵循 [`00-common-conventions.md`](00-common-conventions.md)。

## 2. Workload 设计

实验使用以独立 short-context 请求为主的 text-only workload。

每个请求具有较短的 input context，不包含长共享 prefix。不同请求之间尽量避免大规模 prefix overlap，使 hierarchical cache 无法依赖明显的 context reuse 获得额外收益。

请求内容来自多个不同 prompt，而不是反复访问同一个 context。这样能够把实验重点放在优化机制本身的额外开销，而不是 cache hit 带来的计算节省。

实验覆盖多个 short-context input-length profiles，使结果不依赖某一个特定输入长度。

Output length 使用预先定义的 profiles，覆盖较短和相对较长的生成请求。不同 system configuration 使用相同 output targets，并记录 realized output length。

不同 system configuration 使用完全相同的请求集合、input/output length profile、request ordering 与 arrival schedule。

## 3. Reuse 条件

本实验的主 workload 保持较低的 reusable-prefix overlap。

如果 runtime 自身存在 unavoidable system prompt、模板或公共 token prefix，则这些公共部分在所有配置中保持完全一致，并记录实际 cache reuse。

实验不通过人为关闭 Baseline 已有的正常 prefix caching 来制造无复用环境，也不主动构造大量相同 prefix。

正式结果需要报告实际发生的 reusable-prefix overlap 与 cache reuse，确认 short-context workload 没有意外变成长共享 context workload。

如果实际 reuse 明显偏离预定义的 workload validity range，则该 run 保留 raw result，但不进入主要 regression aggregation。

## 4. 系统配置

实验与 Experiment 1 使用完全相同的五种配置：

1. **Baseline**；
2. **Hierarchical Cache**；
3. **Hierarchical Cache + I/O Optimization**；
4. **Hierarchical Cache + Scheduler Optimization**；
5. **Full Configuration**。

Baseline 表示当前项目确定的现代 serving baseline。

Hierarchical Cache 配置用于判断仅启用额外 cache hierarchy 是否对普通短请求产生固定成本。

Hierarchical Cache + I/O Optimization 在相同 hierarchy 上启用经过验证的 I/O path，并保持 reference/baseline scheduler，用于判断额外 I/O mechanism 是否对非目标 workload 产生影响。

Hierarchical Cache + Scheduler Optimization 在相同 hierarchy 上启用经过验证的 scheduler mechanisms，并保持 reference/baseline I/O path，用于检查 scheduler 在缺乏明显 cache-locality problem 时是否造成额外 overhead。

Full Configuration 同时启用经过验证的 hierarchy、I/O 和 scheduler mechanisms，用于判断完整系统最终是否能够保持 short-context serving 性能。

配置 3 与配置 4 是 parallel attribution branches，不按逐层累加关系解释。

## 5. Cache 初始状态

Short-context regression 的主要结果在统一的 clean initial state 下测量。

每轮实验开始前建立规定的 cache 状态，并确保不同 system configuration 的初始条件可比较。

由于本实验不依赖长期 context reuse，cold-cache 与 warm-cache 不作为独立实验维度进行完整 sweep。

正式测量前执行统一 runtime warm-up，以消除 kernel initialization、memory allocation 和模型首次执行的影响。Warm-up trace 与正式 short-context trace 分离，并验证 warm-up 没有无意建立大量可复用 prefix state。

正式 measurement window 内记录实际 cache occupancy 和 cache activity，用于确认 hierarchy 没有因为历史残留状态改变 workload 性质。

## 6. Load scaling

每类 short-context workload 在多个 offered-load 条件下执行。

低负载条件用于测量不同系统配置在资源竞争很弱时的 fixed overhead。

如果 Full Configuration 在低负载条件下已经稳定增加 TTFT 或 completion time，则说明优化机制本身存在直接 request-path cost。

中等负载条件用于观察 scheduler、cache metadata management 和 background data movement 是否开始与正常请求竞争资源。

高负载条件逐步接近系统饱和区域，用于判断不同配置的 serving capacity 是否发生变化，以及额外机制是否放大 P90/P99 latency。

正式 load grid 在 calibration 后冻结。不同系统配置使用相同 offered-load points 和 arrival schedule，不根据某个配置的实际处理速度重新选择负载。

## 7. 控制变量

严格 paired comparison 中保持以下条件一致：

- model identifier 与 revision；
- GPU / CPU resources；
- serving runtime version / commit；
- precision 与 cache dtype；
- request dataset；
- input-length profile；
- output targets / output-length profile；
- request ordering；
- arrival schedule；
- generation settings；
- GPU reusable-cache budget；
- batch / concurrency limits；
- measurement rule 与 measurement window。

所有启用 hierarchy 的配置保持相同 CPU-tier budget、host-memory policy 与 offload policy。

实验主要改变 system configuration、short-context profile 和 offered load。

如果某一配置必须改变 batch limit、GPU cache budget 或其他关键 serving 参数才能正常运行，则该结果不能进入主要 paired comparison。

## 8. 核心指标

### TTFT

记录 P50、P90 和 P99 TTFT。

P50 用于判断各项优化是否给普通请求增加 fixed latency。P90 和 P99 用于观察高负载条件下是否出现额外 queueing、scheduler interference 或其他 tail amplification。

### Throughput

记录正式 measurement window 内的 request throughput 和 token throughput。

两种 throughput 同时报告，避免 output work 的小幅差异使单一指标产生误导。

### Request Completion Time

记录请求从到达到生成结束的 completion-time distribution。

该指标用于检测优化机制是否没有明显影响 TTFT，却在后续 decode、batch formation 或排队阶段增加整体完成时间。

### GPU Utilization

记录正式 measurement window 内的 GPU utilization。

该指标主要用于辅助判断 throughput regression 是否与 GPU idle、额外 kernel activity 或资源竞争相关。

## 9. 辅助指标

为了定位可能的 regression 来源，实验同时记录：

- realized cache hit / reuse；
- CPU-tier activity；
- CPU-GPU data movement；
- scheduler queueing time；
- scheduling decision overhead when observable；
- GPU idle / stall；
- batch characteristics；
- achieved request/token rate。

这些指标只用于解释 regression，不作为本实验的主要评价目标。

例如，如果 Full Configuration 的 P99 TTFT 上升，同时 cache reuse 接近零且 scheduler queueing 增加，则更可能说明额外 scheduling path 对非目标 workload 产生了负面影响。

## 10. 实验执行流程

每个 system configuration 首先完成统一的模型加载和 runtime warm-up。

随后恢复规定的 clean initial state，并运行固定 short-context request trace。

每个 workload point 使用固定 request set、seed、output targets 和 arrival schedule。

正式 measurement window 覆盖足够多的请求，使 P50、P90 和 P99 latency 以及 throughput 具有稳定统计意义。

不同 system configuration 的运行顺序交替或随机化，避免机器温度、系统负载或其他时间相关因素系统性偏向某一配置。

每个 workload point 进行多次独立重复测量。

每个 run 保存 system configuration、workload identifier、input/output profile、offered request/token work、achieved request/token throughput、cache activity、repetition index 和 validity status。

## 11. Regression / equivalence 判定方式

本实验不根据单次平均值或视觉上“差不多”判断是否发生 regression。

在查看 Full Configuration 的正式 comparison result 之前，冻结：

- practical regression / equivalence margin；
- repetition policy；
- aggregation method；
- uncertainty / interval reporting method；
- invalid-run exclusion rule。

首先比较 Full Configuration 与 Baseline 的 request/token throughput、P50/P90/P99 TTFT 和 request completion time，同时报告绝对值与相对变化。

配置 2、3、4 用于定位 regression 来源，不替代 Full Configuration 相对 Baseline 的主要判定。

如果观测差异小于实验可分辨范围，或者 uncertainty 仍然覆盖具有实际意义的 regression 区间，则结果标记为 `inconclusive`，不能直接写成 `no regression`。

“没有统计显著差异”本身不能作为性能等价的充分证据。

低负载下主要关注 fixed overhead。中高负载下主要关注 serving capacity 和 tail latency。

## 12. 结果组织

结果首先按照 **short-context profile × offered load** 组织。

每个 workload point 横向比较五种 system configuration。

主结果至少形成：

1. P50 / P90 / P99 TTFT 随 offered load 的变化；
2. request throughput 随 offered load 的变化；
3. token throughput 随 offered load 的变化；
4. request completion time 随 offered load 的变化；
5. GPU utilization 随 offered load 的变化；
6. Full Configuration 相对 Baseline 的 regression / equivalence summary。

最终 regression summary 同时保存 absolute measurements、relative deltas、uncertainty 与结论状态。

结论状态至少区分：

- no material regression；
- material regression；
- throughput-latency trade-off；
- inconclusive。

## 13. 结果判断逻辑

### 情况 A：Full Configuration 与 Baseline 在预定义 margin 内等价

如果不同负载下 throughput 和 latency 均满足预定义 equivalence rule，则说明针对 long-context reuse 的优化没有对普通 short-context serving 造成具有实际意义的 regression。

### 情况 B：低负载存在稳定 latency overhead

如果低负载时 Full Configuration 的 TTFT 或 completion time 已经稳定高于 Baseline，而 queueing 几乎不存在，则说明额外机制在普通 request path 中引入了 fixed cost。

此时根据配置 2、3、4 的 parallel attribution comparison 判断 overhead 主要来自 hierarchy、I/O path 还是 scheduler。

### 情况 C：低负载正常，高负载出现 regression

如果低负载结果接近，但中高负载下 throughput 降低或 P99 TTFT 明显增加，则说明额外机制本身的 fixed cost 较小，但在资源竞争条件下会降低 serving efficiency。

此时结合 scheduler queueing、GPU utilization、batch behavior 和 background activity 定位原因。

### 情况 D：某个中间配置退化，但 Full Configuration 恢复

如果 I/O 或 Scheduler attribution branch 单独造成性能下降，而 Full Configuration 又部分或完全恢复性能，则说明机制之间存在 interaction 或 compensating effect。

该结果不能简单写成某个单项优化无效。

### 情况 E：Full Configuration 在 short-context 下明显提升

如果完整系统取得明显性能提升，则首先检查 workload 是否存在超出预期的 prefix reuse，或者 scheduler optimization 本身是否改善了一般 serving。

确认后才能把结果解释为优化具有超出 long-context 场景的泛化收益，而不能默认归因于 hierarchical caching。

### 情况 F：实验精度不足

如果重复实验波动较大，无法排除实际有意义的 regression，则报告 inconclusive，并保留估计区间。不能因为 point estimate 接近 Baseline 就宣称 no regression。

## 14. 实验边界

本实验专门研究非主要目标 workload 上的性能安全性。

它不研究 long-context reuse 带来的正向收益，该问题已经由 Experiment 1 验证。

它不研究长短请求相互竞争、不同 cache locality 和不同 output length 同时存在时的复杂系统行为，这些问题由 Experiment 3 的 Mixed Workload Serving 统一验证。

因此 Experiment 2 的核心结论保持为：在缺乏显著 long-context reuse 的普通 short-context serving 场景中，Strata 类 cache、I/O 与 scheduler 优化是否能够在预定义的 regression / equivalence rule 下保持 Baseline serving performance，以及任何 regression 会在什么负载条件下出现并由哪一类机制关联解释。
