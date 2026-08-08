# Experiment 2: Short-context Serving Regression

## 1. 实验目标

本实验用于验证面向 long-context reuse 设计的 hierarchical cache、I/O optimization 和 scheduler optimization，在普通 short-context serving 场景中是否引入明显的性能回退。

本实验重点回答三个问题：

1. Full Configuration 在缺乏显著 long-context reuse 的情况下，是否增加请求处理开销；
2. 各项优化是否会降低 short-context workload 的 throughput，或者恶化 TTFT 与 request completion time；
3. 当系统负载逐渐提高时，优化机制是否会引入额外的 queueing、调度或资源竞争，从而放大 short-context 请求的 tail latency。

本实验不要求优化后的系统优于 Baseline。只要 Full Configuration 在主要性能指标上与 Baseline 保持接近，并且没有出现稳定、显著的性能回退，就可以认为 long-context 优化没有明显损害普通 serving workload。

所有配置遵循 [`00-common-conventions.md`](00-common-conventions.md)。

## 2. Workload 设计

实验使用以独立 short-context 请求为主的 text-only workload。

每个请求具有较短的输入 context，不包含长共享 prefix。不同请求之间尽量避免大规模 prefix overlap，使 hierarchical cache 无法依赖明显的 context reuse 获得额外收益。

请求内容来自多个不同 prompt，而不是反复访问同一个 context。这样能够把实验重点放在优化机制本身的额外开销，而不是 cache hit 带来的计算节省。

实验覆盖多个 short-context length 档位，使结果不依赖某一个特定输入长度。

Output length 同时覆盖较短和相对较长的生成请求，使实验既包含 prefill 与调度开销较明显的情况，也包含 decode 占比更高的情况。

不同 system configuration 使用完全相同的请求集合、input/output length 分布和 request ordering。

## 3. Reuse 条件

本实验的主 workload 保持较低的 reusable-prefix overlap。

如果 runtime 自身存在 unavoidable system prompt、模板或公共 token prefix，则这些公共部分在所有配置中保持完全一致，并记录实际 cache reuse。

实验不通过人为关闭 Baseline 已有的正常 prefix caching 来制造无复用环境，也不主动构造大量相同 prefix。

正式结果需要报告实际发生的 cache reuse，确认 short-context workload 没有意外变成长共享 context workload。

如果实际 reuse 明显高于预期，则该 workload point 不作为主要 regression 结果，而单独标记。

## 4. 系统配置

实验与 Experiment 1 使用完全相同的五种配置：

1. **Baseline**；
2. **Hierarchical Cache**；
3. **Hierarchical Cache + I/O Optimization**；
4. **Hierarchical Cache + Scheduler Optimization**；
5. **Full Configuration**。

Baseline 表示当前项目确定的现代 serving baseline。

Hierarchical Cache 配置用于判断仅启用额外 cache hierarchy 是否对普通短请求产生固定成本。

Hierarchical Cache + I/O Optimization 用于判断额外 I/O path 即使很少真正参与 context restore，是否仍会对 serving pipeline 产生影响。

Hierarchical Cache + Scheduler Optimization 用于检查 scheduler mechanisms 在缺乏明显 cache-locality problem 时是否造成额外 scheduling overhead。

Full Configuration 用于判断完整系统最终是否能够保持 short-context serving 性能。

所有配置保持与 Experiment 1 相同的语义，不能为了 short-context 实验重新修改某个优化机制的定义。

## 5. Cache 初始状态

Short-context regression 的主要结果在统一的 clean initial state 下测量。

每轮实验开始前建立规定的 cache 状态，并确保不同 system configuration 的初始条件可比较。

由于本实验不依赖长期 context reuse，cold-cache 与 warm-cache 不作为独立实验维度进行完整 sweep。

可以在正式测量前执行统一 runtime warm-up，以消除 kernel initialization、memory allocation 和模型首次执行的影响，但 warm-up 请求不得人为建立大量可复用 short-context cache。

正式 measurement window 内观察实际 cache occupancy 和 cache activity，用于确认 hierarchy 没有因为历史残留状态改变 workload 性质。

## 6. Load scaling

每类 short-context workload 在多个 offered-load 条件下执行。

低负载条件用于测量不同系统配置在资源竞争很弱时的固定 overhead。

如果 Full Configuration 在低负载条件下已经稳定增加 TTFT 或 completion time，则说明优化机制本身存在直接 request-path cost。

中等负载条件用于观察 scheduler、cache metadata management 和 background data movement 是否开始与正常请求竞争资源。

高负载条件逐步接近系统饱和区域，用于判断不同配置的最大 serving capacity 是否发生变化，以及额外机制是否放大 P90/P99 latency。

不同系统配置使用相同 arrival schedule。不会根据某个配置的实际处理速度动态修改请求到达模式。

## 7. 控制变量

严格 paired comparison 中保持以下条件一致：

- model identifier 与 revision；
- GPU / CPU resources；
- serving runtime；
- precision 与 cache dtype；
- request dataset；
- input-length distribution；
- output-length distribution；
- request ordering；
- arrival schedule；
- generation settings；
- GPU cache budget；
- batch / concurrency limits；
- measurement window。

实验主要改变 system configuration、short-context length profile 和 offered load。

如果某一配置必须改变 batch limit、cache budget 或其他关键 serving 参数才能正常运行，则该结果不能直接进入主要 paired comparison。

## 8. 核心指标

### TTFT

记录 P50、P90 和 P99 TTFT。

P50 用于判断各项优化是否给普通请求增加固定 latency。P90 和 P99 用于观察高负载条件下是否出现额外 queueing、scheduler interference 或其他 tail amplification。

TTFT 是本实验最重要的 regression 指标之一。

### Throughput

记录 steady-state request throughput 和 token throughput。

该指标用于判断额外 cache hierarchy 与 scheduler mechanisms 是否降低系统的整体 serving capacity。

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
- achieved request rate。

这些指标只用于解释 regression，不作为本实验的主要评价目标。

例如，如果 Full Configuration 的 P99 TTFT 上升，同时 cache reuse 接近零且 scheduler queueing 增加，则更可能说明额外 scheduling path 对非目标 workload 产生了负面影响。

## 10. 实验执行流程

每个 system configuration 首先完成统一的模型加载和 runtime warm-up。

随后恢复规定的初始 cache 状态，并运行固定 short-context request trace。

每个 workload point 使用固定 request set、seed 和 arrival schedule。

正式 measurement window 覆盖足够多的请求，使 P50、P90 和 P99 latency 具有稳定统计意义。

不同 system configuration 的运行顺序交替或随机化，避免机器温度、系统负载或其他时间相关因素系统性偏向某一配置。

每个 workload point 进行多次独立重复测量。

每个 run 保存 system configuration、workload identifier、input/output length profile、offered load、achieved load、cache activity、repetition index 和 validity status。

## 11. Regression 判定方式

本实验不应只根据单个平均值判断是否发生 regression。

首先比较 Full Configuration 与 Baseline 的 throughput、P50/P90/P99 TTFT 和 request completion time。

结果同时报告绝对值和相对变化。

Regression 判断基于多次重复实验中的稳定差异，而不是单次运行中的小幅波动。

低负载下主要关注 fixed overhead。中高负载下主要关注 serving capacity 和 tail latency。

如果某项指标只有非常小的变化，并且变化幅度与重复实验的自然波动接近，则不解释为明确 regression。

如果 Full Configuration 在多个相邻 workload points 上持续表现出同方向劣化，并且差异超过运行波动，则认为存在稳定 regression。

## 12. 结果组织

结果首先按照 **short-context profile × offered load** 组织。

每个 workload point 横向比较五种 system configuration。

主结果至少形成：

1. P50 / P90 / P99 TTFT 随 offered load 的变化；
2. request throughput 随 offered load 的变化；
3. token throughput 随 offered load 的变化；
4. request completion time 随 offered load 的变化；
5. GPU utilization 随 offered load 的变化；
6. Full Configuration 相对 Baseline 的 regression summary。

最终形成 regression summary table，报告 workload、load、throughput delta、P50/P99 TTFT delta、completion-time delta 与 regression conclusion。

## 13. 结果判断逻辑

### 情况 A：Full Configuration 与 Baseline 基本一致

如果不同负载下 throughput 和 latency 均与 Baseline 接近，并且差异处于实验波动范围内，则说明针对 long-context reuse 的优化没有对普通 short-context serving 造成明显 regression。

### 情况 B：低负载存在稳定 latency overhead

如果低负载时 Full Configuration 的 TTFT 或 completion time 已经稳定高于 Baseline，而 queueing 几乎不存在，则说明额外机制在普通 request path 中引入了固定成本。

此时根据逐步配置对比判断 overhead 主要来自 hierarchy、I/O path 还是 scheduler。

### 情况 C：低负载正常，高负载出现 regression

如果低负载结果接近，但中高负载下 throughput 降低或 P99 TTFT 明显增加，则说明额外机制本身的固定成本较小，但在资源竞争条件下会降低 serving efficiency。

此时结合 scheduler queueing、GPU utilization、batch behavior 和 background activity 定位原因。

### 情况 D：某个中间配置出现 regression，但 Full Configuration 恢复

如果 Hierarchical Cache 或 Scheduler Optimization 单独造成性能下降，而 Full Configuration 又部分或完全恢复性能，则说明不同机制之间存在 compensating effect。

该结果不能简单写成某个单项优化无效，而应结合前面机制实验解释完整组合为什么能够抵消局部开销。

### 情况 E：Full Configuration 在 short-context 下反而明显提升

如果完整系统取得明显性能提升，则首先检查 workload 是否存在超出预期的 prefix reuse，或者现代 scheduler optimization 本身是否改善了一般 serving。

确认后才能把结果解释为优化具有超出 long-context 场景的泛化收益，而不能默认归因于 hierarchical caching。

## 14. 实验边界

本实验专门研究非主要目标 workload 上的性能安全性。

它不研究 long-context reuse 带来的正向收益，该问题已经由 Experiment 1 验证。

它不研究长短请求相互竞争、不同 cache locality 和不同 output length 同时存在时的复杂系统行为，这些问题由 Experiment 3 的 Mixed Workload Serving 统一验证。

因此 Experiment 2 的核心结论保持为：在缺乏显著 long-context reuse 的普通 short-context serving 场景中，Strata 类 cache、I/O 与 scheduler 优化是否能够保持接近 Baseline 的 serving performance，以及任何 regression 会在什么负载条件下出现并由哪一层机制引入。
