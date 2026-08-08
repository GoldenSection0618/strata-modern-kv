# Experiment 3: Mixed Workload Serving

## 1. 实验目标

本实验用于验证 hierarchical cache、I/O optimization 和 scheduler optimization 在异构 serving workload 中能否保持稳定的端到端系统收益。

实验同时引入 long-context reuse 请求和普通 short-context 请求，并使请求具有不同的 output length 和 cache locality，从而形成更接近实际在线 serving 的资源竞争环境。

本实验重点回答四个问题：

1. 当 long-context 与 short-context 请求同时存在时，Full Configuration 是否仍然能够提高整体 serving efficiency；
2. 针对 long-context reuse 的优化是否会通过 cache、I/O 或 scheduling 竞争损害 short-context 请求；
3. 当 workload composition、cache locality 和 output-length heterogeneity 发生变化时，系统收益是否仍然稳定；
4. Full Configuration 是否能够在接近饱和的系统负载下同时维持 throughput、tail latency 和不同请求类型之间的服务质量。

本实验不重新验证单项机制的因果有效性。前述实验组负责解释 cache、I/O 和 scheduler 为什么有效，本实验负责判断这些机制组合后能否处理 heterogeneous workload。

所有配置遵循 [`00-common-conventions.md`](00-common-conventions.md)。

## 2. 请求类型

Mixed workload 由两类基本请求组成。

### Long-context reuse requests

该类请求包含较长的 shared prefix。

请求被划分为多个独立 shared-context groups。同一 group 内的请求复用相同或高度重叠的 long prefix，不同 group 之间使用不同 context。

该类请求用于持续产生 reusable state、cache eviction、CPU-tier restore 和相关 scheduling demand。

### Short-context requests

该类请求具有较短且基本独立的 input context。

不同 short-context 请求之间不主动构造大规模 reusable prefix。

该类请求用于模拟普通在线请求，并作为观察 cross-class interference 的主要对象。

两类请求使用统一 serving endpoint 和相同模型执行，不拆分到独立 GPU 或独立服务实例。

## 3. 基准 Mixed Workload

实验首先定义一个固定的 representative mixed workload，作为 Experiment 3 的主 workload。

该 workload 同时包含 long-context reuse requests 和 short-context requests，两类请求在整个 trace 中持续交错到达。

Long-context 请求不集中在单一 shared prefix，而分布在多个 context groups 中。

Short-context 请求分布在 long-context 请求之间，不单独形成连续时间段。

Long-context 和 short-context 请求内部都包含预定义的 generation-length variation，使 serving 系统同时面对不同 prefill cost 和 decode duration，同时避免 output length 与 request class 完全绑定。

主 workload 使用 moderate cache locality，不构造连续重复访问同一 context 的极端最佳情况，也不采用完全破坏 reuse 的极端最差情况。

Representative workload 的 composition、locality profile、output-length distribution、request pool 与 random seed 在正式 experiment matrix 前冻结。

该 workload 用于完成最完整的 system-configuration 和 load-scaling comparison。

## 4. Workload composition

Mixed workload 使用请求类型比例定义 workload composition。

主实验采用一个 balanced composition，使 long-context reuse 请求和 short-context 请求都占据足够比例，从而能够同时观察两类请求的性能。

在 robustness analysis 中额外设置：

- long-context dominant composition；
- balanced composition；
- short-context dominant composition。

不同 composition 保持 request pool construction rule、context groups、length profiles、locality policy 和 generation settings 一致，主要改变两类请求的相对比例。

### Operational composition sensitivity

第一层比较保持相同总 request-arrival schedule，只改变 request-class ratio。

这组结果反映真实业务流量从 short-heavy 变为 long-heavy 后系统会如何变化。由于 long-context 与 short-context 请求的 token/compute work 不同，总 offered work 会随 composition 改变，因此该结果不能被解释为“composition 本身”的纯因果效应。

### Matched-work composition control

为避免把更多 input/output token work 错误归因于 composition，本实验在代表性中高负载条件补充 matched-work control。

匹配规则在查看 Full Configuration 正式结果前冻结，可使用 offered token volume、Baseline load fraction 或其他可验证 work proxy。该 control 用于判断在总体工作压力近似可比时，不同 class mixture 是否仍然改变 cache、queueing、scheduler behavior 与 cross-class latency。

Composition robustness 不与所有其他 workload 变量进行完整笛卡尔积组合。

## 5. Cache locality

Long-context 请求具有可控的 reuse distance。

主 workload 使用 moderate locality，使同一 shared context 会被稳定 revisit，同时保持不同 context groups 之间的竞争。

在 locality robustness analysis 中设置三个代表性条件：

- high locality；
- moderate locality；
- low locality。

High locality 条件下，相同 context 的请求更靠近。

Low locality 条件下，相同 context 的 revisit 距离更长，并增加 reusable working set 的竞争。

不同 locality trace 保持 long-context request count、shared-context group 数量、request-class ratio、input/output length distribution 和 offered-load schedule 一致。

实验不通过改变 long-context 请求比例来间接改变 locality。

Locality sweep 主要在 balanced composition 下执行，从而尽可能隔离 reuse-distance structure 的影响。

## 6. Output-length heterogeneity

主 mixed workload 同时包含不同 generation length 的请求。

Output length 不与 request class 完全绑定。Long-context 请求和 short-context 请求内部都保留 output-length variation，避免把 input class 和 decode duration 混成同一个变量。

实验额外设置一个 relatively homogeneous output-length control workload，与主 heterogeneous workload 对比。

### Operational output-length sensitivity

第一层比较保持相同 request-arrival schedule，只改变 output-length distribution。

该比较回答业务请求生成长度变得更加异构后系统会怎样表现。由于不同 output distribution 会改变总 decode token work，它不能单独证明 heterogeneity 本身导致了性能变化。

### Matched-work output-length control

在需要判断 decode-duration heterogeneity 是否引入额外 scheduling interference 时，补充 matched-work control，使 aggregate offered output work 或预先定义的 load proxy 近似可比。

若 matched-work 条件下 heterogeneous workload 仍然出现更严重的 queueing、batch inefficiency 或 tail latency，则可以更有力地说明异构 decode duration 本身与调度行为有关。

Output-length robustness 只在代表性 composition 和 load 条件下执行，不进行大规模 sweep。

## 7. 系统配置

实验使用与 Experiments 1 和 2 相同的五种系统配置：

1. **Baseline**；
2. **Hierarchical Cache**；
3. **Hierarchical Cache + I/O Optimization**；
4. **Hierarchical Cache + Scheduler Optimization**；
5. **Full Configuration**。

Hierarchical Cache 使用 reference/baseline I/O 与 scheduler path。

Hierarchical Cache + I/O Optimization 保持 reference/baseline scheduler。

Hierarchical Cache + Scheduler Optimization 保持 reference/baseline I/O path。

Full Configuration 同时启用经过验证的 hierarchy、I/O 和 scheduler mechanisms。

配置 3 与配置 4 是 parallel attribution branches，不按逐层 feature chain 解释。

Experiment 3 不针对 mixed workload 单独修改 mechanism semantics，使系统配置能够与前两个实验直接比较。

## 8. Load scaling

Representative mixed workload 在多个 offered-load 条件下运行。

低负载条件用于确认两类请求共存时不存在明显 fixed overhead 或非必要 interference。

中等负载条件用于观察 cache、I/O、GPU compute 和 scheduler 开始发生资源竞争后的系统行为。

高负载条件逐步接近 saturation region，用于观察 Full Configuration 是否能够提高 system capacity，并抑制 queueing 和 tail-latency amplification。

正式 load grid 在 calibration 后冻结。所有 system configurations 使用完全相同的 arrival schedule、request sequence 和 output targets。

当系统出现持续 queue accumulation、throughput plateau 和快速增长的 latency 时，该 workload point 被视为进入 saturation region。不同配置使用同一 saturation rule。

## 9. 实验矩阵

本实验采用“主矩阵 + targeted robustness checks”的结构，不执行所有变量的完整笛卡尔积。

### Primary matrix

主矩阵固定：

- balanced request composition；
- moderate cache locality；
- heterogeneous output length。

在该条件下完整执行：

```text
5 system configurations
×
multiple offered-load points
```

Primary matrix 用于形成 Experiment 3 的主要 end-to-end 结论。

### Composition robustness

在代表性的 medium-load 和 high-load 区域比较 long-context dominant、balanced 与 short-context dominant compositions。

先报告 same-request-arrival operational sensitivity，再对关键 point 补充 matched-work control。

### Locality robustness

在 balanced composition 下，于代表性中高负载条件比较 high、moderate 与 low locality。

由于 request-class ratio 和 length distribution 保持固定，该比较可以更直接解释为 reuse-distance / locality 变化带来的系统影响。

### Output-length robustness

在 balanced composition 和 representative load 下比较 relatively homogeneous 与 heterogeneous output-length distributions。

先报告 operational sensitivity，再对关键 point 补充 matched-work control。

这种设计保留主要 workload 维度，同时避免形成规模过大的全组合实验。

## 10. 控制变量

每组 paired system-configuration comparison 保持以下条件一致：

- model identifier 与 revision；
- GPU / CPU resources；
- serving runtime version / commit；
- precision 与 cache dtype；
- GPU reusable-cache budget；
- generation parameters；
- concurrency limits；
- workload trace；
- offered-load schedule；
- measurement rule 与 measurement window；
- random seed。

所有启用 hierarchy 的配置保持相同 CPU-tier budget、host-memory policy 与 offload policy。

研究某个 workload variable 时，其余主要变量保持固定或显式做 matched-work control。

Composition experiment 主要改变 request-class ratio。

Locality experiment 只改变 long-context reuse ordering / distance structure。

Output-length experiment 主要改变 generation-length distribution。

任何无法保持严格单变量控制的 comparison 都必须明确标记为 operational sensitivity，而不是 causal attribution。

## 11. 核心指标

实验同时记录 system-level 和 request-class-level 指标。

### Overall throughput

记录整体 request throughput 和 token throughput。

两种 throughput 同时报告，避免 composition 与 output-length 变化使单一 request/s 或 token/s 指标产生误导。

### Overall TTFT

记录所有请求的 P50、P90 和 P99 TTFT。

### Request completion time

记录整体 completion-time distribution。

### GPU utilization

记录正式 measurement window 内的 GPU utilization。

GPU utilization 用于辅助判断系统是否因为 mixed workload 出现 GPU idle、stall 或低效 batch execution，不单独作为性能优劣结论。

## 12. Request-class-level metrics

Mixed workload 不能只报告 aggregate metrics。

### Long-context requests

记录：

- request throughput；
- token throughput；
- P50 / P90 / P99 TTFT；
- request completion time；
- cache reuse realization；
- recomputation；
- restore / I/O activity。

### Short-context requests

记录：

- request throughput；
- token throughput；
- P50 / P90 / P99 TTFT；
- request completion time；
- queueing time。

Class-level metrics 用于防止 aggregate throughput 掩盖请求类型之间的不公平。

Full Configuration 即使提高总体 throughput，如果 short-context 或 long-context 的 P99 TTFT 出现稳定且具有实际意义的恶化，也不能直接认定 mixed-workload performance 全面改善。

## 13. Cross-class interference

实验将 mixed workload 的 class-level performance 与 Experiments 1 和 2 中可匹配的单一 workload performance 进行辅助比较。

该比较用于观察：

- long-context 请求是否显著拖慢 short-context 请求；
- short-context 请求是否降低 long-context reuse realization；
- decode-heavy 请求是否增加其他请求的 queueing；
- hierarchy restore 是否与普通 serving 请求产生资源竞争。

Cross-class comparison 不能只用“相似 request rate”匹配。至少需要同时核对 class-specific input/output work、total offered work、context/reuse condition 和 system load region。

如果 Experiments 1/2 中不存在足够匹配的 point，则补充 matched control run，而不是强行复用不等价结果。

该分析用于量化 interference，不替代同一 mixed trace 下五种 system configurations 的严格 paired comparison。

## 14. 辅助系统指标

为了解释 mixed workload 下出现的性能变化，实验同时记录：

- GPU / CPU cache hit volume；
- reusable-state eviction；
- recomputation；
- CPU-GPU transfer volume；
- non-overlapped I/O stall；
- queueing time；
- scheduler decision / waiting behavior；
- batch composition；
- GPU idle / stall；
- TPOT 或等价 decode metric when needed；
- long-context 与 short-context 请求的资源占用比例 when observable。

这些指标用于建立以下证据链：

```text
workload heterogeneity
        ↓
cache / compute / I/O competition
        ↓
scheduler and batching behavior
        ↓
class-level queueing and execution
        ↓
overall throughput and tail latency
```

## 15. 实验执行流程

每个 system configuration 首先执行统一 runtime warm-up。

随后恢复规定的初始 cache 状态，并运行固定 mixed request trace。

正式 measurement window 覆盖足够多的 shared-context revisit 和 short-context 请求，使两类请求均具有稳定统计量。

同一 workload point 的五种 system configurations 使用完全相同的 request trace、arrival timestamps、request classes、output targets 和 seed。

每个配置进行多次独立重复测量。

不同配置的执行顺序交替或随机化。

每次 run 保存完整 workload composition、locality profile、output-length distribution、offered request rate、offered token/work summary、system configuration、repetition index 和 validity status。

Operational sensitivity 与 matched-work control 使用不同 workload identifiers，不能混入同一个 aggregation。

## 16. Validity conditions

进入主结果的 run 必须满足以下条件：

- 实际 request-class ratio 与目标 composition 一致；
- long-context reuse distance 与目标 locality profile 一致；
- output targets / realized output distribution 与目标配置一致；
- paired system configurations 使用相同 request trace；
- offered-load schedule 未被修改；
- GPU reusable-cache budget 在 paired comparison 中一致；
- hierarchy configurations 使用相同 CPU-tier budget 与 offload policy；
- cache hierarchy、I/O 和 scheduler mechanism 未发生未记录 fallback；
- measurement window 内两类请求均具有足够样本；
- matched-work control 达到预先定义的匹配容差；
- 未发生破坏 paired comparison 的 OOM、runtime failure 或 instrumentation failure。

任何 invalid run 均保留 raw data，并记录具体 invalid reason。

## 17. 结果组织

### Primary matrix

至少形成：

1. overall request / token throughput 随 offered load 的变化；
2. overall P50 / P90 / P99 TTFT 随 offered load 的变化；
3. overall request completion time 随 offered load 的变化；
4. GPU utilization 随 offered load 的变化；
5. long-context class-level request/token throughput、TTFT 和 completion time；
6. short-context class-level request/token throughput、TTFT 和 completion time；
7. representative medium-load / high-load points 下五种 system configurations 的直接比较。

### Composition robustness

分别展示：

- same-request-arrival operational sensitivity；
- representative matched-work attribution control。

结果同时报告 overall 与 class-level performance，并显示 offered token/work summary。

### Locality robustness

展示 high、moderate、low locality 下 actual reuse-distance summary、cache reuse realization、restore / recomputation behavior 和端到端性能。

### Output-length robustness

分别展示 operational sensitivity 与 matched-work control，重点观察 overall/class-level tail latency、throughput、queueing 和 batch behavior。

主要结果与 robustness checks 分开呈现，不把所有 workload dimensions 堆叠在一张图中。

## 18. 结果判断逻辑

### 情况 A：Full Configuration 在 mixed workload 下保持稳定综合收益

如果 Full Configuration 提高总体 serving capacity，并降低或保持 overall tail latency，同时 long-context 和 short-context 两类请求均未出现明显 regression，则说明前述机制能够稳定扩展到 heterogeneous serving workload。

### 情况 B：总体收益存在，但 short-context 或 long-context 请求明显恶化

如果 overall throughput 提高，但任一主要 request class 的 P90/P99 TTFT 或 completion time 明显恶化，则说明存在 cross-class interference。

此时不能将结果表述为无条件的系统整体提升。需要结合 queueing、batch composition、restore activity 和 scheduler behavior 定位 interference 来源。

### 情况 C：收益只在 long-context dominant operational workload 中存在

如果 same-request-rate comparison 中 Full Configuration 只在 long-context dominant workload 明显受益，首先判断这是更高 reusable-work proportion 带来的真实 operational value，还是仅由不同总 offered work 造成。

Matched-work control 用于进一步区分这两种解释。

### 情况 D：收益对 cache locality 高度敏感

如果 high locality 条件下收益明显，而 moderate 或 low locality 下迅速下降，则说明 Full Configuration 的实际价值依赖较强 context reuse / reuse distance structure。

如果 low locality 下 hierarchy restore 和 eviction activity 增加，但有效 reuse 不足，则需要将额外 data movement cost 纳入解释。

### 情况 E：Output-length heterogeneity 引起明显退化

如果 operational comparison 中 heterogeneous workload 表现更差，首先检查总 output work 是否同步增加。

只有 matched-work control 仍然显示更高 queueing、batch inefficiency 或 tail latency 时，才能更有力地将退化与 decode-duration heterogeneity / scheduling interaction 联系起来。

### 情况 F：I/O 与 Scheduler attribution branch 表现不同

如果 Hierarchical Cache + I/O 与 Hierarchical Cache + Scheduler 在 mixed workload 下表现不同，则分别结合 I/O stall、queueing 与 class-level interference 判断当前 workload 中哪类机制更重要。不能按固定 feature accumulation 顺序解释。

### 情况 G：Full Configuration 与现代 Baseline 接近

如果在各类 mixed workload 中 Baseline 与 Full Configuration 长期接近，则结合前述机制实验判断现代 runtime 是否已经吸收相似优化，或者现代 hybrid architecture 已经降低原 Strata bottleneck 的重要性。

该结果本身仍然是有效的现代化复现结论。

## 19. 实验边界

Experiment 3 不扩大模型和硬件维度。

完整的 model / hardware generalization 由后续独立实验组负责。

Experiment 3 也不重复 scheduler component ablation。Delay-hit mitigation、balanced batching 和 stall hiding 的独立贡献已经由 scheduler 实验组验证。

本实验只关注一个最终问题：在 long-context、short-context、不同 generation length 和不同 cache locality 同时存在的异构 serving 条件下，前述优化能否继续转化为稳定的系统级收益，并避免把性能成本转移给另一类请求。
