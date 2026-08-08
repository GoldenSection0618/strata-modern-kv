# Experiment 3: Mixed Workload Serving

## 1. 实验目标

本实验用于验证 hierarchical cache、I/O optimization 和 scheduler optimization 在异构 serving workload 中能否保持稳定的端到端系统收益。

实验同时引入 long-context reuse 请求和普通 short-context 请求，并使请求具有不同的 output length 和 cache locality，从而形成更接近实际在线 serving 的资源竞争环境。

本实验重点回答四个问题：

1. 当 long-context 与 short-context 请求同时存在时，Full Configuration 是否仍然能够提高整体 serving efficiency；
2. 针对 long-context reuse 的优化是否会通过 cache、I/O 或 scheduling 竞争损害 short-context 请求；
3. 当 workload composition、cache locality 和 output-length heterogeneity 发生变化时，系统收益是否仍然稳定；
4. Full Configuration 是否能够在接近饱和的系统负载下同时维持 throughput、tail latency 和不同请求类型之间的服务质量。

本实验不重新验证单项机制的因果有效性。前述实验组负责解释 cache、I/O 和 scheduler 为什么有效，本实验负责判断这些机制组合后能否处理现实中的 heterogeneous workload。

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

Long-context 和 short-context 请求都包含不同的 generation length，使 serving 系统同时面对不同 prefill cost 和 decode duration。

主 workload 使用中等程度的 cache locality，不构造连续重复访问同一 context 的极端最佳情况，也不采用完全破坏 reuse 的极端最差情况。

该 representative workload 用于完成最完整的 system-configuration 和 load-scaling comparison。

## 4. Workload composition

Mixed workload 使用固定的请求类型比例定义 workload composition。

主实验采用一个 balanced composition，使 long-context reuse 请求和 short-context 请求都占据足够比例，从而能够同时观察两类请求的性能。

在 robustness analysis 中额外设置：

- long-context dominant composition；
- balanced composition；
- short-context dominant composition。

三个 composition 保持总请求到达规模和其他 workload 属性可比较，只改变两类请求的相对比例。

该设计用于判断 Full Configuration 的收益是否只在 long-context 请求占绝对多数时成立。

Composition sweep 不与所有其他 workload 变量进行完整笛卡尔积组合，只在代表性的中高负载条件下执行。

## 5. Cache locality

Long-context 请求具有可控的 reuse distance。

主 workload 使用 moderate locality，使同一 shared context 会被稳定 revisit，同时保持不同 context groups 之间的竞争。

在 locality robustness analysis 中设置三个代表性条件：

- high locality；
- moderate locality；
- low locality。

High locality 条件下，相同 context 的请求更靠近。

Low locality 条件下，相同 context 的 revisit 距离更长，并增加 reusable working set 的竞争。

不同 locality trace 保持 long-context 请求数量、shared-context group 数量、input/output length distribution 和总体 arrival condition 一致。

实验不通过改变 long-context 请求比例来间接改变 locality。

Locality sweep 主要在 balanced composition 下执行，从而隔离 cache locality 本身的影响。

## 6. Output-length heterogeneity

主 mixed workload 同时包含不同 generation length 的请求。

请求按照预先固定的 output-length distribution 生成，使短生成和长生成请求能够在同一个 serving trace 中共存。

Output length 不与 request class 完全绑定。Long-context 请求和 short-context 请求内部都保留一定的 output-length variation，避免把 input class 和 decode duration 混成同一个变量。

实验额外设置一个相对 homogeneous 的 output-length control workload，与主 heterogeneous workload 对比。

该对比主要在 balanced composition 和代表性负载下执行。

实验通过这一控制判断，当部分请求长期占据 decode resources 时，scheduler optimization 是否仍然能够维持合理的 batch formation 和 request latency。

Output-length heterogeneity 不进行大规模独立 sweep。

## 7. 系统配置

实验使用与 Experiments 1 和 2 相同的五种系统配置：

1. **Baseline**；
2. **Hierarchical Cache**；
3. **Hierarchical Cache + I/O Optimization**；
4. **Hierarchical Cache + Scheduler Optimization**；
5. **Full Configuration**。

所有机制的定义、cache budget 和 runtime semantics 与前两个实验保持一致。

Experiment 3 不针对 mixed workload 单独调整某项机制，使系统配置能够与前两个实验直接比较。

Full Configuration 代表最终系统。

## 8. Load scaling

Representative mixed workload 在多个 offered-load 条件下运行。

低负载条件用于确认两类请求共存时不存在明显固定 overhead 或非必要 interference。

中等负载条件用于观察 cache、I/O、GPU compute 和 scheduler 开始发生资源竞争后的系统行为。

高负载条件逐步接近 saturation region，用于观察 Full Configuration 是否能够提高系统 capacity，并抑制 queueing 和 tail-latency amplification。

所有系统配置使用完全相同的 arrival schedule。

Long-context 和 short-context 请求的到达顺序在 paired runs 中保持一致。

当系统出现持续 queue accumulation、throughput plateau 和快速增长的 latency 时，该 workload point 被视为进入 saturation region。

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

在代表性的 medium-load 和 high-load 条件下比较：

- long-context dominant；
- balanced；
- short-context dominant。

该组实验判断系统收益对请求类型比例的敏感性。

### Locality robustness

在 balanced composition 下，于代表性中高负载条件比较：

- high locality；
- moderate locality；
- low locality。

该组实验判断系统收益是否依赖理想 cache locality。

### Output-length robustness

在 balanced composition 和 representative load 下比较：

- relatively homogeneous output length；
- heterogeneous output length。

该组实验判断不同 decode duration 共存后是否出现新的 scheduling interference。

这种设计保留主要 workload 维度，同时避免形成规模过大的全组合实验。

## 10. 控制变量

每组 paired comparison 保持以下条件一致：

- model identifier 与 revision；
- GPU / CPU resources；
- serving runtime；
- precision 与 cache dtype；
- GPU cache budget；
- CPU-tier budget；
- generation parameters；
- concurrency limits；
- workload request pool；
- offered-load schedule；
- measurement window；
- random seed。

研究某个 workload variable 时，其余主要变量保持固定。

Composition experiment 只改变 request-class ratio。

Locality experiment 只改变 long-context reuse ordering。

Output-length experiment 只改变 generation-length distribution。

这种设计避免将多个 workload 变化同时归因于某一个系统机制。

## 11. 核心指标

实验同时记录 system-level 和 request-class-level 指标。

### Overall throughput

记录整体 request throughput 和 token throughput。

该指标用于判断 Full Configuration 在 heterogeneous workload 下是否提高总 serving capacity。

### Overall TTFT

记录所有请求的 P50、P90 和 P99 TTFT。

该指标用于评价系统整体 latency behavior。

### Request completion time

记录整体 completion-time distribution。

该指标用于观察不同 output length 和资源竞争对完整请求执行过程的影响。

### GPU utilization

记录正式 measurement window 内的 GPU utilization。

该指标用于辅助判断系统是否因为混合 workload 出现 GPU idle、stall 或低效 batch execution。

## 12. Request-class-level metrics

Mixed workload 不能只报告 aggregate metrics。

实验必须分别统计：

### Long-context requests

- P50 / P90 / P99 TTFT；
- request completion time；
- achieved throughput；
- cache reuse realization；
- recomputation；
- restore / I/O activity。

### Short-context requests

- P50 / P90 / P99 TTFT；
- request completion time；
- achieved throughput；
- queueing time。

Class-level metrics 用于防止 aggregate throughput 掩盖请求类型之间的不公平。

例如，Full Configuration 即使提高总体 throughput，如果 short-context P99 TTFT 明显恶化，也不能直接认定 mixed-workload performance 全面改善。

## 13. Cross-class interference

实验将 mixed workload 的 class-level performance 与 Experiments 1 和 2 中相近负载下的单一 workload performance 进行辅助比较。

该比较用于观察：

- long-context 请求是否显著拖慢 short-context 请求；
- short-context 请求是否降低 long-context reuse realization；
- decode-heavy 请求是否增加其他请求的 queueing；
- hierarchy restore 是否与普通 serving 请求产生资源竞争。

该分析只用于量化 mixed-workload interference，不替代严格 paired system-configuration comparison。

所有 inference 必须基于 matched 或尽可能接近的 workload conditions。

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

随后恢复规定的 cache 初始状态，并运行固定 mixed request trace。

正式 measurement window 覆盖足够多的 shared-context revisit 和 short-context 请求，使两类请求均具有稳定统计量。

同一 workload point 的五种 system configurations 使用完全相同的 request trace、arrival timestamps、request classes、output-length targets 和 seed。

每个配置进行多次独立重复测量。

不同配置的执行顺序交替或随机化。

每次 run 保存完整 workload composition、locality profile、output-length distribution、offered load、system configuration、repetition index 和 validity status。

## 16. Validity conditions

进入主结果的 run 必须满足以下条件：

- 实际 request-class ratio 与目标 composition 一致；
- long-context reuse distance 与目标 locality profile 一致；
- output-length distribution 与目标配置一致；
- paired configurations 使用相同 request trace；
- offered-load schedule 未被修改；
- cache hierarchy、I/O 和 scheduler mechanism 未发生未记录 fallback；
- measurement window 内两类请求均具有足够样本；
- 未发生破坏 paired comparison 的 OOM、runtime failure 或 instrumentation failure。

任何 invalid run 均保留 raw data，并记录具体 invalid reason。

## 17. 结果组织

Primary matrix 首先形成：

1. overall throughput 随 offered load 的变化；
2. overall P50 / P90 / P99 TTFT 随 offered load 的变化；
3. request completion time 随 offered load 的变化；
4. GPU utilization 随 offered load 的变化；
5. long-context class-level TTFT 和 completion time；
6. short-context class-level TTFT 和 completion time；
7. representative load 下五种 system configurations 的直接比较。

Robustness analysis 分别形成：

- request composition 对 Full Configuration 收益的影响；
- cache locality 对系统收益的影响；
- output-length heterogeneity 对系统收益的影响。

主要结果与 robustness checks 分开呈现，不把所有 workload dimensions 堆叠在一张图中。

## 18. 结果判断逻辑

### 情况 A：Full Configuration 在 mixed workload 下保持稳定收益

如果 Full Configuration 提高总体 throughput，并降低或保持 overall tail latency，同时 long-context 和 short-context 两类请求均未出现明显 regression，则说明前述机制能够稳定扩展到 heterogeneous serving workload。

### 情况 B：总体收益存在，但 short-context 请求明显恶化

如果 overall throughput 提高，但 short-context P90/P99 TTFT 或 completion time 明显恶化，则说明 long-context optimization 产生了 cross-class interference。

此时不能将结果表述为无条件的系统整体提升。

需要结合 queueing、batch composition、restore activity 和 scheduler behavior 定位 interference 来源。

### 情况 C：收益只在 long-context dominant workload 中存在

如果 Full Configuration 只有在 long-context 请求占比较高时明显优于 Baseline，而 balanced 或 short-context dominant workload 中收益消失，则说明系统价值高度依赖 workload composition。

该结果用于界定 Strata 类优化的实际适用区域。

### 情况 D：收益对 cache locality 高度敏感

如果 high locality 条件下收益明显，而 moderate 或 low locality 下迅速下降，则说明 Full Configuration 的实际价值仍然依赖较强 context reuse。

如果 low locality 下 hierarchy restore 和 eviction activity 增加，但有效 reuse 不足，则需要将额外 data movement cost 纳入解释。

### 情况 E：Output-length heterogeneity 引起明显退化

如果 homogeneous workload 表现正常，而 heterogeneous output length 下 P99 latency 或 throughput 明显恶化，则说明 decode duration heterogeneity 引入了新的 batching 或 scheduling bottleneck。

该结果需要与 scheduler-related auxiliary metrics 联合解释。

### 情况 F：Full Configuration 与现代 Baseline 接近

如果在各类 mixed workload 中 Baseline 与 Full Configuration 长期接近，则需要结合前述机制实验判断现代 runtime 是否已经吸收相似优化，或者现代 hybrid architecture 已经降低了原 Strata bottleneck 的重要性。

该结果本身仍然是有效的现代化复现结论。

## 19. 实验边界

Experiment 3 不再扩大模型和硬件维度。

完整的 model / hardware generalization 由后续独立实验组负责。

Experiment 3 也不重复 scheduler component ablation。Delay-hit mitigation、balanced batching 和 stall hiding 的独立贡献已经由 scheduler 实验组验证。

本实验只关注一个最终问题：

> 在 long-context、short-context、不同 generation length 和不同 cache locality 同时存在的真实异构 serving 条件下，前述优化能否继续转化为稳定的系统级收益，并避免把性能成本转移给另一类请求。
