# Experiment 1: Long-context Reuse Serving

## 1. 实验目标

本实验用于验证在具有明显 long-context reuse 的 serving workload 中，hierarchical cache、I/O optimization 与 scheduler optimization 是否能够转化为稳定的端到端系统收益。

本实验重点回答三个问题：

1. 长共享 context 的重复访问是否能够减少重复 prefill / recomputation，并最终改善 serving performance；
2. Hierarchical Cache、I/O Optimization 与 Scheduler Optimization 在完整 serving 中分别还有多少增量价值，以及它们同时启用后是否形成稳定的 Full Configuration 收益；
3. 当 request pressure 逐渐提高后，Full Configuration 是否能够改善 serving capacity 与 TTFT，而不是通过明显牺牲另一项指标换取局部收益。

本实验不重新证明 cache、I/O 或 scheduler 单个机制为什么有效。机制层面的因果验证由前述实验组完成，本实验只验证这些机制组合进入完整 serving pipeline 后是否仍然有效。

所有配置遵循 [`00-common-conventions.md`](00-common-conventions.md)。

## 2. Workload 设计

实验使用具有明确 shared-prefix / long-context reuse 的 text-only request trace。

每个 workload 由多个 shared-context groups 组成。每个 group 包含一个较长的共享 context，以及多个基于该 context 的独立请求。Group 内请求共享相同或高度重叠的 prefix，但 suffix / query 与生成内容保持独立。

不同 shared-context groups 之间保持独立，避免所有请求集中在单一热点 context，从而使实验能够同时观察 context reuse 与有限 cache capacity 下的真实竞争行为。

实验设置多个 context-length 档位，覆盖中等长度到较长 context。不同档位主要改变共享 context 的长度，其余请求分布尽量保持一致。

同一 context 会在后续请求中再次出现，从而形成稳定可观测的 cache reuse。Request ordering 不采用只让相同 context 永远连续出现的理想化方式，而是保留代表性的 revisit distance，使结果不只反映极端最佳 locality。

每个正式 workload 包含足够多的请求，使 cold-start 与 steady-state 都具有稳定、可重复的测量样本。

## 3. 系统配置

正式比较使用以下五种配置：

1. **Baseline**；
2. **Hierarchical Cache**；
3. **Hierarchical Cache + I/O Optimization**；
4. **Hierarchical Cache + Scheduler Optimization**；
5. **Full Configuration**。

Baseline 不启用本项目新增的 hierarchical cache、I/O 与 scheduler 优化，作为端到端参照。

Hierarchical Cache 只启用经过前置验证的分层 reusable-state path，并保持 reference/baseline I/O 与 scheduler path。

Hierarchical Cache + I/O Optimization 在相同 hierarchy 上启用前置实验中已经验证的 I/O path，并保持 reference/baseline scheduler。

Hierarchical Cache + Scheduler Optimization 在相同 hierarchy 上启用经过验证的 scheduler mechanisms，并保持 reference/baseline I/O path。

Full Configuration 同时启用最终选定的 hierarchy、I/O 与 scheduler mechanisms，代表本项目的完整系统配置。

配置 3 与配置 4 是 parallel attribution branches。结果不使用“逐层累积”假设解释，而分别判断 I/O 与 scheduler 的增量贡献，并用 Full Configuration 检查二者共同启用时的 interaction。

## 4. Cache 状态

本实验同时观察 cold-start 与 steady-state，但两种状态使用独立、预定义的测量协议。

### Cold-start

每轮 workload 从规定的空 reusable-cache 状态开始。

Cold-start 用于记录首次处理共享 context 的完整成本，并检查完整系统是否通过显著增加首次请求开销来换取后续收益。

Cold-start 的 measurement boundary 在实验配置中固定，不根据不同系统配置的运行速度动态改变。

### Steady-state

Steady-state 使用固定的 cache-population / preconditioning trace 建立可复用 working set。

所有 paired configurations 使用相同逻辑预处理请求和访问顺序。预处理阶段不进入正式性能统计。

正式 measurement 在预先规定的逻辑边界开始，而不是在观察到某个配置“已经稳定”后才开始。测量开始前记录实际 GPU/CPU residency、cache occupancy 与 reuse readiness，用于验证目标状态确实建立成功。

Steady-state 是本实验的主要分析阶段，因为本实验研究的是 long-context reuse serving，而不是单次首次 context processing。

Cold-start 与 steady-state 结果分开报告，不把 initial population cost 与后续 reuse benefit 混成单一平均值。

## 5. Load scaling

每个 context-length workload 在多个 offered-load 条件下运行。

低负载条件用于观察 optimization 本身是否引入额外开销，以及不同配置的基础 request latency 是否接近。

中等负载条件用于观察资源竞争开始后，cache reuse、I/O 与 scheduler 优化能否降低 queueing 与 stall。

高负载条件逐步接近系统饱和，用于比较不同配置的 serving capacity、tail TTFT 与 request completion behavior。

正式 load grid 在 calibration 后冻结。所有 system configurations 使用相同 offered-load points、logical request trace 与 arrival schedule。

当某个配置出现持续 queue accumulation、throughput 不再增长并伴随 latency 快速恶化时，将其视为进入 saturation region。不同配置可以在不同 offered-load point 饱和，但使用相同预定义判定规则。

## 6. 控制变量

同一 paired comparison 中保持以下条件一致：

- model identifier 与 revision；
- GPU / CPU resources；
- serving runtime version / commit；
- precision 与 cache dtype；
- context dataset 与 request trace；
- input-length distribution；
- output target / output-length distribution；
- request arrival schedule；
- GPU reusable-cache budget；
- generation settings；
- batch / concurrency limits；
- measurement rule 与 measurement window。

所有启用 hierarchy 的配置保持相同 CPU-tier budget、host-memory policy 与 offload policy。

实验主要改变 system configuration、context length 与 offered load。

如果某一 system configuration 必须改变其他关键条件才能运行，则该 point 不进入严格 paired comparison，并单独标记原因。

## 7. 核心指标

### Throughput

记录正式 measurement window 内的 request throughput 与 token throughput，并明确 throughput accounting 口径。

两种 throughput 同时保留，可以避免不同 realized output length 使单一 request/s 或 token/s 指标产生误导。

### TTFT

记录 P50、P90 与 P99 TTFT。

P50 用于反映典型请求体验。P90 与 P99 用于判断 hierarchical cache、I/O 与 scheduler 在中高负载条件下是否能够减少严重等待和 tail amplification。

### Request Completion Time

记录完整请求从到达到生成结束的 completion-time distribution。

该指标用于避免系统只改善 prefill / TTFT，却把成本转移到 decode 或后续排队阶段。

### GPU Utilization

记录正式 measurement window 内的 GPU utilization 与必要的 idle / stall observable。

该指标用于辅助判断性能变化是否伴随更有效的 GPU 使用，而不是作为独立性能结论。

## 8. 辅助指标

为把端到端结果与前述机制实验建立证据链，本实验尽可能同时记录：

- GPU / CPU cache hit volume；
- reusable-state eviction；
- recomputation；
- CPU-GPU data movement；
- non-overlapped I/O stall；
- queueing time；
- scheduler stall / idle behavior；
- batch characteristics when relevant。

这些指标用于解释 end-to-end performance，不作为本实验的独立主要结论。

如果 Full Configuration 的 TTFT 或 throughput 改善，同时 cache reuse realization 提高、recomputation 或 I/O stall 降低，则可以把最终系统收益与前置实验中验证过的机制联系起来。

## 9. 实验执行流程

每个 system configuration 首先完成统一的模型加载与 runtime warm-up，避免首次 kernel、allocator 与 initialization cost 污染正式数据。

Cold-start runs 从统一空 cache 状态开始，并执行固定 cold-start trace。

Steady-state runs 首先执行固定 preconditioning trace，验证目标 cache residency / occupancy 后，在预定义 measurement boundary 执行正式 trace。

同一 workload point 的所有系统配置使用相同 request trace、seed、arrival schedule 与 output target。

每个配置进行多次独立重复测量。不同配置的执行顺序交替或随机化，降低机器长期状态变化造成的系统偏差。

每个 run 保存完整 metadata，包括 experiment ID、system configuration、model/runtime/hardware、context-length point、offered request/token work、cache initial state、trace identifier、repetition index 与 validity status。

## 10. Validity conditions

每个进入主结果的正式 run 必须满足：

- request trace 与目标 workload point 一致；
- 目标 system configuration 的 mechanism semantics 已经过前置验证；
- cache / hierarchy path 未发生未记录的 fallback；
- GPU reusable-cache budget 与其他 paired configurations 保持一致；
- 所有 hierarchy configurations 使用相同 CPU-tier budget 与 offload policy；
- offered-load schedule 未因 runtime behavior 被修改；
- cold-start 或 steady-state 初始状态通过 runtime observable behavior 验证；
- measurement boundary 与预定义配置一致；
- measurement window 完整；
- 未发生破坏比较条件的 OOM、runtime failure 或 instrumentation failure。

不满足条件的 run 保留 raw result，并标记为 `partial`、`unsupported` 或 `invalid`，不静默删除。

## 11. 结果组织

结果首先按照 **context length × offered load** 组织。

每个 workload point 横向比较五种 system configurations。

主结果至少形成：

1. request / token throughput 随 offered load 的变化；
2. P50 / P90 / P99 TTFT 随 offered load 的变化；
3. request completion time 随 offered load 的变化；
4. GPU utilization 随 offered load 的变化；
5. cold-start 与 steady-state 的差异；
6. representative medium-load 与 high-load points 下五种系统配置的直接对比。

辅助结果展示 cache reuse、recomputation、data movement、queueing 与 I/O stall，用于解释主结果中的关键差异。

## 12. 结果判断逻辑

### 情况 A：Full Configuration 在中高负载下取得稳定综合收益

如果低负载下与 Baseline 接近，而中高负载下 throughput 更高、P90/P99 TTFT 更低且 request completion time 不恶化，则说明这些优化主要提高了资源竞争条件下的 serving capacity，并成功把 long-context reuse 转化为端到端收益。

### 情况 B：Hierarchical Cache 已获得大部分收益

如果 Hierarchical Cache 已经明显减少 recomputation，并取得大部分 TTFT / throughput 收益，而 I/O 与 scheduler 两个 parallel attribution branch 只带来较小增益，则说明现代模型/runtime 下主要价值来自扩大 reusable working set，原 Strata 的后续 I/O 或 scheduling bottleneck 已经减弱。

### 情况 C：I/O 与 Scheduler 的增量价值不同

如果配置 3 和配置 4 表现明显不同，则分别结合 I/O stall、queueing 和 scheduler behavior 判断哪一类 bottleneck 在当前现代 serving stack 中仍然重要。不能将两者解释成固定的先后累加关系。

### 情况 D：Cache reuse 明显，但端到端收益有限

如果 cache hit / avoided recomputation 明显存在，但 TTFT 与 throughput 改善有限，则结合 CPU-GPU traffic、I/O stall、queueing 与 GPU utilization 判断 reuse benefit 是否被后续系统瓶颈抵消。

### 情况 E：Throughput 提高但 tail latency 恶化

如果 Full Configuration 提高 throughput，但 P99 TTFT 或 request completion time 明显恶化，则结果解释为 throughput-latency trade-off，不能写成无条件的整体性能提升。

### 情况 F：现代 Baseline 已接近 Full Configuration

如果 Baseline 与 Full Configuration 在大多数 workload point 上接近，则进一步核对现代 runtime 是否已经包含相近的 cache、I/O 或 scheduler 能力。若确认存在，应把该结果解释为 Strata 类机制已部分进入现代 baseline，而不是简单判定实验失败。

## 13. 实验边界

本实验只研究 long-context reuse serving 的最终系统收益。

Short-context regression 由 Experiment 2 单独验证。长短请求竞争、不同 output length 与更复杂 cache locality 的联合影响由 Experiment 3 的 Mixed Workload Serving 验证。

本实验不会为了增加结果数量而额外拆分独立 load-scaling 实验。Load scaling 是本实验用于建立 operating behavior 的必要控制维度。
