# Experiment 1: Page Size vs. Cache Reuse

## 1. Objective

本实验定量评估 cache page granularity 对 prefix reuse 的影响，验证在现代 serving workload 下，更小的 page 是否能够减少 page-boundary mismatch，从而提高实际避免重新 prefill 的 token 数量。

本实验不证明“page 越小越好”。实验目标是确定：

- page size 减小时 effective reuse 是否稳定提高；
- 收益出现在哪些 prefix pattern 中；
- 收益在什么 page-size 区间开始趋于饱和；
- capacity pressure 是否会改变这一趋势。

实验输出的代表性 page-size region 进入 Experiment 2。

所有配置遵循 [`00-common-conventions.md`](00-common-conventions.md)。

## 2. Primary causal question

本实验只建立：

```text
configured page size
        ↓
prefix/page boundary alignment
        ↓
effective reused tokens
```

I/O bandwidth、GPU-assisted I/O 和 transfer stall 不属于本实验的主要因果变量。

## 3. Runtime and page-size gate

主路径使用 SGLang 的 `--page-size`。

正式 sweep 前先完成 page-size capability scan。所有进入同一正式曲线的 page size 必须：

1. 由同一个 attention backend 支持；
2. 使用相同 model revision、cache dtype 和 scheduler configuration；
3. 对 hybrid model 满足 recurrent-state tracking/checkpoint 的对齐约束；
4. 能通过 prefix-reuse correctness validation。

不得为了获得更多 page-size 点而在不同点之间切换 attention backend。

如果最终 runtime 将 prefix-match granularity 与 physical cache block 解耦，则本实验改为 sweep **reuse-matching granularity**，并在结果中使用该名称，而不是继续笼统称为 page size。

## 4. Independent variable

唯一主要自变量是 page size。

Page-size candidates 在实现阶段根据目标 model × attention backend 的实际支持范围确定。候选值应覆盖从较细到较粗的多个有效粒度，并尽量保持规则间隔，以观察完整趋势而不是只比较两个端点。

正式结果记录具体 tokens/page，不只记录 small / medium / large 标签。

## 5. Primary isolation mode

Experiment 1 的主要 reuse curve 使用 **warm GPU-resident reuse control** 或等价的无 CPU-restore 条件。

该设计使 prefix reuse 已经可用，但不需要把 CPU→GPU transfer latency 混入 Experiment 1 的核心结论。此时重点记录“能够复用多少 token”，而不是“恢复这些 state 要花多少时间”。

如果 runtime 无法构造稳定的 GPU-resident reuse control，可以使用已验证的等价 residency condition，但必须证明 page-size comparison 不因 CPU restore behavior 不同而改变 reuse accounting。

## 6. Controlled variables

同一 page-size sweep 中保持以下条件固定：

- model 与 revision；
- hardware；
- runtime commit；
- attention backend；
- precision 与 cache dtype；
- GPU cache memory budget；
- cache replacement / eviction policy；
- scheduler / overlap configuration；
- I/O backend；
- host-memory layout 与 write policy，如果 HiCache 仍处于 enabled 状态；
- hybrid/recurrent-state tracking parameters；
- 请求集合和请求顺序；
- context length；
- output length；
- prefix reuse pattern；
- concurrency / arrival condition；
- random seed。

不同 page size 使用同一请求 trace。

GPU cache capacity 以固定 memory budget 为主要控制口径。每个 run 同时记录 runtime 实际 token/state capacity 与 allocator padding，避免固定 page 数量导致不同 page size 获得不同总内存预算。

## 7. Workload design

### 7.1 Controlled prefix-boundary workload

该 workload 构造具有明确共享 prefix 的请求组，并设置低、中、高三档 logically reusable prefix ratio。

共享 prefix 的 cut points 必须包含与候选 page size **不完全对齐**的长度。如果所有 prefix length 都恰好是所有 page size 的共同倍数，实验会人为消除 page-boundary reuse loss。

每个 prefix condition 独立执行完整 page-size sweep。

该 workload 是 Experiment 1 的主要机制证据。

### 7.2 Mixed-reuse workload

该 workload 包含多个 context。不同请求具有不同 shared-prefix length，并按固定 trace 重复访问。

该 workload 用于验证 controlled boundary experiment 的趋势是否在不规则 prefix distribution 下仍然成立。

Mixed workload 不改变 cache locality/scheduler policy。这里研究 prefix boundary，不研究后续 scheduler group 的 locality effect。

### 7.3 No-reuse control

不同请求之间不包含能够形成完整 reusable page 的共享 prefix。

该组作为负对照，用于验证 cache accounting 和 workload generator。如果 no-reuse 场景随 page size 出现类似于 shared-prefix workload 的大幅 effective-reuse improvement，应优先检查 trace、hash match 和指标实现。

## 8. Context-length dimension

主要 workload 覆盖短、中、长三个 context 区间。

不同 context-length 档位保持相同的 prefix-ratio definition 和 boundary-generation rule。

Context length 是 robustness dimension，不与 page size 同时解释为双重主因。

## 9. Cache-pressure robustness

主 reuse curve 在 capacity 相对充足的条件下完成，以减少 eviction 对 page-boundary effect 的干扰。

随后增加一个受限 GPU cache budget 的 robustness condition，检查 page granularity 在容量压力下是否改变：

- actual cache occupancy；
- reusable-state eviction；
- effective reuse。

该部分不进行完整 cache-capacity sweep，也不研究 hierarchical cache value。它只检查 page-size conclusion 是否依赖于近乎无限的 reusable-cache capacity。

如果 runtime 的 page size 只改变 prefix matching、不改变 physical allocation，则不得把这里的差异解释为“大 page 无效占用”。这种解释只能在实际 occupancy 数据支持时成立。

## 10. Core metrics

### 10.1 Effective reused tokens

实际避免重新执行 prefix prefill 的 token 数量。

这是本实验的核心绝对指标。

### 10.2 Reuse efficiency

```text
reuse efficiency = effective reused tokens / logically reusable prefix tokens
```

该指标直接量化 page boundary 造成的 reuse loss，并允许不同 prefix length 之间进行可解释比较。

### 10.3 Cache hit accounting

记录 runtime page/block hit、matched prefix tokens 和 tier/residency information，用于验证 effective reuse 的来源。

Page-level hit count 不能替代 effective reused tokens。

### 10.4 Occupancy and eviction counters

记录实际 cache bytes / token capacity、allocator padding、eviction 和 residency supporting counters。

这些指标用于解释 cache-pressure robustness，不作为主要 reuse 结论的替代指标。

## 11. Execution procedure

每个实验单元由固定 model、hardware、runtime、attention backend、workload、context length、prefix condition 和 cache budget 定义。

在实验单元内部：

1. 初始化到规定 cache state；
2. 执行统一 warm-up；
3. 对所有 page-size candidates 运行相同 request trace；
4. page-size 执行顺序进行随机化或平衡排列，避免温度、共享系统状态与单向 sweep 顺序相关；
5. 每个配置重复多次；
6. 每次 run 独立保存 raw result；
7. runtime error、trace mismatch、unexpected eviction/preemption 或 correctness failure 记录为 invalid / unsupported，并保留 reason。

## 12. Analysis plan

### Analysis A: Page size → effective reuse

绘制 page size 与 effective reused tokens / reuse efficiency 的关系。

判断 page size 变细是否稳定减少 boundary loss，以及收益何时开始趋于饱和。

### Analysis B: Prefix alignment sensitivity

比较不同 prefix ratio 和不同 boundary offset 下的曲线。

目标是确认 page-size benefit 来自 boundary alignment，而不是某一组固定 prefix length 的偶然结果。

### Analysis C: Context robustness

比较短、中、长 context 下的 normalized reuse efficiency。

绝对 reused tokens 与 normalized efficiency 同时保留，避免 normalization 隐藏实际规模差异。

### Analysis D: Cache-pressure robustness

比较 capacity-sufficient 与 constrained 条件下的 occupancy、eviction 和 effective reuse。

只有实际 occupancy / eviction 证据支持时，才讨论 page size 对 storage efficiency 的影响。

### Analysis E: Negative control

No-reuse workload 不应表现出与共享-prefix场景类似的 reusable-token gain。

## 13. Selecting representative page-size regions

Experiment 1 不直接选择一个“最佳 page size”。

结果将 page-size space 划分为：

- **coarse / reuse-limited region**：page boundary 明显损失 logically reusable prefix；
- **transition region**：继续减小 page 仍能获得明显 reuse improvement；
- **reuse-saturated region**：继续减小 page 已很少增加 effective reuse。

Experiment 2 从这些区域选择代表性 points，并继续研究 I/O cost。

## 14. Interpretation boundaries

如果 page size 减小提高 effective reused tokens，可以得出 page granularity 影响 prefix reuse efficiency 的结论。

不能仅凭 cache hit count 提高声称实际计算复用增加。

不能在没有 occupancy evidence 时把 reuse improvement 解释为更高的 cache storage utilization。

不能在本实验中根据 latency 或 bandwidth 推断 GPU-assisted I/O 的价值。

如果目标 hybrid model 无法通过完整 state validation，则该模型的 serving-level reuse result 标记为 partial / unsupported，不以 attention-only hit 代替完整 hybrid-state reuse。

## 15. Final conclusion target

Experiment 1 最终回答：

> 在固定 runtime、attention backend、cache budget 和 workload 下，page size 如何改变实际可利用的 prefix reuse，以及 reuse benefit 在什么 granularity 后开始饱和。

该结果为 Experiment 2 提供需要进一步检查 I/O efficiency 的代表性 page-size region。
