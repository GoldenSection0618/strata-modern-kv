# Experiment 2: Cross-hardware Conclusion Stability

## 1. 实验目标

本实验用于验证在模型与 workload 结构保持不变的条件下，前述系统瓶颈和优化收益是否能够从 A100 40GB 泛化到 L40 48GB。

本实验重点回答三个问题：

1. A100 上观察到的主要 cache/state、I/O 和 scheduling bottleneck，在 L40 上是否仍然存在；
2. Hierarchical Cache、I/O Optimization 和 Scheduler Optimization 的收益方向是否跨硬件保持一致；
3. GPU 平台变化是否改变各类 bottleneck 的相对重要性和优化的有效 operating region。

本实验不要求两个 GPU 获得相同的 absolute throughput 或 latency。主要判断对象是 bottleneck location、mechanism behavior 和 normalized optimization effect 是否稳定。

所有正式比较遵循 [`00-common-conventions.md`](00-common-conventions.md) 与仓库根目录 [`../../../docs/TECHNICAL_BASELINE.md`](../../../docs/TECHNICAL_BASELINE.md)。

## 2. 实验总体设计

A100 40GB 作为 reference platform，L40 48GB 作为 hardware-generalization platform。

实验使用前置实验已经通过 validity gate 并冻结的 representative points，不重新执行完整 workload sweep。

正式比较保持 model、logical request trace、workload structure、system-configuration semantics 与 measurement rule 不变，将 hardware platform 作为主要变化变量。

实验分别对 Qwen3.5-9B 和 Gemma 4 12B 执行 matched hardware comparison。两个模型不需要覆盖前五组全部实验点，只覆盖足以代表主要机制的少量配置。

如果某一 mechanism 在 Experiment 1 中已经确认只对其中一个模型成立，则 Experiment 2 只在满足该前提的模型上验证其 hardware robustness，不为了补齐对称矩阵运行缺乏研究前提的配置。

## 3. Representative-point selection

Experiment 2 不根据 L40 结果重新选择 workload。Representative points 必须在执行 L40 正式测试前从已经验证的 A100 结果中冻结，并保存 point identifier 与 selection reason。

正式集合至少覆盖以下四类 point。

### 3.1 Control point

该配置下 cache/state pressure 较低，I/O 与 scheduler pressure 均不明显。

该点用于建立不同硬件平台的基础行为，并判断优化是否在不存在目标 bottleneck 时产生固定 overhead 或 regression。

### 3.2 Cache / hierarchy point

该配置在 A100 上表现出明确的 GPU reusable-state capacity pressure，并且 hierarchical cache 已经能够减少 eviction-related recomputation 或扩大有效 reusable working set。

该点用于验证 hierarchical cache 的价值是否跨硬件稳定。

### 3.3 I/O-sensitive point

该配置在 A100 上存在明确的 CPU-GPU state movement 与 non-overlapped I/O stall，并且 I/O optimization 已经产生可解释的 mechanism-level improvement。

该点用于验证 I/O bottleneck 与对应 optimization 在 L40 上是否仍然存在。

### 3.4 Scheduler-sensitive point

该配置在 A100 上存在明确的 delay-hit、queueing、loading-bound batch 或 stall exposure，并且 scheduler optimization 已经产生对应 mechanism-level improvement。

该点用于验证 scheduler conclusion 是否依赖 A100 平台。

必要时额外保留一个 representative operating-boundary point，用于判断某项 bottleneck 开始出现或某项 optimization 开始产生净收益的位置是否因硬件变化而移动。

## 4. 两种硬件比较语义

本实验同时保留 same-workload comparison 与 matched-pressure comparison。两类结果分开组织和解释，不能混合为同一种硬件结论。

### 4.1 Same-workload comparison

A100 和 L40 执行相同的 logical request trace、input/output target distribution、reuse/locality structure、arrival schedule 和 system-configuration semantics。

该比较用于回答：同一个实际 workload 部署到不同 GPU 平台后，Strata 所描述的 bottleneck 和 optimization benefit 是否仍然出现。

如果 L40 因更大的 usable memory 或不同系统能力使某个 capacity bottleneck 明显减弱，这本身就是有效的 hardware-generalization result。

Same-workload comparison 反映 deployment-level behavior，不要求两块 GPU 处于相同的相对压力区域。

### 4.2 Matched-pressure comparison

当 same-workload comparison 使 A100 与 L40 处于明显不同的 cache/state-pressure 或 saturation region 时，实验补充少量 matched-pressure controls。

Matched-pressure control 使用预先定义并冻结的相对压力指标，使两个平台处于近似可比的 operating region。例如可以匹配 GPU reusable-state capacity pressure、baseline saturation fraction 或与目标 mechanism 直接相关的 pressure proxy。

Matched-pressure comparison 用于回答：当两个平台面临近似相同程度的系统压力时，对应 bottleneck 和 optimization mechanism 是否仍然成立。

Matched-pressure control 只用于必要的 attribution，不把全部 representative points 再扩展成新的完整参数扫描。

## 5. 模型覆盖

Qwen3.5-9B 与 Gemma 4 12B 都进入本实验，但每个模型只执行已经冻结且具有明确前置证据的 representative points。

每个模型遵循以下比较流程：

```text
validated A100 reference point
        ↓
same logical workload on L40
        ↓
matched-pressure control when necessary
        ↓
compare bottleneck and normalized optimization effect
```

已有 A100 run 只有在 checkpoint revision、runtime semantics、trace、measurement rule 和 validity requirements 与当前冻结配置完全一致时才直接复用。任一关键条件发生变化时重新执行 matched A100 run。

## 6. System configuration 设计

每个 representative point 只比较与当前研究 mechanism 直接相关的最小配置集合，不要求所有 point 都运行 End-to-End Serving 中的完整五配置矩阵。

### 6.1 Cache / hierarchy point

比较 GPU-only Baseline 与 validated Hierarchical Cache。

两种配置保持相同的 workload、scheduler semantics 和非目标 I/O policy。

### 6.2 I/O-sensitive point

比较 reference I/O path 与 validated I/O Optimization。

两种配置保持相同的 hierarchy semantics、logical restored state 与 scheduler semantics。

### 6.3 Scheduler-sensitive point

比较 reference scheduler 与 validated Scheduler Optimization。

两种配置保持相同的 cache/hierarchy semantics 与 I/O path。

### 6.4 Control point

Control point 至少运行 Baseline。根据前置实验结论，可增加对应 optimized configuration，用于检查 fixed overhead 或 hardware-specific regression。

这种最小配置集合使 hardware variable 与目标 mechanism 保持清晰对应，也避免 Experiment 2 重复 Experiment 3 的完整 end-to-end cross-product。

## 7. 控制变量

严格 paired hardware comparison 中保持以下逻辑条件一致：

- model identifier 与 revision；
- numerical precision 与 cache dtype；
- logical request trace；
- input/output target distribution；
- prefix-reuse structure；
- cache-locality / revisit structure；
- request-arrival schedule；
- generation settings；
- target mechanism semantics；
- cache initial-state protocol；
- measurement boundary；
- warm-up rule；
- repetition rule；
- result-processing rule。

硬件相关参数不强行保持相同数值。

GPU memory budget 同时保留 operational 与 normalized 两种口径。Same-workload comparison 保留相同 logical configuration；matched-pressure control 根据预先定义的 relative-pressure rule 调整必要的 workload 或 capacity condition。

每个 run 记录 GPU form factor、usable GPU memory、CPU-GPU interconnect / PCIe topology、CPU model、NUMA placement、host-memory policy、driver、CUDA/runtime 与实际 transfer path。

如果 A100 与 L40 所在节点的 CPU、PCIe topology 或 host-memory subsystem 不同，则结果称为 platform-level hardware comparison，不把全部差异单独归因于 GPU silicon。

## 8. 核心测量指标

### 8.1 Bottleneck-level metrics

根据 representative point 的 mechanism 类型记录：

- GPU / CPU serving-state residency；
- reusable-state eviction；
- recomputation；
- CPU-GPU data movement；
- non-overlapped I/O stall；
- cache resolve / restore behavior；
- queueing；
- scheduler stall / GPU idle behavior。

这些指标用于判断 bottleneck 是否仍然存在，以及硬件变化是否导致 bottleneck hierarchy 发生迁移。

### 8.2 Serving-level metrics

统一记录：

- request throughput；
- token throughput；
- P50 / P90 / P99 TTFT；
- request completion time；
- GPU utilization。

Serving-level metrics 用于判断 mechanism-level change 是否真正传导到最终 serving performance。

## 9. Normalized hardware comparison

每个 optimization 首先在同一模型、同一硬件内部计算相对于自身 baseline 的 normalized effect。

对于越高越好的指标，使用：

```text
relative_gain_h = (optimized_h - baseline_h) / baseline_h
```

其中 `h` 表示 A100 或 L40。

对于 latency、stall、recomputation 等越低越好的指标，使用统一 reduction convention，并在 processed metadata 中记录方向定义。

跨硬件主要比较 A100 与 L40 的 normalized effect、mechanism observables 和 operating-region change，而不是直接用 absolute throughput 差异判定 generalization success 或 failure。

所有 normalized results 必须保留并链接到对应 absolute baseline 与 optimized measurements。

## 10. 实验执行流程

每个 representative point 首先确认其 A100 reference result 已通过当前 validity gate，并保留明确的 mechanism evidence。

随后在 L40 上使用相同 logical trace 与 configuration semantics 执行 same-workload comparison。

当两个平台明显落入不同的 relative-pressure 或 saturation region 时，根据预先冻结的 matched-pressure rule 补充 control run。

每个正式 configuration 完成统一 warm-up 后进行多次独立重复测量。Paired runs 尽量使用相同 trace identifiers，并交替或随机化执行顺序。

每次 run 保存 experiment ID、representative-point ID、comparison type、model、hardware/platform metadata、system configuration、trace identifier、pressure summary、repetition index、capability status 与 validity status。

## 11. Validity conditions

进入主结果的 hardware paired comparison 必须满足：

- 两个平台使用相同 checkpoint revision；
- target mechanism semantics 一致或已验证语义等价；
- logical workload trace 一致；
- measurement rule 一致；
- runtime capability gate 在对应平台通过；
- hierarchy comparison 的 restored-state coverage 一致；
- I/O comparison 恢复相同 logical state；
- scheduler comparison 使用相同目标机制定义；
- 未发生未记录的 runtime fallback；
- same-workload 与 matched-pressure run 使用不同 comparison identifiers；
- hardware-specific unsupported path 被明确标记；
- measurement window 完整且 instrumentation 有效。

如果某项 mechanism 只能在其中一个平台正确实现，则该 pair 的 run status 标记为 `unsupported` 或 `partial`，最终 robustness conclusion 使用 `inconclusive` 或 capability-boundary 描述，不能把实现缺失解释为 optimization failure。

## 12. 结果分析逻辑

### 12.1 Bottleneck 与 optimization direction 均稳定

如果同类 bottleneck 在 A100 和 L40 上都存在，并且 optimization 在两个平台上都带来与目标 mechanism 一致的 improvement 和 serving gain，则该结论记为 `stable`。

### 12.2 Same-workload 下 L40 bottleneck 减弱，matched-pressure 下重新出现

如果相同 workload 在 L40 上不再形成明显 pressure，但 matched-pressure control 下对应 bottleneck 与 optimization effect 重新出现，则说明 mechanism 本身仍然成立，但 hardware resource change 移动了其 operating boundary。

该结果记为 `boundary_case`，并明确报告 boundary shift 的方向。

### 12.3 Matched-pressure 后收益仍明显不同

如果控制相对 pressure 后，某项 optimization 在两个平台上的 normalized effect 仍然明显不同，则说明该 mechanism 存在 hardware sensitivity。

该结果记为 `hardware_sensitive`，并结合 transfer stall、compute interference、queueing 与 utilization 判断差异主要出现在哪个系统环节。

### 12.4 Bottleneck hierarchy 发生迁移

如果 A100 主要受到 capacity 或 I/O 限制，而 L40 上对应问题减弱后暴露出 compute 或 scheduling bottleneck，则结果解释为 hardware change 导致 system bottleneck hierarchy 迁移。

这种结果不等价于原有 bottleneck 被否定。需要同时报告原 bottleneck 的减弱程度和新的限制因素。

### 12.5 Optimization direction 发生稳定反转

如果某项 optimization 在 A100 上产生正收益，但在 L40 上产生稳定且可重复的 regression，则该机制不具有稳定的 cross-hardware robustness。

该结果归入 `hardware_sensitive` 或 `boundary_case`，并保留负结果，不通过重新选择 workload 将其删除。

### 12.6 证据不足

如果 capability、matched-pressure quality、measurement precision 或 platform comparability 不足以支持方向性判断，则结论记为 `inconclusive`。

## 13. 结果组织

Experiment 2 最终形成一张 cross-hardware mechanism matrix：

| Model | Mechanism | Comparison | A100 bottleneck | L40 bottleneck | A100 normalized effect | L40 normalized effect | Conclusion |
|---|---|---|---|---|---:|---:|---|
| Qwen3.5-9B | Hierarchical Cache | same-workload / matched-pressure | ... | ... | ... | ... | ... |
| Qwen3.5-9B | I/O Optimization | same-workload / matched-pressure | ... | ... | ... | ... | ... |
| Qwen3.5-9B | Scheduler | same-workload / matched-pressure | ... | ... | ... | ... | ... |
| Gemma 4 12B | Hierarchical Cache | same-workload / matched-pressure | ... | ... | ... | ... | ... |
| Gemma 4 12B | I/O Optimization | same-workload / matched-pressure | ... | ... | ... | ... | ... |
| Gemma 4 12B | Scheduler | same-workload / matched-pressure | ... | ... | ... | ... | ... |

主结果同时保留 absolute measurements、normalized effects、pressure summary、platform metadata 与 validity status。

结果摘要使用 `stable`、`weakened`、`hardware_sensitive`、`boundary_case` 或 `inconclusive` 等统一类别，并附带必要的 boundary-shift 或 capability-boundary说明。

## 14. 实验边界

Experiment 2 研究的是代表性 mechanism conclusion 的硬件稳定性，不重新比较两个 GPU 的整体性能优劣。

本实验不执行完整 `2 models × 2 GPUs × all workloads × all configurations` 参数空间。完整的少量 `2 × 2` end-to-end cross validation 由 Experiment 3 完成。

本实验的核心判断是区分两种情况：硬件资源变化只是移动了 bottleneck 出现的位置，还是对应 mechanism 本身在不同硬件平台上失去稳定性。只有把 same-workload deployment behavior 与 matched-pressure mechanism robustness 分开，跨硬件结论才具有可解释性。
