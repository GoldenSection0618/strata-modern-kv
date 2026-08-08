# Experiment 3: End-to-End Generalization

## 1. 实验目标

本实验用于验证 Experiments 1–2 得到的跨模型与跨硬件机制结论，是否能够最终转化为稳定的 serving-level 收益。

实验覆盖以下四种 model × hardware 组合：

- Qwen3.5-9B + A100 40GB；
- Qwen3.5-9B + L40 48GB；
- Gemma 4 12B + A100 40GB；
- Gemma 4 12B + L40 48GB。

本实验重点回答三个问题：

1. 一个语义一致、预先冻结的 common Full Configuration 相对 Baseline 的端到端收益，是否在四种组合上保持同一方向；
2. 前两个实验观察到的 cache/state、I/O 和 scheduler 机制差异，是否能够解释最终 throughput 与 latency 的变化；
3. Strata 类优化是否存在明确的 model × hardware applicability boundary。

本实验不重新执行单机制的大规模参数扫描。前置实验已经负责 mechanism attribution，本实验只负责最终系统级泛化验证。

所有正式比较遵循 [`00-common-conventions.md`](00-common-conventions.md) 与仓库根目录 [`../../../docs/TECHNICAL_BASELINE.md`](../../../docs/TECHNICAL_BASELINE.md)。

## 2. 实验总体设计

实验采用完整的 `2 models × 2 GPUs` 交叉矩阵，但只使用少量预先冻结的 representative workloads。

正式主比较只使用：

1. **Baseline**；
2. **Common Full Configuration (`common_full`)**。

`common_full` 在正式 optimized results 生成前冻结。它必须在四种 model × hardware 组合上启用相同的 mechanism set，并且每个 mechanism 在四种组合上都通过对应 capability gate，或具有经过验证的语义等价实现。

某一组合如果无法支持冻结的 `common_full`，则该组合对 full-system cross-product 标记为 `unsupported`。实验不会静默删除一个 mechanism 后仍把该组合称为同一个 Full Configuration。

如果需要展示每个组合自身能够运行的最佳系统配置，可以额外报告 `best_validated_configuration`。该结果用于 deployment-oriented 补充分析，不进入 `common_full` 的跨组合 robustness conclusion。

本实验不预设 `common_full` 必须在所有组合上取得正收益。稳定正收益、收益消失、throughput-latency trade-off 与 regression 都作为有效结果保留。

当主矩阵出现明显异常结果时，再执行少量 targeted attribution runs。Attribution run 可以使用：

- Hierarchical Cache；
- Hierarchical Cache + I/O Optimization；
- Hierarchical Cache + Scheduler Optimization。

这些中间配置不进入所有 workload 的完整主矩阵，只用于解释 Baseline 与 `common_full` 之间无法由已有机制证据直接解释的差异。

## 3. Representative workload 设计

Experiment 3 不重新设计与前置实验无关的新 workload。正式 workload 从前五组实验以及本组 Experiments 1–2 中已经验证的场景冻结。

主实验保留三类 workload。

### 3.1 Long-context reuse

该 workload 包含明确的 long-context shared-prefix reuse，并形成可观测的 reusable-state pressure。

该场景用于验证 hierarchical cache、state restore、I/O 与 scheduler mechanisms 组合后，是否能够在最主要目标场景下跨模型、跨硬件保持端到端收益。

主结果同时检查 reusable-state realization、recomputation、I/O / queueing exposure、throughput 与 TTFT，保证最终性能变化能够与前置机制结论建立一致证据链。

### 3.2 Short-context control

该 workload 主要由独立 short-context requests 构成，不主动制造明显的 long shared-prefix reuse。

该场景用于验证 `common_full` 在非主要目标 workload 上是否引入固定开销、无效 CPU-tier activity、额外 scheduling overhead 或其他资源竞争。

该场景重点检查 throughput、P50/P90/P99 TTFT 与 request completion time 是否出现具有实际意义的 regression。

Material-regression / equivalence rule 使用 [`00-common-conventions.md`](00-common-conventions.md) 中规定的预先冻结口径。

### 3.3 Mixed workload

该 workload 同时包含 long-context reuse requests 与 ordinary short-context requests，并保留代表性的 output-length heterogeneity 与 cache-locality variation。

该场景用于验证完整系统在更接近真实 serving 的异构 workload 中能否保持整体收益，同时避免 aggregate metrics 掩盖 cross-class interference。

结果必须同时报告 overall performance 和 long-context / short-context request-class-level performance。

如果 overall throughput 提升，但任一主要 request class 的 tail latency 稳定恶化，则结果解释为 cross-class trade-off，而不是无条件的 end-to-end improvement。

## 4. Workload 冻结原则

三类 representative workload 的 logical trace、reuse structure、locality profile、output target distribution 与 selection rule 必须在正式 generalization result 生成前冻结。

四种 model × hardware 组合使用相同 workload-generation rule 和相同逻辑请求结构。

跨模型时，请求按照各自 tokenizer 分别 materialize，并记录实际 input/output token counts 与 realized output length。

如果模型之间的 actual work volume 存在明显差异，则使用 matched-work 或 normalized-work control 作为补充解释，不能把 tokenizer 或模型计算量差异直接解释成系统泛化差异。

跨硬件主结果保持 same-workload semantics。同一个模型在 A100 与 L40 上使用同一份 materialized logical trace 和相同 arrival schedule。

如果 A100 与 L40 因 memory capacity 或 serving capacity 差异落入明显不同的 operating region，则复用 Experiment 2 的 matched-pressure methodology 增加少量解释性 control。Same-workload 与 matched-pressure 结果分开报告。

## 5. Load 设计

每类 representative workload 对每个模型建立三个 primary operating points：

- Low；
- Medium；
- High / near-saturation。

每个模型的三个 point 使用该模型在 A100 Baseline 上的 calibration 结果确定，并在任何 generalization optimized result 生成前冻结 exact arrival/load schedule。

Low point 用于观察 `common_full` 是否引入 fixed overhead。

Medium point 用于观察 cache、I/O 与 scheduler mechanisms 开始发挥作用后的实际收益。

High point 用于观察 serving capacity、tail latency 与 saturation behavior。

同一个模型在 A100 与 L40 上使用相同的 frozen Low / Medium / High arrival schedules，从而使 primary hardware comparison 保持 same-workload 语义。

Qwen3.5 与 Gemma 4 的 Low / Medium / High 不要求使用相同 absolute requests/s。跨模型比较主要使用 normalized Full-vs-Baseline effect、实际 offered-work metadata 和对应 operating-region evidence，而不是把相同 requests/s 作为必要条件。

如果 frozen schedule 使 L40 相对 A100 落入明显不同的 pressure / saturation region，则补充少量 `matched_pressure` controls。Matched-pressure result 只用于解释 boundary shift，不进入 primary matrix。

## 6. 主实验矩阵

主实验矩阵为：

```text
2 Models
× 2 GPUs
× 3 Representative Workloads
× 3 Frozen Per-Model Load Points
× 2 Main System Configurations
```

对应：

```text
Qwen3.5-9B / Gemma 4 12B
            ×
A100 40GB / L40 48GB
            ×
Long-context / Short-context / Mixed
            ×
Low / Medium / High
            ×
Baseline / common_full
```

该矩阵用于形成最终 model × hardware generalization conclusion。

`common_full` mechanism set 在整个矩阵中保持一致。不能为某个 model × hardware cell 单独改变 feature set 后继续进入同一 primary matrix。

实验不继续加入 context length、page size、reuse ratio、scheduler threshold 等大规模 sweep。相关变量已经由前置 mechanism experiments 负责验证。

## 7. Targeted attribution runs

主矩阵只有在出现以下情况时才触发 targeted attribution runs：

- `common_full` 在某一组合上的收益明显低于其他组合；
- `common_full` 出现稳定 regression；
- throughput 与 tail latency 出现明显 trade-off；
- mixed workload 出现 cross-class interference；
- Experiments 1–2 的 mechanism-level prediction 与最终 serving result 不一致。

Attribution run 只比较定位问题所需的最小配置集合。

如果怀疑 I/O Optimization 的增量价值在某个平台上消失，则比较：

```text
Hierarchical Cache
vs
Hierarchical Cache + I/O Optimization
```

如果怀疑 Scheduler Optimization 引入 mixed-workload interference，则比较：

```text
Hierarchical Cache
vs
Hierarchical Cache + Scheduler Optimization
```

如果 hierarchy 本身已经无法产生 avoided recomputation 或有效 CPU-tier reuse，则不继续把后续 I/O / scheduler configuration 的结果解释为同一条完整 Strata mechanism chain。

如果某个 primary cell 因 `common_full` capability 缺失而为 `unsupported`，targeted attribution 可以用于定位 capability boundary，但不能生成一个删减 feature set 的伪 `common_full` 替代该 cell。

## 8. 控制变量

同一个 paired comparison 中保持以下逻辑条件一致：

- exact model identifier 与 revision；
- serving runtime revision；
- numerical precision；
- cache dtype；
- workload-generation rule；
- logical request trace；
- input/output target distribution；
- reuse / locality pattern；
- primary arrival/load schedule；
- Baseline semantics；
- frozen `common_full` mechanism set 与 semantics；
- warm-up protocol；
- measurement window；
- repetition rule；
- processing rule。

不同 GPU 的硬件相关参数完整记录，但不人为要求数值完全相同。

每次 run 至少记录：

- GPU model 与 usable HBM；
- CPU model；
- CPU-GPU topology；
- NUMA placement；
- host-memory policy；
- driver；
- CUDA/runtime；
- configured GPU reusable-state budget；
- configured CPU-tier budget；
- resolved GPU/CPU allocation by observable state group；
- runtime capability status；
- `common_full` feature-set identifier。

如果 A100 与 L40 所在节点的 CPU、PCIe topology 或其他 host platform 条件不同，则最终结论使用 platform-level generalization 表述，不把全部差异归因于 GPU silicon。

## 9. 核心指标

### 9.1 End-to-End metrics

所有四种组合统一记录：

- request throughput；
- token throughput；
- P50 / P90 / P99 TTFT；
- request completion time；
- GPU utilization。

Mixed workload 进一步按 request class 分别报告：

- request throughput；
- token throughput；
- P50 / P90 / P99 TTFT；
- request completion time。

### 9.2 Mechanism explanation metrics

为了把最终结果与前置实验连接起来，同时保留：

- reusable-state hit / reuse realization；
- eviction；
- recomputation；
- CPU-GPU traffic；
- non-overlapped I/O stall；
- queueing；
- scheduler stall / idle behavior；
- GPU idle；
- representative batch behavior when available。

这些指标不构成本实验新的独立研究问题，而用于验证以下证据链是否一致：

```text
mechanism change
        ↓
resource behavior change
        ↓
end-to-end performance change
```

## 10. Normalized gain

每个 `model × hardware × workload × load` point 都同时保存 absolute measurement 和 `common_full` 相对自身 Baseline 的 normalized effect。

Throughput gain 定义为：

```text
throughput_gain = (throughput_common_full - throughput_baseline) / throughput_baseline
```

Latency reduction 使用：

```text
latency_reduction = (latency_baseline - latency_common_full) / latency_baseline
```

跨组合 robustness 主要比较 normalized gain 的方向与幅度，而不是直接比较 Qwen、Gemma、A100 与 L40 的 absolute throughput。

所有 normalized result 必须能够回溯到其对应 absolute baseline、`common_full` measurement 与 uncertainty summary。

## 11. 实验执行流程

首先从前置实验冻结三类 representative workloads、每个模型的 A100 Baseline load calibration、primary load schedules、point identifiers 与 `common_full` mechanism set。

随后确认四种 model × hardware 组合均通过 `common_full` 所需的 runtime/state capability gate。

某一组合无法支持 `common_full` 时，在正式矩阵中记录 `unsupported`。不能在执行阶段动态改变 feature set。

每种组合先执行 Baseline validation，确认 workload materialization、target primary load point 与 instrumentation 有效。

正式 paired runs 在相同 logical trace 上执行 Baseline 与 `common_full`。同一个模型的 A100/L40 paired runs 使用相同 frozen arrival schedule。

每个正式 point 在统一 warm-up 后执行多次独立重复测量，并保存完整 raw measurement 与 metadata。

不同配置的执行顺序尽量交替或随机化，降低节点长期状态变化造成的系统偏差。

已有 A100 results 只有在 model/runtime revision、trace、system semantics、resolved cache/state contract、measurement rule 与当前冻结配置完全一致时才直接复用。否则补跑 matched A100 point。

主矩阵执行完成后，再依据预定义触发条件决定是否执行 targeted attribution runs 或 matched-pressure explanatory controls。

## 12. Validity conditions

进入最终 robustness matrix 的 run 必须满足：

- target model 与 exact checkpoint revision 正确；
- Baseline semantics 明确；
- `common_full` feature-set identifier 与冻结配置一致；
- `common_full` 所包含的全部 mechanisms 在当前 combination 上通过 capability gate；
- hybrid serving-state restore coverage 满足对应 full-hit claim；
- configured budget 与 resolved per-state-group allocation 已记录；
- workload trace 与目标 point 一致；
- actual token distribution 与 realized output work 已记录；
- cache initial state 与 measurement boundary 符合协议；
- primary offered-load schedule 与该模型冻结的 schedule 一致；
- paired system configurations 未发生未记录 fallback；
- instrumentation 正常；
- measurement window 完整；
- 未发生破坏比较条件的 OOM 或 runtime failure。

不满足运行有效性的 run 保留 raw result，并标记为 `partial` 或 `invalid`。如果 `common_full` 本身在该组合上缺少必要 capability，则该 matrix cell 标记为 `unsupported`。

Capability 缺失不能解释为 optimization failure，也不能通过删减 mechanism 后继续生成同名 Full Configuration 结果。

## 13. 结果判断逻辑

### 情况 A：四种组合均获得稳定收益

如果 `common_full` 在主要 long-context 与 mixed workload 中均取得稳定正收益，同时 short-context 没有 material regression，则说明这套共同机制集合在当前两个模型和两个平台上具有较强的 model × hardware robustness。

该结果只支持 systems mechanism 跨当前两个模型和两个平台的 robustness，不扩张为对任意模型或任意 GPU 的普遍结论。

### 情况 B：收益方向一致，但幅度不同

如果四种组合的 normalized effect 方向一致，但 gain magnitude 明显不同，则结论为 mechanism set generalizes but benefit magnitude is model/hardware sensitive。

差异由 Experiments 1–2 的 state pressure、I/O、scheduler 与 hardware-pressure evidence 解释。

### 情况 C：某些组合收益消失，同时对应 bottleneck 减弱

如果某个组合上 `common_full` 几乎无额外收益，而前置实验同时显示其 cache/state、I/O 或 scheduling bottleneck 已明显减弱，则该结果属于 applicability boundary。

该结果解释为优化必要性下降，而不是 mechanism 被反证。

### 情况 D：Bottleneck 仍存在，但 `common_full` 无收益

如果前置实验明确显示目标 bottleneck 仍然存在，而 `common_full` 没有形成 end-to-end gain，则触发 targeted attribution。

重点检查 optimization overhead、mechanism interaction、GPU compute interference、transfer overlap、scheduler interaction 与 secondary bottleneck。

该结果表示单机制收益没有成功组合为完整系统收益。

### 情况 E：Throughput-latency trade-off

如果 `common_full` 提高 throughput，但 P99 TTFT 或 request completion time 稳定恶化，则结果标记为 `throughput_latency_tradeoff`，不能写成无条件 system improvement。

### 情况 F：Cross-class regression

如果 mixed workload 的 overall performance 改善，但 long-context 或 short-context request class 的 tail latency 出现具有实际意义的稳定恶化，则结果标记为 `cross_class_tradeoff`。

Aggregate performance 不替代 class-level result。

### 情况 G：共同配置无法覆盖全部组合

如果某一 model × hardware combination 无法实现冻结的 `common_full` mechanism set，则该 cell 标记为 `unsupported`，并将缺失 capability 作为 applicability boundary 单独报告。

该情况不允许用该组合自身的 `best_validated_configuration` 结果填补 `common_full` matrix cell。

## 14. 最终结果组织

Experiment 3 最终至少形成一张 model × hardware robustness matrix：

| Model | GPU / Platform | Workload | Load point | Baseline | common_full | Throughput gain | TTFT change | Mechanism evidence | Conclusion |
|---|---|---|---|---|---|---:|---:|---|---|
| Qwen3.5-9B | A100 | Long | ... | ... | ... | ... | ... | ... | ... |
| Qwen3.5-9B | L40 | Long | ... | ... | ... | ... | ... | ... | ... |
| Gemma 4 12B | A100 | Long | ... | ... | ... | ... | ... | ... | ... |
| Gemma 4 12B | L40 | Long | ... | ... | ... | ... | ... | ... | ... |

Short-context 与 mixed workload 使用同一 schema 继续组织。Mixed workload 额外生成 request-class-level summary。

Primary matrix 只包含 same-workload load points。Matched-pressure controls 和 `best_validated_configuration` 使用独立表格或 identifier，不进入 primary aggregation。

最终 conclusion 使用受限类别：

- `stable_generalization`；
- `model_sensitive`；
- `hardware_sensitive`；
- `boundary_shifted`；
- `throughput_latency_tradeoff`；
- `cross_class_tradeoff`；
- `unsupported`；
- `inconclusive`。

## 15. 与 Experiments 1–2 的关系

三个实验形成明确的递进关系：

```text
Experiment 1
跨模型时，mechanism 是否仍然成立
        ↓
Experiment 2
跨硬件时，mechanism 是否仍然成立
        ↓
Experiment 3
同一共同 mechanism set 最终能否形成稳定的 serving gain
```

Experiment 3 不重新承担“为什么有效”的主要证明任务，而负责验证已经建立的 mechanism evidence 在完整 serving pipeline 中是否能够跨当前模型和硬件组合转化为稳定收益。

因此，本组最终形成 `mechanism robustness → hardware robustness → end-to-end robustness` 的完整证据链。
