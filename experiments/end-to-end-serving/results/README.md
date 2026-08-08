# Results

本目录用于存放 “End-to-End Serving” 的实验结果。

所有结果遵循 [`../docs/00-common-conventions.md`](../docs/00-common-conventions.md)，并保持以下可追溯关系：

```text
raw measurements
    ↓
processed / aggregated data
    ↓
figures and tables
```

## Raw results

Raw results 保存每次 run 的原始 measurement payload 与 metadata，不被处理脚本覆盖。

Metadata 至少包含：

- experiment ID；
- system configuration；
- model identifier 与 revision；
- hardware、driver、CUDA/runtime；
- serving runtime version / commit；
- precision 与 cache dtype；
- cache / offload backend and policy；
- scheduler policy；
- GPU reusable-cache budget；
- CPU-tier budget when applicable；
- workload class；
- trace identifier 与 trace type；
- context-length、short-context-profile 或 mixed-workload profile；
- request-class composition when applicable；
- locality profile when applicable；
- output-length profile when applicable；
- input/output token distribution；
- offered request rate；
- offered input/output token-work summary；
- achieved request throughput；
- achieved token throughput；
- cache initial state；
- preconditioning identifier / measurement boundary when applicable；
- observed reusable-prefix overlap when applicable；
- actual request-class ratio when applicable；
- actual reuse-distance summary when applicable；
- realized output-length summary；
- matched-work rule / deviation when applicable；
- regression/equivalence rule identifier when applicable；
- run timestamp 与 repetition index；
- runtime capability status；
- validity status 与 invalid reason。

Raw measurement payload 至少尽可能保留：

- per-request request class；
- per-request arrival time；
- per-request input token length；
- per-request output target；
- per-request realized output length；
- per-request TTFT；
- per-request completion time；
- completed request / token accounting；
- GPU utilization samples；
- queue length / queueing time when available；
- GPU / CPU cache hit volume；
- reusable-state eviction；
- recomputation；
- CPU-GPU transfer activity；
- non-overlapped I/O stall；
- batch composition when available；
- runtime errors、fallbacks 与 scheduler events。

## Processed results

Processed data 由脚本从 raw measurements deterministic 生成，并保留 run identifiers。

Processed data 至少支持：

- achieved request throughput；
- achieved token throughput；
- P50 / P90 / P99 TTFT；
- request completion-time distribution；
- GPU utilization summary；
- cold-start vs fixed-preconditioned steady-state comparison when applicable；
- clean-initial-state short-context comparison；
- overall 与 request-class-level mixed-workload aggregation；
- system-configuration relative gain / regression；
- uncertainty / interval summary；
- offered-load scaling curves；
- saturation-region identification；
- operational-sensitivity vs matched-work comparison；
- cache / recomputation / I/O / queueing / batch-behavior auxiliary analysis。

任何 filtering 都需要保留明确的 invalid reason。Raw data 不因进入不了主 aggregation 而删除。

## Experiment 1 outputs

Long-context Reuse Serving 至少形成：

1. 不同 context length 下 request throughput 与 token throughput 随 offered load 的变化；
2. 不同 context length 下 P50 / P90 / P99 TTFT 随 offered load 的变化；
3. request completion time 随 offered load 的变化；
4. GPU utilization 随 offered load 的变化；
5. cold-start 与 fixed-preconditioned steady-state performance 对比；
6. representative medium-load / high-load points 下五种 system configurations 的直接比较；
7. Hierarchical + I/O 与 Hierarchical + Scheduler 两个 parallel attribution branches 的对比；
8. 与关键性能差异对应的 cache reuse、recomputation、CPU-GPU traffic、I/O stall 与 queueing 辅助结果。

Experiment 1 的主结果按照 `context length × offered load × system configuration` 组织。

Cold-start 与 steady-state 必须使用明确的 trace / preconditioning identifiers，不能根据不同配置的运行表现动态选择 measurement 起点。

## Experiment 2 outputs

Short-context Serving Regression 至少形成：

1. 不同 short-context profile 下 P50 / P90 / P99 TTFT 随 offered load 的变化；
2. request throughput 与 token throughput 随 offered load 的变化；
3. request completion time 随 offered load 的变化；
4. GPU utilization 随 offered load 的变化；
5. actual reusable-prefix overlap / cache reuse validity summary；
6. Full Configuration 相对 Baseline 的 regression / equivalence summary；
7. 出现 regression 时对应的 Hierarchical、Hierarchical + I/O、Hierarchical + Scheduler attribution results；
8. scheduler queueing、CPU-tier activity、CPU-GPU data movement 与 GPU idle/stall 辅助结果。

Experiment 2 的主结果按照 `short-context profile × offered load × system configuration` 组织。

Regression summary 同时保存 absolute measurements、relative deltas、uncertainty、predeclared decision margin 与 rule identifier。

结论至少区分：

- `no_material_regression`；
- `material_regression`；
- `throughput_latency_tradeoff`；
- `inconclusive`。

如果实验精度不足以排除具有实际意义的 regression，则必须使用 `inconclusive`，不能仅因为差异不显著就写成 `no_material_regression`。

## Experiment 3 outputs

Mixed Workload Serving 采用 primary matrix 与 targeted robustness checks 两层结果结构。

### Primary matrix

Balanced composition、moderate locality 和 heterogeneous output length 下至少形成：

1. overall request throughput 与 token throughput 随 offered load 的变化；
2. overall P50 / P90 / P99 TTFT 随 offered load 的变化；
3. overall request completion time 随 offered load 的变化；
4. GPU utilization 随 offered load 的变化；
5. long-context requests 的 request/token throughput、P50 / P90 / P99 TTFT 与 completion time；
6. short-context requests 的 request/token throughput、P50 / P90 / P99 TTFT 与 completion time；
7. representative medium-load / high-load points 下五种 system configurations 的直接比较；
8. Hierarchical + I/O 与 Hierarchical + Scheduler parallel attribution comparison；
9. 对应的 cache reuse、recomputation、CPU-GPU traffic、I/O stall、queueing 与 batch-composition 辅助结果。

Primary matrix 按 `representative mixed workload × offered load × system configuration` 组织。

### Composition robustness

Composition robustness 分开保存两类结果。

#### Operational sensitivity

保持相同 request-arrival schedule，比较：

- long-context dominant；
- balanced；
- short-context dominant。

结果同时报告 offered token/work summary、overall performance 与两类 request 的 class-level performance。

这组结果表示 workload mix 真实变化后的 operational behavior。因为总 offered work 可以变化，不把它解释为 composition 的纯因果效应。

#### Matched-work control

对代表性中高负载 point 使用预先冻结的 matched-work rule，使 offered token/work 或 baseline load proxy 达到规定匹配容差。

结果用于判断总体工作压力近似可比后，不同 request-class mixture 是否仍然改变 cache、queueing、scheduler behavior 与 cross-class latency。

### Locality robustness

在 balanced composition 下至少比较：

- high locality；
- moderate locality；
- low locality。

结果同时报告 actual reuse-distance summary、cache reuse realization、restore / recomputation behavior 和端到端性能，确认 locality 变化没有通过 request-class ratio 或 length distribution 间接产生。

### Output-length robustness

Output-length robustness 分开保存：

- same-request-arrival operational sensitivity；
- representative matched-work control。

两类结果均比较 relatively homogeneous 与 heterogeneous output-length distributions，并报告 offered output work、overall / class-level tail latency、request/token throughput、queueing 和 batch behavior。

只有 matched-work control 仍然显示稳定退化时，才把额外 degradation 更有力地与 decode-duration heterogeneity / scheduler interaction 联系起来。

### Cross-class interference summary

Experiment 3 至少形成一张 request-class summary：

| Workload profile | Comparison type | System config | Offered-work summary | Overall throughput | Long-context P99 TTFT | Short-context P99 TTFT | Conclusion |
|---|---|---|---|---:|---:|---:|---|
| Representative | Primary | Baseline | ... | ... | ... | ... | ... |
| Representative | Primary | Full | ... | ... | ... | ... | ... |
| Long-dominant | Operational | Full | ... | ... | ... | ... | ... |
| Short-dominant | Operational | Full | ... | ... | ... | ... | ... |
| Long-dominant | Matched-work | Full | ... | ... | ... | ... | ... |

Aggregate performance 不能替代 class-level performance。若 overall throughput 提升但任一主要 request class 的 tail latency 出现稳定且具有实际意义的恶化，则必须报告 cross-class trade-off。

与 Experiments 1/2 的 cross-class interference 对照必须匹配 class-specific input/output work、total offered work、reuse condition 与 system load region。若没有足够匹配的历史 point，则使用补充 matched control run，不强行复用不等价结果。

## Result interpretation

结果摘要必须同时报告 request/token throughput 与 latency。不能只依据单一吞吐提升判断系统整体更优。

如果 Full Configuration 提高 throughput 但显著恶化 P99 TTFT 或 request completion time，则结果标记为 throughput-latency trade-off。

如果低负载无明显差异、中高负载开始出现收益，则将其解释为 serving-capacity / contention-region improvement。

如果 short-context workload 在低负载下已经出现稳定 latency overhead，则优先解释为 request-path fixed overhead，而不是高负载 queueing effect。

如果 short-context workload 只有中高负载出现 regression，则结合 scheduler queueing、GPU utilization、batch behavior 与 background activity 定位资源竞争来源。

如果 mixed workload 的 overall performance 改善但 short-context 或 long-context class-level P99 latency 明显恶化，则结果解释为 cross-class interference，而不是无条件系统提升。

如果 mixed-workload 收益只在 long-context dominant operational trace 中存在，则先检查 offered work 与 reuse proportion。Matched-work control 决定是否能够进一步支持 composition-related attribution。

如果 heterogeneous output length 在 operational comparison 中表现更差，则先检查 output-work 增量。只有 matched-work control 仍然显示额外退化时，才进一步讨论 decode-duration heterogeneity 的独立作用。

如果 short-context workload 出现明显正收益，则首先检查 actual prefix reuse 和一般 scheduler improvement，不能默认归因于 hierarchical caching。

如果 cache hit 与 avoided recomputation 明显，但端到端性能变化有限，则使用 data movement、I/O stall、queueing 与 GPU utilization 辅助解释后续瓶颈。

## Figures and tables

所有正式 figure/table 只从 processed data 生成，不手工录入最终数值。

正式报告中的任何图表必须能够追溯到：

```text
figure/table
    ↓
processed dataset + processing config/commit
    ↓
raw run identifiers
    ↓
run metadata + validity status
```

Relative metrics 必须保留 underlying absolute measurements 与 uncertainty。

Operational-sensitivity 与 matched-work results 不得在图表中使用相同标签混淆展示。

## Storage policy

大体积 profiler dump、模型权重与可重新生成的大型中间文件不提交到 Git。

需要长期保留的外部结果记录：

- external path or object identifier；
- checksum；
- generating run identifier；
- runtime / processing version；
- retention note when applicable。
