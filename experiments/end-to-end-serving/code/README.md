# Code

本目录用于存放 “End-to-End Serving” 实验实现。

所有实现必须遵循 [`../docs/00-common-conventions.md`](../docs/00-common-conventions.md) 和仓库根目录 [`../../../docs/TECHNICAL_BASELINE.md`](../../../docs/TECHNICAL_BASELINE.md)。

## Responsibilities

代码需要覆盖以下职责：

- deterministic request trace 生成；
- long-context shared-prefix workload 构造；
- short-context low-reuse workload 构造；
- mixed workload 构造与 request-class tagging；
- Baseline、Hierarchical Cache、I/O Optimization、Scheduler Optimization 与 Full Configuration 的统一运行入口；
- cold-start、clean initial state 与 steady-state cache state 建立；
- offered-load scaling 与 saturation behavior 采集；
- throughput、TTFT、request completion time 与 GPU utilization 采集；
- request-class-level latency / throughput 采集；
- cache hit、eviction、recomputation、CPU-GPU traffic、I/O stall、queueing 与 batch behavior 等辅助指标采集；
- runtime capability / fallback validation；
- raw results 到 processed results 的 deterministic processing；
- figure/table generation。

## Experiment 1 requirements

Experiment 1 的实现必须能够：

1. 生成多个 shared-context groups，而不是只使用单一热点 prefix；
2. 控制 context-length point，并保持其余请求分布尽量一致；
3. 生成可重复的 revisit / reuse 行为；
4. 对同一 trace 执行五种 system configurations；
5. 区分 cold-start 与 steady-state measurement；
6. 对每个 workload point 执行多个 offered-load conditions；
7. 检测持续 queue accumulation、throughput plateau 与 latency amplification；
8. 保存每次 run 的完整 metadata 与 validity status。

## Experiment 2 requirements

Experiment 2 的实现必须能够：

1. 生成以独立 short-context requests 为主的 workload，并避免人为构造长共享 prefix；
2. 控制多个 short-context input-length profiles 与 output-length profiles；
3. 记录 actual reusable-prefix overlap，确认主 regression workload 没有意外形成明显 long-context reuse；
4. 在统一 clean initial state 下对同一 trace 执行五种 system configurations；
5. 对每个 short-context profile 执行多个 offered-load conditions；
6. 同时记录 request throughput、token throughput、P50/P90/P99 TTFT 与 request completion time；
7. 保留 scheduler queueing、CPU-tier activity、CPU-GPU data movement 与 GPU idle/stall 等 regression attribution 指标；
8. 输出 Full Configuration 相对 Baseline 的 absolute measurements 与 relative deltas。

## Experiment 3 requirements

Experiment 3 的实现必须能够：

1. 在同一 request trace 中混合 long-context reuse requests 与 independent short-context requests；
2. 为每条请求保存 request class、context/prefix identifier、output-length profile 与 arrival timestamp；
3. 构造 balanced representative workload，并在该 workload 上完成五种 system configurations 的完整 load scaling；
4. 分别生成 long-context dominant、balanced 与 short-context dominant compositions，同时保持其他主要 workload 属性可比较；
5. 在固定 composition 下生成 high、moderate 与 low locality traces，并保持 long-context request count、shared-context groups 与 length distributions 一致；
6. 构造 relatively homogeneous 与 heterogeneous output-length controls，避免 output length 与 request class 完全绑定；
7. 对 overall 和 long-context / short-context class 分别统计 throughput、P50/P90/P99 TTFT 与 request completion time；
8. 记录 cache reuse、recomputation、restore activity、queueing、batch composition 与 GPU idle/stall，用于解释 cross-class interference；
9. 验证 actual request-class ratio、reuse-distance profile 和 output-length distribution 与目标 workload 一致；
10. 将 primary matrix 与 composition/locality/output-length targeted robustness checks 分开处理和绘图，不生成完整笛卡尔积。

## Runtime validation

正式实验入口不得只检查模型是否能够启动。

至少需要验证：

- 实际运行配置与目标 system configuration 一致；
- hierarchy / I/O / scheduler mechanism 没有发生未记录的 fallback；
- paired runs 使用相同 GPU cache budget、generation settings 与 request trace；
- configured arrival schedule 与实际 trace injection 一致；
- cold-start 或 clean initial state 能够被重复建立；
- Experiment 1 的 steady-state reuse 能通过 observable cache/runtime behavior 确认；
- Experiment 2 的 actual reusable-prefix overlap 能够被记录并用于 workload validity check；
- Experiment 3 的 request-class ratio、reuse-distance profile 与 output-length distribution 能够被记录并用于 workload validity check；
- instrumentation failure 不会静默生成缺失或错误指标。

Validation result 必须写入 run metadata。

## Trace metadata

每条正式 trace 至少保存：

- trace identifier；
- seed / config hash；
- request count；
- request identifier；
- request class；
- context / prefix identifier；
- input token length；
- shared-prefix token length；
- output token target or realized length；
- arrival timestamp；
- reuse / revisit metadata；
- reuse-distance summary；
- workload class；
- context-length、short-context-profile 或 mixed-workload profile；
- composition profile when applicable；
- locality profile when applicable；
- output-length profile when applicable；
- offered-load condition。

## Run metadata

每次 run 至少保存：

- experiment ID；
- system configuration；
- model identifier 与 revision；
- serving runtime version / commit；
- hardware、driver、CUDA/runtime；
- precision 与 cache dtype；
- cache / offload backend and policy；
- GPU / CPU cache budget；
- scheduler policy；
- trace identifier；
- context-length、short-context-profile 或 mixed-workload profile；
- composition / locality / output-length profile when applicable；
- offered-load condition；
- achieved request rate；
- cache initial state；
- observed reusable-prefix overlap when applicable；
- actual request-class ratio when applicable；
- actual reuse-distance summary when applicable；
- repetition index；
- runtime capability status；
- validity status 与 invalid reason。

## Processing rules

- Raw measurements 不被 processing scripts 修改或覆盖。
- Invalid / partial / unsupported runs 不删除。
- 主 aggregation 只包含满足当前实验 validity requirements 的 runs。
- P50 / P90 / P99 等统计量从 per-request raw records 计算，不手工录入。
- Experiment 3 必须同时生成 overall 与 request-class-level aggregation。
- Relative gain / regression 必须保留对应 absolute measurement。
- Saturation point 由统一规则从观测数据判定，不能为不同系统配置手工选择有利阈值。
- Primary matrix 与 targeted robustness checks 使用明确的 experiment/workload identifiers 分开处理。
- Figure/table 只从 versioned processed data 生成。

## Suggested structure

后续实现可以按以下职责拆分：

```text
code/
├── configs/
├── workloads/
├── validation/
├── runners/
├── profiling/
├── analysis/
└── README.md
```

实际目录以实现规模为准，不为了形式预建无内容目录。
