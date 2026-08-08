# Code

本目录用于存放 “End-to-End Serving” 实验实现。

所有实现必须遵循 [`../docs/00-common-conventions.md`](../docs/00-common-conventions.md) 和仓库根目录 [`../../../docs/TECHNICAL_BASELINE.md`](../../../docs/TECHNICAL_BASELINE.md)。

## Responsibilities

代码需要覆盖以下职责：

- deterministic request trace 生成；
- long-context shared-prefix workload 构造；
- 后续 short-context 与 mixed workload 构造；
- Baseline、Hierarchical Cache、I/O Optimization、Scheduler Optimization 与 Full Configuration 的统一运行入口；
- cold-start 与 steady-state cache state 建立；
- offered-load scaling 与 saturation behavior 采集；
- throughput、TTFT、request completion time 与 GPU utilization 采集；
- cache hit、eviction、recomputation、CPU-GPU traffic、I/O stall 与 queueing 等辅助指标采集；
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

## Runtime validation

正式实验入口不得只检查模型是否能够启动。

至少需要验证：

- 实际运行配置与目标 system configuration 一致；
- hierarchy / I/O / scheduler mechanism 没有发生未记录的 fallback；
- paired runs 使用相同 GPU cache budget、generation settings 与 request trace；
- configured arrival schedule 与实际 trace injection 一致；
- cold-start 状态能够被重复建立；
- steady-state reuse 能通过 observable cache/runtime behavior 确认；
- instrumentation failure 不会静默生成缺失或错误指标。

Validation result 必须写入 run metadata。

## Trace metadata

每条正式 trace 至少保存：

- trace identifier；
- seed / config hash；
- request count；
- request identifier；
- context / prefix identifier；
- input token length；
- shared-prefix token length；
- output token target or realized length；
- arrival timestamp；
- reuse / revisit metadata；
- reuse-distance summary；
- workload class；
- context-length point；
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
- context-length point；
- offered-load condition；
- achieved request rate；
- cache initial state；
- repetition index；
- runtime capability status；
- validity status 与 invalid reason。

## Processing rules

- Raw measurements 不被 processing scripts 修改或覆盖。
- Invalid / partial / unsupported runs 不删除。
- 主 aggregation 只包含满足当前实验 validity requirements 的 runs。
- P50 / P90 / P99 等统计量从 per-request raw records 计算，不手工录入。
- Relative gain 必须保留对应 absolute measurement。
- Saturation point 由统一规则从观测数据判定，不能为不同系统配置手工选择有利阈值。
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
