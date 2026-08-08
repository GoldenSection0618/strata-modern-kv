# Code

本目录用于存放 “End-to-End Serving” 实验实现。

所有实现必须遵循 [`../docs/00-common-conventions.md`](../docs/00-common-conventions.md) 和仓库根目录 [`../../../docs/TECHNICAL_BASELINE.md`](../../../docs/TECHNICAL_BASELINE.md)。

## Responsibilities

代码需要覆盖以下职责：

- deterministic request trace 生成；
- long-context shared-prefix workload 构造；
- short-context low-reuse workload 构造；
- mixed workload 构造与 request-class tagging；
- Baseline、Hierarchical Cache、Hierarchical Cache + I/O、Hierarchical Cache + Scheduler 与 Full Configuration 的统一运行入口；
- cold-start、clean initial state 与 fixed-preconditioning steady-state 建立；
- offered-load scaling 与 saturation behavior 采集；
- request/token throughput、TTFT、request completion time 与 GPU utilization 采集；
- request-class-level request/token throughput 与 latency 采集；
- cache hit、eviction、recomputation、CPU-GPU traffic、I/O stall、queueing 与 batch behavior 等辅助指标采集；
- runtime capability / fallback validation；
- regression/equivalence decision metadata 管理；
- operational-sensitivity 与 matched-work workload/control 生成；
- raw results 到 processed results 的 deterministic processing；
- figure/table generation。

## System-configuration contract

五种配置的运行语义必须显式实现并写入 metadata：

1. `baseline`：reference cache/I/O/scheduler path；
2. `hierarchical`：hierarchy + reference I/O + reference scheduler；
3. `hierarchical_io`：hierarchy + validated I/O optimization + reference scheduler；
4. `hierarchical_scheduler`：hierarchy + reference I/O + validated scheduler optimization；
5. `full`：hierarchy + validated I/O optimization + validated scheduler optimization。

配置 3 与配置 4 是 parallel attribution branches。Runner 不得通过共享隐式默认值使其中一个分支意外启用另一个 optimization。

所有 paired configurations 使用相同 GPU reusable-cache budget。所有启用 hierarchy 的配置使用相同 CPU-tier budget、host-memory policy 与 offload policy，除非当前 experiment 明确研究这些变量。

## Experiment 1 requirements

Experiment 1 的实现必须能够：

1. 生成多个 shared-context groups，而不是只使用单一热点 prefix；
2. 控制 context-length point，并保持其余请求分布尽量一致；
3. 生成可重复的 revisit / reuse 行为；
4. 对同一 trace 执行五种 system configurations；
5. 分离 cold-start run 与 steady-state run；
6. 使用固定 preconditioning trace 建立 steady-state，并在预定义 measurement boundary 开始正式测量；
7. 记录 measurement 开始时的 cache residency / occupancy；
8. 对每个 workload point 执行冻结的 offered-load grid；
9. 检测持续 queue accumulation、throughput plateau 与 latency amplification；
10. 保存每次 run 的完整 metadata 与 validity status。

Experiment 1 不允许根据不同配置的实时表现动态决定“何时进入 steady-state”。

## Experiment 2 requirements

Experiment 2 的实现必须能够：

1. 生成以独立 short-context requests 为主的 workload，并避免人为构造长共享 prefix；
2. 控制多个 short-context input-length profiles 与 output-length profiles；
3. 同时保存 output target 与 realized output length；
4. 记录 actual reusable-prefix overlap，确认主 regression workload 没有意外形成明显 long-context reuse；
5. 在统一 clean initial state 下对同一 trace 执行五种 system configurations；
6. 对每个 short-context profile 执行冻结的 offered-load grid；
7. 同时记录 request throughput、token throughput、P50/P90/P99 TTFT 与 request completion time；
8. 保留 scheduler queueing、CPU-tier activity、CPU-GPU data movement 与 GPU idle/stall 等 regression attribution 指标；
9. 从 versioned config 读取预先冻结的 regression/equivalence margin 与 analysis rule；
10. 输出 Full Configuration 相对 Baseline 的 absolute measurements、relative deltas 与 uncertainty。

如果实验精度不足以排除预定义的 material regression，analysis 必须输出 `inconclusive`，不能自动输出 `no_regression`。

## Experiment 3 requirements

Experiment 3 的实现必须能够：

1. 在同一 request trace 中混合 long-context reuse requests 与 independent short-context requests；
2. 为每条请求保存 request class、context/prefix identifier、input length、output target、realized output length 与 arrival timestamp；
3. 构造 balanced representative workload，并在该 workload 上完成五种 system configurations 的完整 load scaling；
4. 分别生成 long-context dominant、balanced 与 short-context dominant compositions；
5. 为 composition robustness 生成 same-request-arrival operational traces 和独立 matched-work controls；
6. 在固定 composition 下生成 high、moderate 与 low locality traces，并保持 request-class ratio 与 length distributions 一致；
7. 构造 relatively homogeneous 与 heterogeneous output-length workloads；
8. 为 output-length robustness 同时支持 operational sensitivity 与 matched-work control；
9. 对 overall 和 long-context / short-context class 分别统计 request throughput、token throughput、P50/P90/P99 TTFT 与 request completion time；
10. 记录 cache reuse、recomputation、restore activity、queueing、batch composition 与 GPU idle/stall，用于解释 cross-class interference；
11. 验证 actual request-class ratio、reuse-distance profile、output-length distribution 与 matched-work tolerance；
12. 将 primary matrix、operational robustness 与 matched-work controls 使用不同 identifiers 分开处理和绘图，不生成完整笛卡尔积。

## Runtime validation

正式实验入口不得只检查模型是否能够启动。

至少需要验证：

- 实际运行配置与目标 system configuration 一致；
- hierarchy / I/O / scheduler mechanism 没有发生未记录的 fallback；
- `hierarchical_io` 未意外启用 optimized scheduler；
- `hierarchical_scheduler` 未意外启用 optimized I/O path；
- paired runs 使用相同 GPU reusable-cache budget、generation settings 与 logical request trace；
- hierarchy runs 使用相同 CPU-tier budget 与 offload policy；
- configured arrival schedule 与实际 trace injection 一致；
- cold-start 或 clean initial state 能够被重复建立；
- Experiment 1 的 fixed preconditioning 与 measurement boundary 能够被重复建立；
- Experiment 2 的 actual reusable-prefix overlap 能够被记录并用于 workload validity check；
- Experiment 3 的 request-class ratio、reuse-distance profile、output-length distribution 与 work summary 能够被记录并用于 validity check；
- instrumentation failure 不会静默生成缺失或错误指标。

Validation result 必须写入 run metadata。

## Trace metadata

每条正式 trace 至少保存：

- trace identifier；
- trace type: primary / operational_sensitivity / matched_work_control；
- seed / config hash；
- request count；
- request identifier；
- request class；
- context / prefix identifier；
- input token length；
- shared-prefix token length；
- output token target；
- realized output token length after execution；
- arrival timestamp；
- reuse / revisit metadata；
- reuse-distance summary；
- workload class；
- context-length、short-context-profile 或 mixed-workload profile；
- composition profile when applicable；
- locality profile when applicable；
- output-length profile when applicable；
- offered request rate；
- offered input/output token-work summary；
- matched-work rule / target when applicable。

## Run metadata

每次 run 至少保存：

- experiment ID；
- system configuration；
- model identifier 与 revision；
- serving runtime version / commit；
- hardware、driver、CUDA/runtime；
- precision 与 cache dtype；
- cache / offload backend and policy；
- GPU reusable-cache budget；
- CPU-tier budget when applicable；
- scheduler policy；
- trace identifier；
- workload profile；
- composition / locality / output-length profile when applicable；
- offered request rate；
- offered token/work summary；
- achieved request throughput；
- achieved token throughput；
- cache initial state；
- preconditioning identifier / measurement boundary when applicable；
- observed reusable-prefix overlap when applicable；
- actual request-class ratio when applicable；
- actual reuse-distance summary when applicable；
- realized output-length summary；
- matched-work deviation when applicable；
- regression/equivalence rule identifier when applicable；
- repetition index；
- runtime capability status；
- validity status 与 invalid reason。

## Processing rules

- Raw measurements 不被 processing scripts 修改或覆盖。
- Invalid / partial / unsupported runs 不删除。
- 主 aggregation 只包含满足当前实验 validity requirements 的 runs。
- P50 / P90 / P99 等统计量从 per-request raw records 计算，不手工录入。
- Request throughput 与 token throughput 同时保留。
- Experiment 3 必须同时生成 overall 与 request-class-level aggregation。
- Relative gain / regression 必须保留对应 absolute measurement 与 uncertainty。
- Saturation point 由统一规则从观测数据判定，不能为不同系统配置手工选择有利阈值。
- Load grid、regression margin 与 matched-work rule 必须从 versioned configuration 读取，不能在查看优化结果后临时调整。
- Primary matrix、operational-sensitivity checks 与 matched-work controls 使用明确 identifiers 分开处理。
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
