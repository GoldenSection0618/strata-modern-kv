# Code

本目录用于存放 “Hierarchical Cache Value Evaluation” 的实验实现。

所有实现必须遵循 [`../docs/00-common-conventions.md`](../docs/00-common-conventions.md) 和仓库根目录 [`../../../docs/TECHNICAL_BASELINE.md`](../../../docs/TECHNICAL_BASELINE.md)。

## Responsibilities

代码需要覆盖以下职责：

- text-only shared-prefix workload 构造与 deterministic request trace 生成；
- Experiment 3 的 eligible revisit slots 与 matched unique-prefix replacement；
- GPU-only 与 GPU + CPU hierarchical 两种运行模式；
- cold-cache 与 warm-cache 初始状态建立和验证；
- full / partial / unsupported hierarchy capability validation；
- hybrid model 各 state group 的 residency / restore 覆盖验证；
- GPU reusable-cache pressure calibration；
- active-request preemption、effective concurrency 和 achieved request rate 监控；
- GPU hit / eviction、CPU hit、recomputation、CPU-GPU traffic、non-overlapped restore stall、TTFT 与 throughput 采集；
- Experiments 1–3 的 primary-model sweep；
- Experiment 4 的 second-model matched validation；
- raw results 到 processed results 的 deterministic processing；
- figure/table generation。

## Runtime validation

正式实验入口不得只检查模型是否能够启动。

至少需要验证：

1. prefix-cache reuse 与 full recomputation 的输出一致性；
2. GPU-resident 与 CPU-resident reuse 能够通过 runtime observable behavior 确认；
3. CPU restore 覆盖跳过目标 prefix computation 所需的全部 state groups；
4. Qwen3.5 在作为 full-hierarchy target 时同时覆盖 attention KV 与 Gated DeltaNet recurrent state；
5. Gemma 4 在作为 full-hierarchy target 时覆盖 pinned runtime 实际保留的 local/sliding-window 与 global-attention state groups；
6. restore failure 不会被静默计为 CPU hit；
7. paired runs 使用相同 cache policy、cache dtype 和 offloading backend。

Validation output 必须写入 run metadata，而不是只打印到终端。

## Workload requirements

每条 trace 必须具有稳定 identifier，并能够由版本控制配置和 seed 重建。

Experiment 3 不通过 request reordering、减少 prefix-group 数量或集中到少量热点 prefix 来提高 reuse。它通过固定 eligible revisit slots，在相同位置选择 revisit existing prefix 或 matched unique prefix 来改变 revisit fraction。

Trace metadata 至少保存：

- request count；
- prefix identifier；
- prefix token length；
- revisit flag；
- reuse distance；
- input/output token lengths；
- offered-load parameters；
- seed / configuration hash。

## Cache-pressure calibration

Experiment 2 在 sweep 前需要先确定固定 active workload 不发生 scheduler preemption 所需的运行容量。

主 pressure curve 只改变可供 reusable state 使用的 capacity headroom。

任何出现以下情况的 point 均不能静默进入主曲线：

- OOM；
- active-request preemption；
- effective concurrency 改变；
- offered-load condition 改变；
- CPU tier capacity eviction 成为新的主要变量。

这些 run 仍保存 raw data，并记录 invalid reason。

## Measurement requirements

优先记录 token/state-volume-weighted cache statistics，而不是只记录 request-level hit count。

CPU-GPU transfer activity 与 non-overlapped restore stall 必须区分。Raw transfer duration 不与 computation time 直接相加形成 TTFT decomposition。

每次 run 的 metadata 至少包括：

- experiment ID；
- model role: primary / secondary；
- model identifier 与 revision；
- runtime version/commit；
- hardware、driver、CUDA/runtime；
- precision 与 cache dtype；
- hierarchy validation status: full / partial / unsupported；
- validated state groups；
- cache/offload backend and policy；
- GPU cache budget 与 CPU tier budget；
- initial cache state；
- workload trace identifier；
- configured / actual reuse；
- cache-pressure calibration identifier；
- offered / achieved request rate；
- active-request preemption count；
- repetition index；
- validity status 与 invalid reason。

## Processing rules

- Raw measurements 不被处理脚本修改或覆盖。
- Full hierarchy、partial hierarchy 与 unsupported results 分开处理。
- Invalid runs 不删除，只从主 aggregation 中排除并保留原因。
- 派生指标由 raw/processed data 计算，不把最终数字手工写死在脚本中。
- Cross-model normalization 必须同时保留 absolute measurements。
- Experiment 4 的 representative-point selection rule 必须在 second-model performance analysis 前冻结并记录。

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
