# Code

本目录用于存放 “Cache Locality and Scheduler Behavior” 的实验实现。

所有实现遵循 [`../docs/00-common-conventions.md`](../docs/00-common-conventions.md) 和仓库根目录 [`../../../docs/TECHNICAL_BASELINE.md`](../../../docs/TECHNICAL_BASELINE.md)。

## Responsibilities

代码需要覆盖以下职责：

- 固定 request set 与 context/shared-prefix groups 的构造；
- `min-distance`、`shuffle`、`max-distance` workload trace 生成；
- reuse-distance distribution 计算与 trace validation；
- request arrival-rate calibration 与 workload replay；
- baseline scheduler 的统一运行入口；
- 后续 delay-hit mitigation、balanced batching、bubble filling / stall hiding 与 full scheduler 的可切换对照入口；
- delay hit、redundant prefill、queueing delay、I/O stall、TTFT 与 throughput 采集；
- offered / achieved request rate 与 backlog/saturation 状态记录；
- 多次 repetition 的 run metadata 管理；
- raw results 到 processed results 的 deterministic processing；
- locality × load surface、ablation figure 与 summary table 生成。

## Workload requirements

不同 locality conditions 必须使用相同请求集合，只改变 request ordering 或等价的 reuse-distance structure。

每条 trace 必须具有稳定 identifier，并能够从配置和 seed 重建。

Trace metadata 至少保存：

- request count；
- context/prefix group identifier；
- request ordering；
- locality condition；
- reuse-distance summary；
- input/output token lengths；
- offered-load parameters；
- seed / configuration hash。

## Arrival-rate calibration

Experiment 1 正式 sweep 前先完成当前模型、硬件和 serving configuration 的 capacity calibration。

Calibration 输出至少包含 offered request rate、achieved throughput、queueing behavior 与 backlog status，并据此冻结 Low、Medium、High 和 Overload 四个 workload levels。

Calibration 配置与结果保存稳定 identifier，使正式实验能够追溯每个 load level 的来源。

## Measurement requirements

每次 run 至少保存：

- experiment ID；
- model identifier 与 revision；
- runtime version/commit；
- hardware、driver、CUDA/runtime；
- precision 与 cache dtype；
- cache hierarchy / capacity / policy；
- I/O backend；
- scheduler configuration；
- workload trace identifier；
- locality condition；
- reuse-distance summary；
- offered / achieved request rate；
- load level；
- delay-hit count/volume；
- redundant-prefill count/volume；
- queueing delay；
- I/O stall；
- per-request TTFT；
- throughput accounting；
- repetition index；
- validity status 与 invalid reason。

可以获取时同时保存 request-level 与 token/state-volume-weighted cache statistics。

## Processing rules

- Raw measurements 不被处理脚本修改或覆盖。
- Invalid runs 不删除，只从主 aggregation 中排除并保留原因。
- Locality comparison 必须验证 request set 与 theoretical reuse opportunity 一致。
- Arrival-rate comparison 同时保留 offered 与 achieved load，不能只按配置标签解释。
- Overload results 与稳定 serving 区间分开解释。
- Experiment 2 的 representative workload selection rule 从 Experiment 1 processed results 生成并冻结，不能根据优化后的 speedup 反向挑选 workload。
- 派生指标由 raw/processed data 计算，不把最终数字手工写死在绘图脚本中。

## Suggested structure

后续实现可以按以下职责拆分：

```text
code/
├── configs/
├── workloads/
├── runners/
├── profiling/
├── analysis/
└── README.md
```

实际目录以实现规模为准，不为了形式预建无内容目录。