# Code

本目录用于存放 “Model and Hardware Generalization” 实验实现。

所有实现遵循 [`../docs/00-common-conventions.md`](../docs/00-common-conventions.md) 与仓库根目录 [`../../../docs/TECHNICAL_BASELINE.md`](../../../docs/TECHNICAL_BASELINE.md)。

## Responsibilities

代码需要覆盖以下职责：

- representative-point configuration materialization；
- Qwen3.5-9B 与 Gemma 4 12B 的 matched workload generation；
- tokenizer-aware token/work summary；
- relative state-pressure calibration；
- relative serving-load calibration；
- baseline bottleneck profiling；
- hierarchy / I/O / scheduler mechanism capability validation；
- matched system-configuration execution；
- serving-state、cache、I/O、scheduler 与 serving metrics 采集；
- raw result metadata 与 validity status 管理；
- normalized-effect calculation；
- cross-model / cross-hardware robustness matrix generation；
- figures and tables generation。

## Experiment 1 requirements

Experiment 1 的实现必须能够：

1. 在 A100 40GB 上运行 Qwen3.5-9B 与 Gemma 4 12B；
2. 使用同一 logical workload definition 为两个 tokenizer materialize request traces；
3. 保存 actual input/output token counts 与 offered-work summary；
4. 构造 State Pressure、Reuse and I/O、Locality and Scheduling 三类 representative workloads；
5. 在 optimized results 产生前冻结 relative pressure/load regions；
6. 对每个模型建立 baseline bottleneck profile；
7. 只启用与当前 mechanism attribution 有关的 validated system configurations；
8. 记录 full / partial hierarchy capability；
9. 记录 cache/state residency、eviction、restore、recomputation、CPU-GPU traffic、stall、queueing、TTFT 与 throughput；
10. 对同一 representative point 执行多次独立重复测量；
11. 保存 absolute measurements、normalized effect 与 uncertainty；
12. 输出 `stable`、`weakened`、`model_sensitive`、`boundary_case`、`inconclusive` 等 conclusion 所需的证据字段。

## Runtime validation

正式 runner 不得只验证模型能够启动。

至少需要验证：

- exact model revision 与 runtime implementation；
- 目标 serving-state group 可观测；
- full hierarchy 所需 state group 均能正确 restore；
- I/O backend 的实际 path 与配置一致；
- scheduler mechanism semantics 与前置实验冻结的定义一致；
- paired runs 未发生未记录的 fallback；
- trace、cache/state budget、arrival/load rule 与 measurement boundary 与配置一致；
- instrumentation failure 不会静默生成缺失指标。

Validation result 必须写入 run metadata。

## Suggested structure

后续实现可以按以下职责拆分：

```text
code/
├── configs/
├── workloads/
├── calibration/
├── validation/
├── runners/
├── profiling/
├── analysis/
└── README.md
```

实际目录以实现规模为准，不为了形式预建无内容目录。
