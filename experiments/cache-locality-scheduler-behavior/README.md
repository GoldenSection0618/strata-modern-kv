# Cache Locality and Scheduler Behavior

本目录用于评估 Strata 的 control-plane / scheduling 优化在现代 hybrid LLM serving workload 中是否仍然具有实际价值，并识别这些机制的有效 workload 区域。

本部分关注 request ordering、cache locality、arrival pressure 与 scheduler behavior 之间的关系。实验不预设 locality-aware scheduling 一定产生收益，而是先确认 scheduler pathology 是否仍然存在，再通过消融确定不同机制分别解决什么问题。

## Scope

本部分包含四个实验：

1. **Locality × Arrival Rate Baseline Profiling**：只使用 baseline scheduler，联合控制 request arrival rate 与 cache locality，建立 delay hit、redundant prefill、queueing、I/O stall、TTFT 和 throughput 的基础画像。
2. **Scheduler Component Ablation**：在实验一选出的 representative workloads 上逐步加入 delay-hit mitigation、balanced batching 与 bubble filling / stall hiding，分离各机制的贡献。
3. **Same-Context Concurrency Stress**：构造大量请求集中访问相同或高度相关 context 的 workload，研究高并发下 cache reuse opportunity 是否被 contention、delay hit 或 redundant prefill 破坏。
4. **Scheduler Operating Region**：复用前三组结果并补充少量边界点，确定不同 scheduler mechanism 在 locality × load space 中的有效区域与收益边界。

实验一负责回答“问题在哪里”。实验二负责回答“哪个机制解决哪个问题”。实验三负责测试高共享 context 下的压力场景。实验四负责形成最终 workload-to-mechanism 结论。

本组实验对应 Strata 原文 Fig.9、Fig.11 与 Fig.12 所研究的主要 scheduling 问题，但不机械复制三套独立实验，而是在统一 workload-control framework 中重新组织验证。

## Directory structure

```text
cache-locality-scheduler-behavior/
├── README.md
├── docs/
│   ├── 00-common-conventions.md
│   ├── 01-locality-arrival-rate-baseline.md
│   ├── 02-scheduler-component-ablation.md
│   └── 03-same-context-concurrency-stress.md
├── code/
│   └── README.md
└── results/
    └── README.md
```

- `docs/`：统一 workload/measurement 约定与各实验设计。
- `code/`：workload trace 构造、实验运行、指标采集、结果处理与绘图代码。
- `results/`：raw measurements、processed data、统计结果与可复现图表产物。

Experiments 1–3 已有详细设计。Experiment 4 的详细设计在对应实验方案冻结后加入 `docs/`，不提前创建空文档。

## Experiment logic

```text
Experiment 1
locality 与 load 在哪里暴露 scheduler pathology？
        ↓
Experiment 2
每个 scheduler mechanism 分别解决什么问题？
        ↓
Experiment 3
高共享 context + 高并发下机制是否仍然成立？
        ↓
Experiment 4
不同机制的有效 workload 区域在哪里？
```

## Core metrics

本部分重点观察：

- cache hit / reuse realization；
- delay hit；
- redundant prefill；
- queueing delay；
- I/O stall；
- TTFT distribution；
- achieved throughput。

实验分析必须把 scheduler-level pathology 与用户可见 serving performance 放在同一证据链中。单独的 cache hit rate 或单独的 throughput 变化都不足以说明机制成立。

## Execution discipline

所有实验固定模型、硬件、cache hierarchy、I/O backend、cache capacity 和请求内容分布，只在当前实验明确声明的变量上进行变化。

Experiment 1 只使用 baseline scheduler。后续 scheduler optimization 不进入 Experiment 1，以保证 baseline pathology 的定位不被机制本身改变。

所有 workload trace 必须具有稳定 identifier，并能够从配置与 seed 重建。不同 locality 条件使用同一请求集合，只改变请求访问顺序或 reuse distance 结构。

正式实验结果保留 raw run、processed result 与 figure/table 的可追溯关系。