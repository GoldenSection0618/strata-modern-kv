# End-to-End Serving

本目录用于验证前述 cache、I/O 与 scheduler 机制在完整 serving pipeline 中能否转化为稳定的端到端系统收益。

本部分不再单独证明某个机制为什么有效，而是把前面 microbenchmark 与 ablation 中得到的机制结论放回完整 serving workload，检查最终 throughput、latency 与 GPU utilization 是否真正改善。

## Scope

本部分包含三个实验：

1. **Long-context Reuse Serving**：在具有明显 shared-prefix / long-context reuse 的 workload 下，验证 hierarchical cache、I/O optimization、scheduler optimization 与 full configuration 的端到端收益。
2. **Short-context Serving**：在普通短请求 workload 下检查针对 long-context 的优化是否引入 throughput 或 latency regression。
3. **Mixed Workload Serving**：同时混合长共享 context、普通短请求、不同 output length 与不同 cache locality，验证完整系统在更接近真实 serving 的异构 workload 下是否仍然稳定有效。

Load scaling 不单独拆成第四个实验，而作为三个实验中的统一控制维度。每个实验都覆盖从低负载到接近饱和的多个 offered-load 条件，用于观察优化在不同系统压力下的收益边界。

目前已完成 Experiments 1–2 的详细方案设计。Experiment 3 的详细文档在对应方案确定后补充。

## Directory structure

```text
end-to-end-serving/
├── README.md
├── docs/
│   ├── 00-common-conventions.md
│   ├── 01-long-context-reuse-serving.md
│   └── 02-short-context-serving.md
├── code/
│   └── README.md
└── results/
    └── README.md
```

- `docs/00-common-conventions.md`：统一系统配置、workload、负载、测量与结果判定口径。
- `docs/01-long-context-reuse-serving.md`：Experiment 1 的详细实验方案。
- `docs/02-short-context-serving.md`：Experiment 2 的 short-context regression 实验方案。
- `code/`：workload trace、实验运行、指标采集、结果处理与绘图代码。
- `results/`：raw measurements、processed data、统计结果、图表与结果摘要。

## Comparison configurations

三个实验统一沿用逐步增加能力的系统配置链：

1. Baseline；
2. Hierarchical Cache；
3. Hierarchical Cache + I/O Optimization；
4. Hierarchical Cache + Scheduler Optimization；
5. Full Configuration。

该配置链同时提供 end-to-end comparison 与轻量 attribution。前面实验负责机制层面的严格因果验证，本组只检查这些机制组合后是否产生最终 serving 收益。

## Core metrics

本部分重点观察：

- throughput；
- P50 / P90 / P99 TTFT；
- request completion time；
- GPU utilization。

必要的 cache hit、recomputation、CPU-GPU data movement、queueing time 与 I/O stall 作为辅助解释指标保留，用于把端到端结果与前四组实验建立证据链。

## Experiment logic

```text
Experiment 1
目标场景下，long-context reuse 能否转化为最终 serving 收益？
        ↓
Experiment 2
这些优化是否会损害普通 short-context serving？
        ↓
Experiment 3
在更真实的异构 mixed workload 中，整体收益是否仍然成立？
```

三个实验分别回答目标场景收益、非目标场景 regression 与复杂 workload robustness。不会为了保持实验数量一致而增加缺乏独立研究问题的分实验。

## Execution discipline

所有正式比较遵循 [`docs/00-common-conventions.md`](docs/00-common-conventions.md) 和仓库根目录 [`docs/TECHNICAL_BASELINE.md`](../../docs/TECHNICAL_BASELINE.md)。

实验不预设 Full Configuration 必须在所有 workload point 上取得正收益。若优化只在中高负载有效、只改善 throughput、造成 tail-latency regression，或在现代 runtime 上几乎没有额外收益，都作为有效结果报告。
