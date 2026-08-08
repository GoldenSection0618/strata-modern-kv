# End-to-End Serving

本目录用于验证前述 cache、I/O 与 scheduler 机制在完整 serving pipeline 中能否转化为稳定的端到端系统收益。

本部分不再单独证明某个机制为什么有效，而是把前面 microbenchmark 与 ablation 中得到的机制结论放回完整 serving workload，检查最终 throughput、latency 与 GPU utilization 是否真正改善。

## Scope

本部分包含三个实验：

1. **Long-context Reuse Serving**：在具有明显 shared-prefix / long-context reuse 的 workload 下，验证 hierarchical cache、I/O optimization、scheduler optimization 与 full configuration 的端到端收益。
2. **Short-context Serving**：在普通短请求 workload 下检查针对 long-context 的优化是否引入 throughput 或 latency regression。
3. **Mixed Workload Serving**：同时混合长共享 context、普通短请求、不同 output length 与不同 cache locality，验证完整系统在更接近真实 serving 的异构 workload 下是否仍然稳定有效，并检查不同请求类型之间的 cross-class interference。

Load scaling 不单独拆成第四个实验，而作为三个实验中的统一控制维度。每个实验都覆盖从低负载到接近饱和的多个 offered-load 条件，用于观察优化在不同系统压力下的收益边界。

Experiments 1–3 均已完成详细方案设计。

## Directory structure

```text
end-to-end-serving/
├── README.md
├── docs/
│   ├── 00-common-conventions.md
│   ├── 01-long-context-reuse-serving.md
│   ├── 02-short-context-serving.md
│   └── 03-mixed-workload-serving.md
├── code/
│   └── README.md
└── results/
    └── README.md
```

- `docs/00-common-conventions.md`：统一系统配置、workload、负载、测量与结果判定口径。
- `docs/01-long-context-reuse-serving.md`：Experiment 1 的详细实验方案。
- `docs/02-short-context-serving.md`：Experiment 2 的 short-context regression 实验方案。
- `docs/03-mixed-workload-serving.md`：Experiment 3 的 heterogeneous mixed workload、cross-class interference 与 robustness 实验方案。
- `code/`：workload trace、实验运行、指标采集、结果处理与绘图代码。
- `results/`：raw measurements、processed data、统计结果、图表与结果摘要。

## Comparison configurations

三个实验统一使用以下五种 system configurations：

1. **Baseline**；
2. **Hierarchical Cache**；
3. **Hierarchical Cache + I/O Optimization**；
4. **Hierarchical Cache + Scheduler Optimization**；
5. **Full Configuration**。

这五种配置构成统一 comparison set，但不是严格的逐层 feature chain。

- 配置 2 只增加经过验证的 hierarchy。
- 配置 3 在 hierarchy 上加入经过验证的 I/O optimization，并保持 reference/baseline scheduler。
- 配置 4 在 hierarchy 上加入经过验证的 scheduler optimization，并保持 reference/baseline I/O path。
- 配置 5 同时启用经过验证的 hierarchy、I/O 与 scheduler mechanisms。

配置 3 与配置 4 是 parallel attribution branches。它们用于判断 I/O 与 scheduler 在完整 serving 中各自是否还有增量价值。Full Configuration 用于验证机制同时启用后的最终系统收益和 interaction。

## Core metrics

本部分统一观察：

- request throughput 与 token throughput；
- P50 / P90 / P99 TTFT；
- request completion time；
- GPU utilization。

Mixed workload 额外要求按 long-context 与 short-context request class 分别报告 request/token throughput 与 latency，避免 aggregate metrics 掩盖 cross-class interference。

TPOT 或等价 decode-latency metric 仅在解释 decode-heavy 或 output-length heterogeneity 时作为辅助指标记录。

必要的 cache hit、recomputation、CPU-GPU data movement、queueing time、I/O stall 与 batch behavior 作为辅助解释指标保留，用于把端到端结果与前四组实验建立证据链。

## Experiment logic

```text
Experiment 1
目标场景下，long-context reuse 能否转化为最终 serving 收益？
        ↓
Experiment 2
这些优化是否会损害普通 short-context serving？
        ↓
Experiment 3
在更真实的异构 mixed workload 中，整体收益是否仍然成立，是否出现跨请求类型干扰？
```

三个实验分别回答目标场景收益、非目标场景 regression 与复杂 workload robustness。不会为了保持实验数量一致而增加缺乏独立研究问题的分实验。

Experiment 3 采用 representative primary matrix 加 targeted robustness checks，而不是对 workload composition、cache locality、output-length heterogeneity 和 offered load 做完整笛卡尔积。

Composition 与 output-length robustness 同时区分两种比较语义：固定 request-arrival trace 的 operational sensitivity，以及控制 offered work / matched load 后的 attribution check。前者反映真实业务混合变化，后者用于避免把更多 token/compute work 错误归因于 composition 或 heterogeneity 本身。

## Execution discipline

所有正式比较遵循 [`docs/00-common-conventions.md`](docs/00-common-conventions.md) 和仓库根目录 [`docs/TECHNICAL_BASELINE.md`](../../docs/TECHNICAL_BASELINE.md)。

所有 paired comparisons 使用冻结的 workload trace、load grid、GPU reusable-cache budget 与 measurement rule。CPU-tier budget 在所有启用 hierarchy 的配置中保持一致。

实验不预设 Full Configuration 必须在所有 workload point 上取得正收益。若优化只在中高负载有效、只改善 throughput、造成 tail-latency regression、产生 cross-class interference，或在现代 runtime 上几乎没有额外收益，都作为有效结果报告。
