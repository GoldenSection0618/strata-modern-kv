# Results

本目录用于存放“Hierarchical Cache Value Evaluation”实验结果。

结果保持与其他实验目录一致的可追溯结构：

```text
raw measurements
    ↓
processed / aggregated data
    ↓
figures and tables
```

## Raw results

Raw results 保存每次 run 的原始测量输出与 metadata。Metadata 至少包含：

- experiment ID；
- model identifier 与 revision；
- hardware；
- serving runtime version/commit；
- precision；
- cache architecture；
- cache initial state；
- GPU cache budget；
- cache/state policy；
- prefix reuse 条件；
- workload trace identifier；
- output length；
- offered load / concurrency condition；
- run timestamp 与 repetition index；
- validity status 与 invalid reason。

Raw measurements 不被处理脚本覆盖。

## Processed results

Processed data 由脚本从 raw measurements 确定性生成。

处理后结果至少能够支持以下分析：

- GPU cache hit 与 CPU cache hit；
- recomputation reduction；
- CPU-GPU traffic；
- TTFT distribution；
- throughput；
- cold-cache 与 warm-cache 差异；
- GPU-only 与 hierarchical cache 的相对收益。

任何过滤和异常 run 排除规则必须能够追溯到明确的 raw run 与 invalid reason。

## Figures and tables

图表只从 processed data 生成，不手工录入最终数值。

Experiment 1 的正式结果至少应形成以下对比：

1. GPU-only vs hierarchical cache 的 cache hit；
2. GPU-only vs hierarchical cache 的 recomputation；
3. hierarchical cache 的 CPU-GPU traffic；
4. cold-cache 下的 TTFT 与 throughput；
5. warm-cache 下的 TTFT 与 throughput；
6. cold-cache 运行过程中 reuse benefit 随请求进程的变化。

正式报告中的任何结论都应能够从 figure/table 追溯到 processed dataset，再追溯到具体 raw runs 与配置。

大体积 profiler dump、模型权重和可重新生成的大型中间文件不提交到 Git。需要保留时记录外部路径、摘要或 manifest。
