# Results

本目录用于存放“现代模型上的 KV / 状态瓶颈画像”实验结果。

结果应保持三层可追溯结构：

```text
raw measurements
    ↓
processed / aggregated data
    ↓
figures and tables
```

## Raw results

Raw results 保存每次 run 的原始测量输出和 metadata。Metadata 至少包含：

- experiment ID；
- model identifier 与 revision；
- hardware；
- serving runtime version/commit；
- precision；
- context length、prefix ratio、output length；
- cache-residency mode；
- cache/state policy；
- offered request rate / concurrency ceiling；
- run timestamp 与 repetition index；
- validity status 与 invalid reason。

Raw measurements 不被处理脚本覆盖。

## Processed results

Processed data 由脚本从 raw measurements 确定性生成，用于计算 median、P90/P99、throughput、stall ratio、reuse benefit 和 cross-model normalization。

任何过滤规则必须能够追溯到 raw run 和明确的 invalid reason。

## Figures and tables

图表只从 processed data 生成，不手工录入最终数值。

正式报告时至少能够从 figure/table 追溯到 processed dataset，再追溯到具体 raw runs 和配置。

大体积 profiler dump、模型权重和可重新生成的大型中间文件不提交到 Git。需要保留时记录外部路径、摘要或 manifest。
