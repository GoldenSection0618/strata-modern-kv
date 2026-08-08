# Results

本目录用于存放 Page Granularity and GPU-Assisted I/O 实验结果。

结果保持以下三层可追溯结构：

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
- serving runtime version / commit；
- precision；
- page size；
- context length 与 output length；
- prefix overlap / reuse condition；
- cache capacity 与 cache pressure；
- cache replacement / eviction policy；
- scheduler configuration；
- I/O backend；
- request trace identifier 与 random seed；
- repetition index；
- run timestamp；
- validity status 与 invalid reason。

Raw measurements 不被处理脚本覆盖。

## Processed results

Processed data 由脚本从 raw measurements 确定性生成。

Experiment 1 至少汇总：

- cache hit rate；
- effective reused tokens；
- cache utilization efficiency；
- cache occupancy / eviction supporting counters；
- 不同 page size、prefix overlap、context length 与 cache pressure 下的统计结果。

后续实验继续在同一结果结构中加入 host→GPU bandwidth、bandwidth utilization、prefill / decode throughput、GPU interference 与 end-to-end metrics。

任何过滤规则必须能够追溯到具体 raw run 和明确的 invalid reason。

## Figures and tables

图表只从 processed data 生成，不手工录入最终数值。

Experiment 1 的正式结果至少应包含 page size 与 effective reused tokens、cache hit rate、cache utilization efficiency 的关系，以及不同 workload dimension 下的对照和 reuse-efficient page-size region 总结。

大体积 profiler dump、模型权重和可重新生成的大型中间文件不提交到 Git。需要保留时记录外部路径、摘要或 manifest。
