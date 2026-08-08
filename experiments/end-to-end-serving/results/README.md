# Results

本目录用于存放 “End-to-End Serving” 的实验结果。

所有结果遵循 [`../docs/00-common-conventions.md`](../docs/00-common-conventions.md)，并保持以下可追溯关系：

```text
raw measurements
    ↓
processed / aggregated data
    ↓
figures and tables
```

## Raw results

Raw results 保存每次 run 的原始 measurement payload 与 metadata，不被处理脚本覆盖。

Metadata 至少包含：

- experiment ID；
- system configuration；
- model identifier 与 revision；
- hardware、driver、CUDA/runtime；
- serving runtime version / commit；
- precision 与 cache dtype；
- cache / offload backend and policy；
- scheduler policy；
- GPU / CPU cache budget；
- workload class；
- trace identifier；
- context-length point；
- input/output length distribution；
- offered-load condition；
- achieved request rate；
- cache initial state；
- run timestamp 与 repetition index；
- runtime capability status；
- validity status 与 invalid reason。

Raw measurement payload 至少尽可能保留：

- per-request arrival time；
- per-request TTFT；
- per-request completion time；
- completed request / token accounting；
- GPU utilization samples；
- queue length / queueing time when available；
- GPU / CPU cache hit volume；
- reusable-state eviction；
- recomputation；
- CPU-GPU transfer activity；
- non-overlapped I/O stall；
- runtime errors、fallbacks 与 scheduler events。

## Processed results

Processed data 由脚本从 raw measurements deterministic 生成，并保留 run identifiers。

Processed data 至少支持：

- achieved throughput；
- P50 / P90 / P99 TTFT；
- request completion time distribution；
- GPU utilization summary；
- cold-start vs steady-state comparison；
- system-configuration relative gain；
- offered-load scaling curves；
- saturation-region identification；
- cache / recomputation / I/O / queueing auxiliary analysis。

任何 filtering 都需要保留明确的 invalid reason。Raw data 不因进入不了主 aggregation 而删除。

## Experiment 1 outputs

Long-context Reuse Serving 至少形成：

1. 不同 context length 下 throughput 随 offered load 的变化；
2. 不同 context length 下 P50 / P90 / P99 TTFT 随 offered load 的变化；
3. request completion time 随 offered load 的变化；
4. GPU utilization 随 offered load 的变化；
5. cold-start 与 steady-state performance 对比；
6. representative medium-load / high-load points 下五种 system configurations 的直接比较；
7. 与关键性能差异对应的 cache reuse、recomputation、CPU-GPU traffic、I/O stall 与 queueing 辅助结果。

Experiment 1 的主结果按照 `context length × offered load × system configuration` 组织。

## Result interpretation

结果摘要必须同时报告 throughput 与 latency。不能只依据单一吞吐提升判断系统整体更优。

如果 Full Configuration 提高 throughput 但显著恶化 P99 TTFT 或 request completion time，则结果标记为 throughput-latency trade-off。

如果低负载无明显差异、中高负载开始出现收益，则将其解释为 serving-capacity / contention-region improvement。

如果 cache hit 与 avoided recomputation 明显，但端到端性能变化有限，则使用 data movement、I/O stall、queueing 与 GPU utilization 辅助解释后续瓶颈。

## Figures and tables

所有正式 figure/table 只从 processed data 生成，不手工录入最终数值。

正式报告中的任何图表必须能够追溯到：

```text
figure/table
    ↓
processed dataset + processing config/commit
    ↓
raw run identifiers
    ↓
run metadata + validity status
```

Relative metrics 必须保留 underlying absolute measurements。

## Storage policy

大体积 profiler dump、模型权重与可重新生成的大型中间文件不提交到 Git。

需要长期保留的外部结果记录：

- external path or object identifier；
- checksum；
- generating run identifier；
- runtime / processing version；
- retention note when applicable。
