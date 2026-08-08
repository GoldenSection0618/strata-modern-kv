# Results

本目录用于存放 “Cache Locality and Scheduler Behavior” 的实验结果。

所有结果遵循 [`../docs/00-common-conventions.md`](../docs/00-common-conventions.md)，并保持以下可追溯结构：

```text
raw measurements
    ↓
processed / aggregated data
    ↓
figures and tables
```

## Raw results

Raw results 保存每次 run 的原始测量输出与 metadata，不被处理脚本覆盖。

Metadata 至少包含：

- experiment ID；
- model identifier 与 revision；
- hardware、driver、CUDA/runtime；
- serving runtime version/commit；
- precision 与 cache dtype；
- cache hierarchy / capacity / policy；
- I/O backend；
- scheduler configuration；
- workload trace identifier；
- locality condition；
- request count；
- context/prefix group summary；
- configured and observed reuse-distance summary；
- input/output length distribution；
- offered-load condition；
- offered request rate；
- achieved request rate；
- backlog/saturation status；
- repetition index；
- validity status 与 invalid reason。

Raw measurement payload 至少尽可能保留：

- realized cache hit / reuse；
- delay hit；
- redundant prefill；
- queueing delay；
- I/O stall；
- per-request TTFT；
- completed-request / throughput accounting；
- runtime errors、fallbacks 和异常 scheduler events。

## Processed results

Processed data 由脚本从 raw measurements deterministic 生成。

处理过程保留 run identifiers，使任何 aggregation 都能回溯到原始 run。

任何过滤规则都记录明确 invalid reason，例如：

- OOM；
- workload trace mismatch；
- scheduler configuration drift；
- runtime fallback；
- measurement failure；
- initialization instability；
- calibration mismatch。

Processed data 至少支持：

- locality → actual reuse-distance validation；
- locality × load → delay hit；
- locality × load → redundant prefill；
- locality × load → queueing delay；
- locality × load → I/O stall；
- locality × load → TTFT distribution；
- locality × load → achieved throughput；
- offered vs achieved load comparison；
- Experiment 2 representative workload selection。

## Experiment 1 outputs

Locality × Arrival Rate Baseline Profiling 至少形成：

1. 三种 locality condition 的实际 reuse-distance distribution；
2. locality × arrival rate 的 delay-hit surface；
3. locality × arrival rate 的 redundant-prefill surface；
4. queueing-delay / I/O-stall summary；
5. TTFT median 与 tail-latency summary；
6. achieved-throughput summary；
7. offered vs achieved request-rate / saturation summary；
8. representative workload selection table。

Experiment 1 的主结果同时保留 scheduler-level pathology 与 user-visible performance，避免只凭单一指标判断 locality 或 load 的影响。

## Representative workload table

Experiment 2 使用的 workload points 从 Experiment 1 主结果中预先选择并冻结。

结果文件至少记录：

| Point | Locality | Load level | Key pathology | TTFT behavior | Throughput behavior | Selection role |
|---|---|---|---|---|---|---|
| C0 | ... | ... | weak | ... | ... | control |
| L1 | ... | ... | locality-sensitive | ... | ... | locality case |
| H1 | ... | ... | load-amplified | ... | ... | stress case |

最终 point 数量由 Experiment 1 结果决定，但 selection rule 必须在 Experiment 2 scheduler ablation 之前冻结。

## Figures and tables

图表只从 processed data 生成，不手工录入最终数值。

正式报告中的 figure/table 保持以下追溯关系：

```text
figure/table
    ↓
processed dataset + processing commit/config
    ↓
raw run identifiers
    ↓
run metadata + workload/calibration identifiers
```

所有 relative metrics 同时保留 underlying absolute measurements。

## Storage policy

大体积 profiler dump、模型权重和可重新生成的大型中间文件不提交到 Git。

需要长期保留的外部数据记录 external path/object identifier、checksum、generating run identifier、runtime/processing version 与 retention note。