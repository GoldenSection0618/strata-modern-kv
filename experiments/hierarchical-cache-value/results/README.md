# Results

本目录用于存放 “Hierarchical Cache Value Evaluation” 的实验结果。

所有结果遵循 [`../docs/00-common-conventions.md`](../docs/00-common-conventions.md)，并保持三层可追溯结构：

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
- model role: primary / secondary；
- model identifier 与 revision；
- hardware、driver、CUDA/runtime；
- serving runtime version/commit；
- precision 与 cache dtype；
- cache architecture: GPU-only / hierarchical；
- hierarchy validation status: full / partial / unsupported；
- validated state groups；
- cache/offload backend and policy；
- cache initial state；
- GPU cache budget；
- CPU tier budget；
- observed GPU cache occupancy；
- workload trace identifier；
- prefix length / context distribution；
- configured revisit fraction；
- actual request-weighted reuse；
- actual token/state-volume-weighted reuse；
- reuse-distance summary；
- output length distribution；
- offered-load condition；
- achieved request rate；
- effective concurrency when available；
- active-request preemption count；
- cache-pressure calibration identifier；
- run timestamp 与 repetition index；
- validity status 与 invalid reason。

Raw measurement payload 至少应尽可能保留：

- GPU hit volume；
- GPU eviction volume；
- CPU hit volume；
- state-group hit/restore breakdown；
- recomputed token/state volume or verified computation measure；
- CPU-GPU transfer volume / activity；
- non-overlapped restore stall；
- per-request TTFT；
- completed requests / throughput accounting；
- runtime errors、fallbacks 和 preemption events。

## Validation status

不同 hierarchy capability status 必须分开保存和聚合。

### `full`

CPU restore 已验证覆盖跳过目标 prefix computation 所需的全部 state groups。

### `partial`

只验证或只支持部分 state groups。Partial hierarchy 可以作为 runtime-support observation，但不能并入 full-hierarchy performance curve。

### `unsupported`

Pinned runtime 无法建立或验证所需 hierarchical path。Unsupported 是有效的 capability conclusion，不使用其他机制填充缺失结果。

## Processed results

Processed data 由脚本从 raw measurements deterministic 生成。

处理过程必须保留 run identifiers，使任何 aggregation 都能回溯到原始 run。

任何过滤规则都需要明确 invalid reason。例如：

- active-request preemption；
- OOM；
- effective concurrency drift；
- CPU tier capacity pressure；
- workload trace mismatch；
- hierarchy validation failure；
- restore fallback；
- unstable initialization。

Processed data 至少支持：

- GPU hit / eviction；
- CPU-tier contribution；
- recomputation reduction；
- restore traffic 与 non-overlapped stall；
- TTFT distribution；
- throughput；
- cold vs warm difference；
- GPU-only vs hierarchical relative benefit；
- capacity-pressure value curve；
- prefix-reuse value curve；
- cross-model representative-point comparison。

## Experiment 1 outputs

Baseline Benefit 至少形成：

1. cold-cache GPU-only vs hierarchical 的 GPU/CPU hit、eviction 与 recomputation；
2. warm-cache 相同对比；
3. CPU-GPU restore traffic / stall；
4. cold-cache TTFT 与 throughput；
5. warm-cache TTFT 与 throughput；
6. cold-cache 运行过程中 reuse benefit 随请求进程的变化。

## Experiment 2 outputs

GPU Cache Pressure Scaling 至少形成：

1. GPU reusable-cache budget / observed pressure → GPU hit and eviction；
2. pressure → CPU-tier hit contribution；
3. pressure → recomputation reduction + restore traffic/stall；
4. pressure → relative TTFT improvement；
5. pressure → throughput gain；
6. 每个 point 的 active-request preemption 与 validity status。

任何发生 preemption、OOM 或 effective concurrency drift 的配置不进入主 capacity-pressure curve。

## Experiment 3 outputs

Prefix Reuse Scaling 至少形成：

1. configured revisit fraction → actual reuse；
2. reuse-distance distribution validation；
3. prefix reuse → GPU hit / CPU hit / eviction；
4. prefix reuse → recomputation reduction；
5. prefix reuse → restore traffic/stall；
6. prefix reuse → TTFT / throughput benefit。

结果必须证明不同 reuse trace 没有同时改变 request ordering、hotspot concentration 或 reuse-distance structure。

## Experiment 4 outputs

Cross-Model Validation 至少形成一张 representative-point summary：

| Point | Model | Observed pressure | Actual reuse | CPU-tier contribution | Recomputation reduction | TTFT improvement | Throughput gain | Validation status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| V0 | primary | ... | ... | ... | ... | ... | ... | full |
| V0 | secondary | ... | ... | ... | ... | ... | ... | full / partial / unsupported |
| V1 | primary | ... | ... | ... | ... | ... | ... | full |
| V1 | secondary | ... | ... | ... | ... | ... | ... | full / partial / unsupported |
| V2 | primary | ... | ... | ... | ... | ... | ... | full |
| V2 | secondary | ... | ... | ... | ... | ... | ... | full / partial / unsupported |

Only rows with `full` status enter the full-hierarchy cross-model performance comparison. `partial` and `unsupported` rows remain visible as runtime-capability evidence.

Cross-model analysis 主要比较方向和 mechanism chain，不只比较单个 speedup 数字。

## Figures and tables

图表只从 processed data 生成，不手工录入最终数值。

正式报告中的任何 figure/table 必须能够追溯到：

```text
figure/table
    ↓
processed dataset + processing commit/config
    ↓
raw run identifiers
    ↓
run metadata + runtime validation status
```

Cross-model normalized metrics 必须同时保留 underlying absolute measurements。

## Storage policy

大体积 profiler dump、模型权重和可重新生成的大型中间文件不提交到 Git。

需要长期保留的外部数据记录：

- external path or object identifier；
- checksum；
- generating run identifier；
- runtime / processing version；
- retention note when applicable。
