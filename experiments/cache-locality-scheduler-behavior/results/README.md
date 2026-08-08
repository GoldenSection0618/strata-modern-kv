# Results

本目录用于存放 “Cache Locality and Scheduler Behavior” 的实验结果。

所有结果遵循 [`../docs/00-common-conventions.md`](../docs/00-common-conventions.md)，并保持：

```text
raw measurements
    ↓
processed / aggregated data
    ↓
figures and tables
```

## Raw results

Raw results 保存每次 run 的原始 measurement 与 metadata，不被处理脚本覆盖。

Metadata 至少包含：

- experiment ID；
- model identifier / revision；
- hardware、driver、CUDA/runtime；
- serving runtime version / commit；
- precision / cache dtype；
- cache hierarchy / capacity / policy；
- hierarchy validation status；
- I/O backend / host layout when applicable；
- scheduler configuration；
- mechanism capability status；
- workload trace identifier；
- cache-distance condition；
- configured / observed reuse-distance summary；
- request count；
- context/prefix group summary；
- input/output length distribution；
- theoretical reusable volume；
- offered-load condition；
- offered / achieved request rate；
- backlog/saturation status；
- resolve mode；
- configured / observed same-context fan-in when applicable；
- observed cache resolve time when applicable；
- repetition index；
- validity status / invalid reason。

Raw payload 尽可能保留：

- realized cache reuse / reuse realization；
- delay hit / affected volume；
- redundant prefill / recomputation；
- GPU/CPU hit and restore events when applicable；
- duplicate restore activity；
- batch load / compute ratio；
- loading-bound decisions；
- bundle-hit count / volume；
- queueing / deferral time；
- non-overlapped I/O stall；
- GPU idle / filled-bubble time / inserted-work type；
- per-request TTFT；
- TPOT or equivalent decode latency；
- completion / throughput accounting；
- runtime fallback、preemption、starvation 和异常 scheduler events。

## Capability and validity status

不同 capability status 分开保存与聚合。

### `supported`

目标 scheduler mechanism 与 cache/state path 的语义已经验证。

### `partial`

只能验证部分 hybrid state group 或部分 scheduler semantics。Partial result 可作为 capability evidence，但不能混入 full-mechanism performance curve。

### `unsupported`

当前 pinned runtime 无法建立所需机制或状态路径。Unsupported 是有效结果，不使用另一机制填补缺口。

Invalid run 同样保留 raw data，并记录明确原因。

## Processed results

Processed data 由脚本 deterministic 生成，并保留 run identifiers。

处理至少支持：

- configured condition → actual reuse-distance validation；
- cache distance × load → delay hit / redundant work；
- cache distance × load → host restore / I/O stall when valid；
- offered vs achieved load；
- TTFT / TPOT / throughput；
- Experiment 2 W0–W3 frozen selection；
- progressive scheduler ablation；
- mechanism-level attribution；
- Experiment 3 fan-in / cache-resolve analysis；
- Experiment 4 operating-region classification / boundary validation。

## Experiment 1 outputs

至少形成：

1. Min/Shuffle/Max 的 actual reuse-distance distributions；
2. cache distance × arrival rate 的 delay-hit / redundant-work surface；
3. cache distance × arrival rate 的 host-restore / I/O-stall surface when hierarchy is valid；
4. queueing summary；
5. P50/P90/P99 TTFT summary；
6. offered vs achieved throughput summary；
7. W0–W3 representative workload selection table。

### Representative workload table

使用与 Experiment 2 一致的命名：

| Point | Cache distance | Load | Dominant baseline pathology | Selection role | Source runs |
|---|---|---|---|---|---|
| W0 | ... | ... | weak | control | ... |
| W1 | ... | ... | delay hit / redundant work | delay-hit-sensitive | ... |
| W2 | ... | ... | host-loading imbalance | loading-balance-sensitive | ... |
| W3 | ... | ... | residual I/O stall | stall-sensitive | ... |

某一角色在 baseline 中没有出现时记录 `not observed`，不人工挑选替代点。

## Experiment 2 outputs

至少形成：

1. W0–W3 的 S0→S3 progressive throughput / TTFT ablation；
2. S0 vs S1 的 delay hit、deferral、redundant work、reuse realization；
3. S1 vs S2 的 load/compute ratio、loading-bound fraction、bundle hits、exposed I/O stall；
4. S2 vs S3 的 residual stall、GPU idle、filled-bubble time、inserted-work type；
5. P50/P90/P99 TTFT + TPOT safety summary；
6. targeted leave-one-out result when semantics permit；
7. scheduler mechanism support / partial / unsupported table。

## Experiment 3 outputs

Cold-miss 主实验至少形成：

1. C0–C3 observed fan-in 与 cache resolve time；
2. fan-in → delay-hit / redundant-work / reuse-realization curve；
3. S0 vs S1 vs S3 的 P50/P90/P99 TTFT 与 throughput；
4. representative C3 burst timeline；
5. cold-miss C3 vs gpu-ready C3 control。

如果 full hierarchy 可验证，再单独增加：

6. cold-miss vs cpu-restore C0/C3 comparison。

Cold-miss、gpu-ready 与 cpu-restore 结果不得混在同一 residency category 中。

## Experiment 4 outputs

至少形成：

1. delay-hit mechanism operating map；
2. balanced-batching operating map；
3. stall-hiding operating map；
4. full-scheduler operating map；
5. 4–6 个以内 frozen boundary points 的 validation；
6. hot-context concurrency rule；
7. final scheduler decision matrix。

每个 workload point 至少分类为：

- effective；
- mechanism-only；
- neutral；
- regressive；
- capacity-limited；
- partial / unsupported。

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
run metadata + capability/workload/calibration identifiers
```

所有 relative / normalized metrics 同时保留 underlying absolute measurements。

## Storage policy

大体积 profiler dump、模型权重和可重新生成的大型中间文件不提交到 Git。

需要长期保留的外部数据记录：external path/object identifier、checksum、generating run identifier、runtime/processing version 与 retention note。
