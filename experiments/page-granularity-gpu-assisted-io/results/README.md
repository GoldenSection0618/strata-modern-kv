# Results

本目录用于存放 Page Granularity and GPU-Assisted I/O 实验结果。

结果保持三层可追溯结构：

```text
raw measurements
    ↓
processed / aggregated data
    ↓
figures and tables
```

所有结果遵循 [`../docs/00-common-conventions.md`](../docs/00-common-conventions.md)。

## 1. Raw results

每次 run 的 raw output 与 metadata 独立保存，不被处理脚本覆盖。

Metadata 至少包含：

### Identity

- experiment ID / sub-experiment ID；
- run ID；
- repetition index；
- timestamp；
- validity status：`valid` / `invalid` / `partial` / `unsupported`；
- invalid / partial / unsupported reason。

### Model and runtime

- model identifier 与 revision；
- serving runtime version / commit；
- attention backend；
- precision / quantization；
- cache dtype；
- hybrid/recurrent-state tracking configuration。

### Hardware

- GPU model / form factor；
- CPU model；
- CPU-GPU topology；
- NUMA placement；
- driver；
- CUDA/runtime version；
- pinned-host-memory condition。

### Cache and granularity

- configured `page_size`；
- GPU cache memory budget；
- CPU HiCache budget；
- runtime-observed token/state capacity；
- allocator padding / relevant capacity counters；
- cache replacement / eviction policy。

如果 runtime 将不同 granularity 解耦，还必须分别记录：

- prefix-match granularity；
- physical cache block size；
- recurrent-state checkpoint / tracking granularity；
- offload / transfer granularity configuration；
- observed transfer/operation size statistics。

### HiCache / I/O controls

- `hicache_io_backend`；
- `hicache_mem_layout`；
- `hicache_write_policy`；
- scheduler / overlap configuration；
- any kernel-resource configuration exposed by the pinned implementation。

### Workload

- request trace identifier；
- random seed；
- context length；
- output length；
- logically reusable prefix length / ratio；
- prefix boundary offset / alignment descriptor；
- cache-residency initialization；
- arrival / concurrency condition。

## 2. Raw measurement fields

按实验需要保存以下 measured fields。

### Reuse

- logically reusable prefix tokens；
- effective reused tokens；
- reuse efficiency；
- page/block hit counters；
- GPU / CPU residency / hit counters；
- cache occupancy / eviction supporting counters。

### I/O

- logical restore bytes；
- actual HtoD payload bytes；
- DtoH backup/write-back bytes；
- copy / I/O operation count；
- observed operation / transfer-size statistics；
- direct/kernel backend activity evidence；
- transfer interval；
- sustained HtoD bandwidth；
- matched reference bandwidth；
- bandwidth utilization；
- restore duration；
- non-overlapped I/O stall。

### Compute / serving

- prefill execution time；
- prefill throughput；
- decode throughput；
- per-token decode latency；
- TTFT；
- request completion time；
- overall throughput；
- profiler / GPU execution summary needed to support interference analysis。

## 3. Processed results

Processed data 由脚本从 raw measurements 确定性生成。

任何 filtering、outlier handling 或 aggregation rule 必须版本化并能够追溯到具体 raw runs。

### Experiment 1

至少汇总：

- page size → effective reused tokens；
- page size → reuse efficiency；
- prefix boundary / overlap robustness；
- context-length robustness；
- cache-pressure occupancy / eviction supporting results；
- coarse / transition / reuse-saturated region summary。

### Experiment 2

至少汇总：

- page size → observed operation / transfer granularity；
- fragmentation → sustained bandwidth；
- bandwidth utilization；
- restore duration；
- non-overlapped I/O stall；
- joint reuse-I/O trade-off region。

Controlled I/O 与 Serving-level results 分开聚合。

### Experiment 3

至少汇总 matched `direct` / `kernel` pairs：

- absolute bandwidth；
- bandwidth recovery ratio；
- restore duration；
- non-overlapped stall；
- TTFT / completion time / throughput；
- backend active-path validation status。

### Experiment 4

至少汇总：

- compute-only / direct / kernel prefill performance；
- compute-only / direct / kernel decode performance；
- total slowdown；
- incremental kernel cost；
- I/O stall reduction；
- direct vs kernel end-to-end net benefit；
- unnecessary / net-benefit / interference-limited region summary。

## 4. Figures and tables

图表只从 processed data 生成，不手工录入最终数值。

核心图表必须能够从 figure/table 追溯到：

```text
processed dataset
    ↓
raw run IDs
    ↓
exact runtime + model + hardware + workload config
```

Normalized metrics 必须保留对应 absolute values。

Bandwidth figures 必须注明 reference bandwidth 的测量条件，不能只写 theoretical PCIe peak。

Page-size figures 必须注明 attention backend 和 page-size support set，避免把不同 backend 的点画在同一条因果曲线上。

## 5. Large artifacts

以下内容默认不提交 Git：

- raw profiler dumps；
- Nsight traces；
- 大型 server logs；
- model weights；
- 可重新生成的大型中间文件。

需要保留时提交小型 manifest，记录外部路径、run ID、文件摘要/checksum 和生成命令。
