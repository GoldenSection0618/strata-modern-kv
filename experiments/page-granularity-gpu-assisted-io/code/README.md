# Code

本目录用于存放 Page Granularity and GPU-Assisted I/O 实验实现。

实现必须遵循 [`../docs/00-common-conventions.md`](../docs/00-common-conventions.md)。

## Required responsibilities

代码应覆盖：

- fixed request trace 与 prefix-boundary workload 构造；
- model × runtime × attention-backend capability validation；
- supported page-size discovery / validation；
- SGLang HiCache explicit configuration；
- warm GPU-resident reuse control；
- CPU-resident restore-state preparation；
- `direct` / `kernel` backend paired execution；
- HtoD restore 与 DtoH write-back 分离采集；
- actual operation / transfer behavior instrumentation；
- sustained bandwidth 与 matched reference bandwidth 测量；
- non-overlapped I/O stall measurement；
- prefill / decode interference profiling；
- end-to-end serving metrics；
- raw → processed → figures 的确定性处理。

## Runtime configuration rules

每个正式 run 必须显式写入配置或 metadata：

- model identifier / revision；
- runtime commit；
- attention backend；
- `page_size`；
- precision / cache dtype；
- GPU / CPU cache budgets；
- `hicache_io_backend`；
- `hicache_mem_layout`；
- `hicache_write_policy`；
- hybrid/recurrent-state tracking parameters；
- scheduler / overlap controls；
- hardware / NUMA / CPU-GPU topology；
- request trace identifier / seed。

不得依赖未记录的 runtime defaults。

如果使用非 SGLang runtime，代码必须分别记录 prefix-match granularity、physical cache block size、offload/transfer granularity 和 observed transfer size。不能把它们压缩成一个 generic `page_size` 字段。

## Experiment-specific rules

### Experiment 1

- Primary reuse curve 使用 warm GPU-resident reuse control 或经验证的等价 no-restore condition。
- Prefix lengths 必须包含与 candidate page sizes 不完全对齐的 boundary cases。
- Page-level hit、matched/reused tokens 与 reuse efficiency 分开记录。
- Cache budget 按 bytes / memory budget 控制，并记录实际 token/state capacity。

### Experiment 2

- Controlled I/O group 固定 logical HtoD payload bytes。
- `direct` backend、host layout、write policy 和 attention backend 固定。
- Configured page size 不能替代 observed operation / transfer granularity。
- HtoD restore 与 DtoH background traffic 分离。
- Reference bandwidth 使用 matched host-memory condition，不直接使用 theoretical PCIe peak。
- 每个 serving-level page-size point 同时生成 GPU-resident hit control 与 matched CPU-resident direct-restore run，用同 page-size pair 估计 restore-related penalty。

### Experiment 3

- 每个 `direct` / `kernel` pair 使用相同 logical state、page size、layout、write policy、cache state 与 request trace。
- Backend active path 必须由 runtime / profiler evidence 验证。
- Kernel backend 不要求与 direct backend 共享相同 transfer-operation semantics。共同比较口径是 logical bytes、elapsed time、bandwidth、stall 和 correctness。

### Experiment 4

- Controlled compute experiment 明确区分 compute-only、direct I/O、kernel I/O。
- Prefill 和 decode interference 分开实现。
- Kernel configuration 不按每个 end-to-end workload 单独调优。
- Net benefit 由实际 serving measurement 计算，不通过 profiler 时间项手工拼接。

## Data integrity

- 每个 repetition 生成独立 raw result，不覆盖旧结果。
- Invalid / unsupported / partial runs 保留并记录 reason。
- 不因为结果与预期不符而修改 workload 或过滤规则。
- 分析和绘图脚本只读取 raw / processed data，不把最终数字写死在代码中。
- 大型 profiler dump 不提交 Git。保留外部路径、run ID、checksum 或 manifest。

后续实现可按 `configs/`、`validation/`、`workloads/`、`runners/`、`profiling/`、`analysis/` 等职责拆分。只有代码规模需要时再创建子目录。
