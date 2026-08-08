# Experiment 2: Page Size vs. I/O Efficiency

## 1. Objective

本实验定量评估 page granularity 对 CPU→GPU cache/state restore efficiency 的影响，验证较小 page 是否真实产生更细碎的 transfer behavior、降低 sustained bandwidth，并判断这种 degradation 是否进入 serving critical path。

本实验只研究：

```text
page size
    ↓
observed transfer fragmentation
    ↓
I/O efficiency
    ↓
non-overlapped restore stall
```

GPU-assisted `kernel` backend 留到 Experiment 3。本实验使用固定的 standard-copy baseline。

所有配置遵循 [`00-common-conventions.md`](00-common-conventions.md)。

## 2. Primary backend

SGLang 主路径使用：

```text
--hicache-io-backend direct
```

该 backend 作为 standard CUDA memory-copy baseline。

`hicache_mem_layout`、`hicache_write_policy`、attention backend 和 overlap policy 必须显式指定并在整个 Experiment 2 中固定。不得依赖 runtime default。

Host layout 必须选择一个同时能用于 Experiment 3 的 `direct` 与 `kernel` paired comparison 的配置。若某个 layout 只适用于其中一个 backend，则不能用它构造公平的 backend comparison。

## 3. Independent variable

主要自变量是与 Experiment 1 相同的 SGLang `page_size`。

只使用通过 Experiment 1 capability gate、且由同一 attention backend 支持的 page-size candidates。

如果替换为把 reuse matching 与 physical/transfer granularity 解耦的 runtime，则 Experiment 2 不再沿用 Experiment 1 的 generic page-size axis。此时独立变量改为 physical / transfer granularity，并单独记录 prefix-match granularity。

## 4. Controlled variables

同一比较组保持以下条件固定：

- model 与 revision；
- hardware 与 CPU-GPU topology；
- runtime commit；
- attention backend；
- precision 与 cache dtype；
- CPU pinned-memory condition；
- GPU cache budget 与 CPU HiCache budget；
- `hicache_io_backend=direct`；
- `hicache_mem_layout`；
- `hicache_write_policy`；
- scheduler / overlap configuration；
- hybrid/recurrent-state tracking parameters；
- transfer direction；
- request trace；
- repetition protocol。

Controlled I/O 子实验另外固定 logical transfer bytes 与 access pattern。

## 5. Experimental structure

Experiment 2 分为两个层次。

### Experiment 2A: Controlled I/O fragmentation

固定 logical data movement，只改变 page granularity，直接验证 fragmentation 对 I/O efficiency 的影响。

### Experiment 2B: Serving-level validation

使用真实 CPU-resident cache hit，保留 page size 对 actual reuse / transfer pattern 的自然影响，判断 microbenchmark 中的 I/O degradation 是否真正形成 non-overlapped serving stall。

两层结果必须分开报告。

## 6. Experiment 2A: Controlled I/O fragmentation

### 6.1 Purpose

Controlled I/O experiment 需要最大限度隔离：

```text
same logical bytes
+ same host layout
+ same backend
+ same direction
+ different page size / page count
```

这样 observed bandwidth difference 才能主要归因于 page/operation granularity，而不是总数据量变化。

### 6.2 Transfer volume

设置 small、medium、large 三个代表性的 logical transfer volume。

具体字节数由目标 model 的实际 cache/state footprint 与可稳定测量范围确定。每一个 transfer-volume group 内，不同 page size 的 logical payload bytes 必须一致。

Medium / large transfer 是主要结果。Small transfer 主要用于确认 fixed overhead 区域。

### 6.3 Access patterns

至少包含两种模式。

#### A. Contiguous page range

恢复一段连续逻辑 page range。

该组用于建立相对理想的 transfer baseline，并观察即使逻辑访问连续，细 page 是否因 operation granularity 产生额外 overhead。

#### B. Fragmented page selection

恢复离散 page 集合，同时保持 logical payload bytes 与 contiguous group 可比。

该组模拟 prefix/cache reuse 中实际需要加载的非连续 state，并用于放大真实 fragmentation behavior。

### 6.4 Read-path isolation

Experiment 2A 主要研究 CPU→GPU restore。

正式 measurement window 前先准备好 CPU-resident state。GPU→CPU backup/write-back traffic 不得混入 read bandwidth measurement。

如果 runtime 无法完全停止后台 write traffic，则必须独立采集 HtoD 与 DtoH bytes / events，并从 restore analysis 中区分。

## 7. Experiment 2B: Serving-level validation

### 7.1 Purpose

Serving-level validation 检查：

> Controlled I/O 中观察到的 bandwidth degradation 是否足以增加实际 cache restore stall，并影响用户可见 serving performance。

该阶段不要求不同 page size 的 actual transferred bytes 相同，因为 page size 对 effective reuse 本身的影响就是真实系统行为的一部分。

### 7.2 Page-size points

从 Experiment 1 的同一 primary workload 中选择三个 page-size regions：

1. **coarse / reuse-limited point**；
2. **transition point**；
3. **reuse-saturated fine-page point**。

这三个点由 Experiment 1 processed results 预先确定，而不是根据 Experiment 2 的 bandwidth 结果重新挑选。

### 7.3 Workloads

Primary serving validation 使用 Experiment 1 的 controlled prefix-boundary workload，以保持因果链清晰。

随后增加一个 mixed-reuse workload 作为 robustness check。

不需要重新运行 Experiment 1 的全部 context × prefix × cache-pressure 矩阵。

### 7.4 Cache residency

正式测量必须验证目标 reusable state 位于 CPU tier，并且 request 确实触发 CPU→GPU restore。

没有实际 CPU-resident hit 的 run 不进入 serving I/O analysis。

## 8. Core metrics

### 8.1 Actual HtoD payload bytes

记录目标 restore 的实际 CPU→GPU payload bytes。

这是 bandwidth denominator 的基础，同时用于确认不同 controlled configurations 的 logical bytes 是否匹配。

### 8.2 Transfer / operation count

记录完成一次 logical restore 所需的 copy operations、I/O batches、kernel/copy launches 或 runtime 可观测的等价事件数量。

### 8.3 Observed transfer-size distribution

记录实际操作粒度，而不是直接把 configured page size 当作 transfer size。

如果 runtime 自动 coalesce / batch 多个 page，正式结论必须基于 observed behavior。

### 8.4 Sustained host→GPU bandwidth

```text
sustained bandwidth = target HtoD payload bytes / target transfer interval
```

Measurement window 必须只覆盖目标 restore path，并记录 overlap policy。

### 8.5 Matched reference bandwidth

Reference bandwidth 使用同一硬件、同一 host-memory condition、同一 direction 下的大块连续 transfer 测量。

不得使用理论 PCIe peak 直接代替 matched runtime reference。

### 8.6 Bandwidth utilization

```text
bandwidth utilization = sustained bandwidth / matched reference bandwidth
```

### 8.7 Restore duration

记录完成目标 cache/state restore 的实际持续时间。

### 8.8 Non-overlapped I/O stall

Serving-level validation 记录不能被 model computation overlap 隐藏的 restore stall。

Raw transfer duration 与 non-overlapped stall 分开保存。

### 8.9 Serving supporting metrics

记录 TTFT、prefill time / throughput、overall throughput，用于判断 I/O degradation 是否进入系统 critical path。

Experiment 2 不将这些指标解释为 GPU-assisted I/O benefit。

## 9. Execution procedure

### 9.1 Controlled I/O

基本实验单元为：

```text
fixed logical bytes
× fixed access pattern
× fixed direct backend
× page-size sweep
```

每个配置：

1. 初始化 host/device buffers 与 cache metadata；
2. 完成 warm-up；
3. 验证目标 bytes 和 page indices；
4. 执行正式 transfer measurement；
5. 重复多次；
6. 随机化或平衡 page-size execution order；
7. 保留 raw trace 和 invalid reason。

### 9.2 Serving-level

每个 representative page-size point 使用相同 request trace、cache budget、initial CPU residency 和 runtime controls。

每个 run 同时记录 reuse、HtoD restore、DtoH background traffic、stall 和 serving metrics，使机制链能够在同一次 execution 中对应。

## 10. Primary analyses

### Analysis A: Page size → observed fragmentation

联合分析：

- configured page size；
- operation count；
- observed transfer-size distribution。

只有 observed I/O behavior 随 page size 变细而真实变碎，才能认为 fragmentation mechanism 成立。

### Analysis B: Fragmentation → bandwidth

绘制 observed transfer granularity / operation count 与 sustained bandwidth、bandwidth utilization 的关系。

该图比简单 `page size → bandwidth` 更直接支持机制解释。

### Analysis C: Page size → restore stall

在 serving-level run 中比较 restore duration 与 non-overlapped I/O stall。

Bandwidth loss 只有在 non-overlapped stall 同步增加时，才进一步构成实际 serving bottleneck evidence。

### Analysis D: Joint reuse-I/O trade-off

将 Experiment 1 的 effective reuse / reuse efficiency 与 Experiment 2 的 bandwidth utilization / non-overlapped stall 放在相同 page-size operating points 上。

目标是识别：

- **reuse-limited coarse region**；
- **balanced trade-off region**；
- **fragmentation-dominated fine region**。

## 11. Validity checks

正式分析前必须确认：

1. 同一 Controlled I/O group 的 logical payload bytes 一致；
2. attention backend 在整个 page-size sweep 中不变；
3. host memory layout、write policy 与 direct I/O backend 不变；
4. actual HtoD 与 DtoH traffic 能够区分；
5. actual operation count / transfer granularity 可观测；
6. runtime batching/coalescing behavior 已记录；
7. bandwidth reference 使用匹配的 host-memory condition；
8. serving run 确实触发目标 CPU-resident restore；
9. raw transfer duration 与 non-overlapped stall 不互相替代；
10. invalid / unsupported runs 保留 reason。

## 12. Interpretation boundaries

本实验可以验证 page granularity 是否通过实际 fragmented I/O 降低 standard-copy efficiency，并判断该损失是否进入 serving critical path。

本实验不能仅凭 configured page size 宣称 fragmentation 已发生。

本实验不能证明 `kernel` GPU-assisted I/O 能够修复该问题，因为该 backend 尚未作为比较变量引入。

如果小 page 提高 reuse 但 actual transfer 已被 runtime 自动聚合、bandwidth 没有显著下降，则应该得出“现代 runtime 已缓解原始 fragmentation mechanism”的结论，而不是强行复现 Strata 的历史结果。

## 13. Final conclusion target

Experiment 2 最终回答：

> 在 Experiment 1 确认有 reuse value 的 page-size region 中，哪些 granularity 会真实造成 fragmented CPU→GPU restore、带宽利用率下降和 non-overlapped I/O stall。

这些结果直接确定 Experiment 3 需要比较 `direct` 与 `kernel` I/O 的 representative operating points。
