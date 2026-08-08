# Experiment 3: GPU-Assisted I/O Compensation

## 1. Objective

本实验验证 GPU-assisted I/O 是否能够在 Experiment 2 已确认存在 fragmented restore penalty 的配置中恢复 CPU→GPU transfer efficiency，并判断这种恢复是否减少 non-overlapped I/O stall、改善实际 serving performance。

SGLang 主路径将 backend comparison 明确定义为：

```text
baseline: --hicache-io-backend direct
assisted: --hicache-io-backend kernel
```

所有配置遵循 [`00-common-conventions.md`](00-common-conventions.md)。

## 2. Primary causal question

本实验建立：

```text
same page size
+ same logical state
+ same host layout
+ same write policy
+ same workload
        ↓
direct vs kernel I/O
        ↓
transfer efficiency / restore stall
        ↓
serving performance
```

Page size 不再进行完整 sweep。Experiment 3 的主要独立变量是 I/O backend。

## 3. Representative operating points

从 Experiments 1–2 的 processed results 预先选择三个 page-size points。

### A. Coarse / I/O-efficient control

选择 effective reuse 较低，但 `direct` backend 已具有较高 bandwidth utilization 的 coarse page point。

该点作为负对照，检验 kernel I/O 是否只在存在实际 I/O problem 时才有明显价值。

### B. Reuse-I/O trade-off point

选择已经获得明显 reuse improvement，同时 Experiment 2 开始出现 fragmented restore / bandwidth loss 的 page size。

该点是 Experiment 3 的主要 operating point。

### C. Fine / fragmentation-dominated point

选择 reuse 已接近饱和，但 `direct` path 的 observed fragmentation 与 bandwidth degradation 已明显的 fine page point。

该点用于测量 kernel I/O 的最大 compensation capability 和收益边界。

三个 points 必须在运行 Experiment 3 之前确定，不能根据 kernel backend 的结果重新选择。

## 4. Paired-comparison controls

每一对 `direct` / `kernel` run 必须保持一致：

- model 与 revision；
- hardware 与 CPU-GPU topology；
- runtime commit；
- attention backend；
- page size；
- precision 与 cache dtype；
- GPU cache budget 与 CPU HiCache budget；
- `hicache_mem_layout`；
- `hicache_write_policy`；
- hybrid/recurrent-state tracking parameters；
- scheduler / overlap configuration；
- request trace；
- initial cache residency；
- logically restored state；
- random seed 与 repetition protocol。

如果 pinned kernel implementation 还存在其他影响 GPU resource usage 的可配置参数，这些参数在 Experiment 3 中使用一个预先固定的配置并写入 metadata。它们的 computation cost 由 Experiment 4 评估，不在 Experiment 3 中针对端到端结果逐 workload 调优。

## 5. Experimental structure

Experiment 3 分为两个层次。

### Experiment 3A: Transfer-only compensation

在没有模型计算竞争的 controlled window 中比较 `direct` 与 `kernel` 的纯 restore efficiency。

### Experiment 3B: Serving-level compensation

在真实 prefix reuse serving 中比较两种 backend，验证 bandwidth recovery 是否转化为 critical-path stall reduction 和实际 serving benefit。

## 6. Experiment 3A: Transfer-only compensation

### 6.1 Purpose

该子实验直接回答：

> 在相同 logical HtoD restore workload 下，kernel backend 是否比 direct backend 更有效地利用 CPU→GPU transfer path。

没有并发 model computation，因此结果主要描述 I/O backend 本身的能力，而不是 GPU interference。

### 6.2 Transfer workloads

复用 Experiment 2A 已验证的 controlled I/O cases，不重新执行完整 transfer-volume sweep。

至少保留：

- medium logical transfer volume；
- large logical transfer volume；
- contiguous page range；
- fragmented page selection。

对于每个 representative page size，两种 backend 使用完全相同的 logical payload 与 source/destination state。

### 6.3 Host state

正式 measurement 前将目标 state 预置到 CPU tier。

Measurement window 只研究 CPU→GPU restore。GPU→CPU backup/write-back 不进入主要 bandwidth calculation。

## 7. Experiment 3A metrics

### 7.1 Actual HtoD payload bytes

确认 direct 与 kernel 实际恢复的 logical bytes 一致。

### 7.2 Sustained HtoD bandwidth

分别计算两种 backend 的 sustained bandwidth。

### 7.3 Bandwidth utilization

两种 backend 使用 Experiment 2 中定义的 matched reference methodology。

### 7.4 Restore completion time

比较完成同一 logical restore 所需的时间。

### 7.5 Backend execution evidence

记录足以证明 `direct` 与 `kernel` 路径真实生效的 runtime / profiler evidence。

对于 direct path，记录 copy operation / batching behavior。

对于 kernel path，记录 GPU-assisted transfer kernel 的 launch / activity evidence 和实际 payload。

不能要求两种 backend 使用同一种“transfer-size distribution”语义。Kernel backend 可能通过 GPU threads 执行 host-memory access，其 execution model 与标准 DMA copy 不同。正式比较的共同口径是 logical bytes、elapsed transfer interval、bandwidth 和 correctness。

## 8. Experiment 3B: Serving-level compensation

### 8.1 Primary workload

使用 Experiment 2B 的 controlled prefix-boundary serving workload 和相同 representative page-size points。

该设置保持前三个实验的因果链一致。

### 8.2 Robustness workload

增加 Experiment 2B 已使用的 mixed-reuse workload，验证 backend benefit 是否只存在于人工规则 workload。

不增加新的 context × load × locality 大矩阵。

### 8.3 Cache-state requirement

每个正式 run 必须验证：

- target reusable state 实际存在于 CPU tier；
- request 实际触发 restore；
- direct 与 kernel 命中的 logical prefix/state 相同；
- effective reused tokens 在 paired runs 中一致或处于预定义容差内。

如果 backend 切换改变了 reuse outcome，则该 pair 不再是纯 I/O backend comparison，必须标记 invalid 或单独解释。

## 9. Serving-level metrics

每个 run 至少记录：

- effective reused tokens / reuse efficiency；
- HtoD restore bytes；
- DtoH background bytes；
- backend execution evidence；
- sustained HtoD bandwidth；
- restore duration；
- non-overlapped I/O stall；
- prefill execution time / throughput；
- decode throughput / per-token latency；
- TTFT；
- request completion time；
- overall throughput。

Prefill/decode metrics 在本实验中用于确认 serving consequence。GPU-side compute interference 的专门分解留给 Experiment 4。

## 10. Primary analyses

### Analysis A: Backend → bandwidth recovery

在每个 page-size point 上比较：

```text
direct bandwidth
vs
kernel bandwidth
```

同时报告绝对 bandwidth 与 normalized utilization。

可以计算：

```text
bandwidth recovery ratio = kernel bandwidth / direct bandwidth
```

但不能只报告 ratio 而隐藏绝对值。

### Analysis B: Backend → restore latency

比较相同 logical restore 的 completion time。

如果 bandwidth 上升但 restore completion time 没有改善，需要检查 fixed overhead、batching、measurement window 或其他 bottleneck。

### Analysis C: Restore latency → non-overlapped stall

在 serving run 中比较两种 backend 的 non-overlapped I/O stall。

只有 stall 真正减少，I/O recovery 才进入 serving critical path。

### Analysis D: Stall → serving benefit

比较 TTFT、request completion time 和 throughput。

Kernel backend 的 end-to-end improvement 必须能够与 restore/stall evidence 对应。

## 11. Expected operating regions

Experiment 3 最终区分：

- **No-need region**：direct path 已高效，kernel 的额外 transfer benefit 很小；
- **Effective-compensation region**：kernel 显著恢复 bandwidth 并减少 non-overlapped stall；
- **Residual-bottleneck region**：kernel 改善 I/O 后，end-to-end performance 仍受计算、调度或其他 bottleneck 限制。

这些 region 是 Experiment 4 的输入，不是本实验对 GPU compute cost 的最终判断。

## 12. Validity checks

正式结果必须满足：

1. direct / kernel pair 的 model outputs 数值一致；
2. logical restored state 与 effective reused tokens 一致；
3. page size、attention backend、layout、write policy 和 cache budget 不变；
4. direct 与 kernel backend 都有实际 execution evidence，不存在 silent fallback；
5. HtoD restore 与 DtoH background traffic 可区分；
6. transfer-only run 没有并发 model computation；
7. serving-level performance difference 能与 restore/stall behavior 对应；
8. kernel resource configuration 没有根据每个 workload 的最终 speedup 单独调优；
9. unsupported / partial hybrid-state restore 不报告为完整 serving result。

## 13. Interpretation boundaries

如果 kernel 提高 transfer-only bandwidth，但 serving stall 不变，不能声称存在 end-to-end I/O benefit。

如果 kernel 减少 stall，但 throughput / TTFT 改善有限，应解释为 bottleneck migration，而不是 optimization failure。

如果 direct path 在 fine page 下已经通过 runtime aggregation 达到较高 efficiency，kernel benefit 很小是有效结论，并说明现代 runtime 已部分消解 Strata 当年的 I/O mechanism。

本实验不以 GPU utilization 上升作为正面证据，也不在这里判断 GPU resource contention 是否值得。该问题由 Experiment 4 单独处理。

## 14. Final conclusion target

Experiment 3 最终回答：

> 在相同 page granularity、cache state 和 serving workload 下，SGLang kernel GPU-assisted I/O 相对于 direct standard-copy path 能恢复多少 CPU→GPU transfer efficiency，这种 recovery 有多少真正转化为 non-overlapped stall reduction 和 serving benefit。

Experiment 4 在这些 representative points 上继续计算 GPU compute cost 与最终净收益。
