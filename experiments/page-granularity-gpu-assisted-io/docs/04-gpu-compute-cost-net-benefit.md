# Experiment 4: GPU Compute Cost and End-to-End Net Benefit

## 1. Objective

本实验定量评估 SGLang `kernel` GPU-assisted I/O 在恢复 fragmented I/O efficiency 的同时，对模型 prefill / decode computation 造成多少 GPU-side interference，并最终判断 Experiment 3 的 I/O benefit 在扣除 compute cost 后是否仍然形成正的 end-to-end net benefit。

本实验不再证明 kernel backend 能否提高 raw transfer bandwidth。该问题已经由 Experiment 3 回答。

所有配置遵循 [`00-common-conventions.md`](00-common-conventions.md)。

## 2. Primary causal question

本实验研究两条同时发生的路径：

```text
kernel I/O
    ↓
restore efficiency ↑
    ↓
non-overlapped I/O stall ↓

kernel I/O
    ↓
GPU resource contention / execution overlap
    ↓
model computation efficiency ?

两者共同决定
    ↓
end-to-end net benefit
```

最终判断必须来自实际 serving measurement，而不是把 profiler 中可能 overlap 的时间项手工相加减。

## 3. Experimental inputs

Experiment 4 不重新探索 page-size space。

直接复用 Experiment 3 的三个 representative page-size / workload points：

1. coarse / no-need control；
2. reuse-I/O trade-off point；
3. fine / fragmentation-dominated point。

I/O backend 比较继续使用：

```text
direct
kernel
```

Kernel backend 的实现配置在 Experiment 4 开始前固定。不得根据每个 end-to-end workload 的最终 speedup 单独调优。

## 4. Experimental structure

Experiment 4 分为两层。

### Experiment 4A: Controlled GPU interference

固定 model compute workload，在可控 I/O demand 下比较 compute-only、direct I/O 与 kernel I/O，测量 GPU-assisted I/O 对 prefill / decode 的直接干扰。

### Experiment 4B: End-to-End net benefit

将 Experiment 3 已确认的 I/O recovery 与 Experiment 4A 测得的 compute interference 放回相同 serving workload，通过实际 TTFT / completion time / throughput 判断最终净收益。

## 5. Experiment 4A: Comparison states

每个固定 compute workload 比较三种状态：

```text
A. compute only
B. compute + direct I/O
C. compute + kernel I/O
```

A 提供模型计算基线。

B 量化 standard-copy path 本身与模型计算并发时的影响。

C 量化 GPU-assisted kernel path 的总影响。

因此分析时既报告 C 相对于 A 的 total slowdown，也报告 C 相对于 B 的 incremental kernel cost。

## 6. I/O demand levels

I/O demand 设置 low、medium、high 三个代表性等级。

这些等级由 Experiment 3 的实际 restore behavior 定义，使用可复现的 target HtoD bytes / restore cadence 或等价 workload trace，而不是凭主观标签生成。

### Low

I/O 很少进入 critical path，用作 negative control。

### Medium

存在明确 restore stall，但 I/O 尚未完全主导 serving。该点是主要 trade-off 区域。

### High

大量 CPU-resident state 需要恢复，用于观察 kernel I/O 的最大 potential benefit 与 GPU interference 上限。

I/O demand 是 robustness dimension。它不替代 page-size point，也不允许同时任意改变 batch size、context length 或 scheduler policy。

## 7. Prefill interference experiment

Prefill 使用固定 input tokens、batch composition 和 execution configuration。

对于每个 I/O demand level，分别运行 compute-only、direct 和 kernel 三种状态。

记录：

- prefill execution time；
- prefill throughput；
- GPU kernel timeline；
- I/O/compute overlap interval；
- relevant GPU execution-resource / occupancy evidence；
- target HtoD bytes 和 achieved bandwidth。

Primary normalized metrics：

```text
prefill total slowdown
= kernel prefill time / compute-only prefill time

prefill incremental kernel cost
= kernel prefill time / direct-I/O prefill time
```

绝对 execution time 与 throughput 必须同时保留。

## 8. Decode interference experiment

Decode 使用固定 active request count、context condition 和 output-generation window。

对于每个 I/O demand level，同样比较 compute-only、direct 和 kernel 三种状态。

记录：

- decode throughput；
- per-token latency；
- decode-step execution time；
- GPU kernel timeline；
- I/O/compute overlap；
- target HtoD bytes 和 bandwidth。

Primary normalized metrics：

```text
decode total slowdown
= kernel decode latency / compute-only decode latency

decode incremental kernel cost
= kernel decode latency / direct-I/O decode latency
```

Prefill 与 decode 结果分开分析。不能根据一个 phase 的 interference 推断另一个 phase。

## 9. Optional kernel-resource subtest

只有当 pinned SGLang/kernel implementation 提供稳定、可记录且无需修改核心语义的 GPU-I/O resource control 时，才增加一个小规模 resource-budget sweep。

该子实验用于观察：

```text
more GPU resource for I/O
        ↓
I/O bandwidth recovery ↑
        vs
model compute slowdown ↑
```

如果 runtime 没有可靠的公开控制项，则不为了复刻历史 figure 强行修改 kernel 或引入不可维护的私有 knob。此时 Experiment 4 使用固定 kernel implementation，通过 low/medium/high I/O demand 测量实际 interference 即可。

## 10. Experiment 4B: End-to-End workloads

复用 Experiment 3 的 serving workloads。

### A. Coarse / low-I/O control

Direct I/O 已经较高效，用于检查 kernel 是否产生不必要 regression。

### B. Reuse-I/O trade-off workload

同时存在明显 reuse value 和 Experiment 3 已确认的 direct-path I/O penalty。该场景是最终结论的主要 operating point。

### C. Fine / high-fragmentation workload

Direct path 的 fragmentation penalty 明显，用于判断 kernel 的高 I/O benefit 是否会被 GPU interference 抵消。

不重新添加新的 workload family。

## 11. End-to-End paired protocol

对于每个固定 workload，至少比较：

```text
direct I/O
kernel I/O
```

Paired runs 保持一致：

- model / revision；
- runtime commit；
- attention backend；
- page size；
- cache dtype；
- GPU/CPU cache budget；
- host layout；
- write policy；
- scheduler / overlap policy；
- request trace；
- initial CPU cache state；
- random seed；
- repetition protocol。

每个 run 必须确认 effective reused tokens 和 logical restored state 可比。

## 12. End-to-End metrics

至少记录：

- HtoD restore bytes / bandwidth；
- DtoH background traffic；
- restore duration；
- non-overlapped I/O stall；
- prefill execution time / throughput；
- decode throughput / per-token latency；
- TTFT；
- request completion time；
- overall throughput；
- kernel/backend activity evidence。

GPU utilization 可以记录，但不能单独作为 compute interference 或 resource efficiency 的结论。

## 13. Net-benefit interpretation

### I/O benefit

Kernel 相对于 direct 减少的 critical-path restore stall，以及由此对应的 end-to-end improvement potential。

### Compute cost

Kernel path 相对于 direct / compute-only 引入的 prefill 或 decode slowdown。

### Net benefit

Direct 与 kernel serving run 的实际 end-to-end latency / throughput 差异。

机制解释可以写成：

```text
net benefit
≈ I/O stall reduction
  - added compute delay
  - other induced overhead
```

该式只用于解释。正式数字以 end-to-end measurement 为准。

## 14. Primary analyses

### Analysis A: I/O demand → prefill interference

展示 absolute prefill time / throughput、total slowdown 和 incremental kernel cost。

### Analysis B: I/O demand → decode interference

展示 absolute decode latency / throughput、total slowdown 和 incremental kernel cost。

### Analysis C: I/O recovery vs compute cost

在相同 operating point 上并列展示：

- bandwidth / stall recovery；
- prefill slowdown；
- decode slowdown。

该分析说明 optimization 是否只是把 bottleneck 从 I/O 移到 GPU compute。

### Analysis D: End-to-End net benefit

比较 direct 与 kernel 的 TTFT、completion time 和 throughput。

这是 Experiment 4 的最终判断依据。

## 15. Expected operating regions

最终将配置划分为：

- **Unnecessary region**：direct I/O 已足够高效，kernel 的 I/O benefit 小于或接近其额外成本；
- **Net-benefit region**：stall reduction 明显大于 compute interference，end-to-end performance 改善；
- **Interference-limited region**：kernel 恢复了 I/O，但 GPU compute slowdown 抵消大部分或全部收益。

如果 kernel 几乎不产生 measurable compute interference，也应明确报告，不人为制造 trade-off。

## 16. Validity checks

正式结果必须满足：

1. compute-only、direct、kernel 使用相同 model compute workload；
2. direct / kernel paired serving runs 的 effective reuse 与 logical restore 可比；
3. backend activity 有实际 evidence，不存在 silent fallback；
4. HtoD restore 与 DtoH background traffic 可区分；
5. prefill / decode interference 分开统计；
6. GPU utilization 不单独作为 interference 证据；
7. profiler instrumentation 不显著改变正常 runtime behavior；
8. overlap 时间项不重复累计；
9. kernel configuration 没有根据每个 E2E workload 单独 tuning；
10. final net benefit 由实际 serving measurement 得出；
11. partial / unsupported hybrid-state path 不报告为完整结果。

## 17. Final conclusion target

Experiment 4 最终回答：

> 在哪些 page-size / restore-pressure / serving operating points 上，kernel GPU-assisted I/O 减少的 non-overlapped I/O stall 足以覆盖其对 prefill 和 decode computation 的额外成本，并形成真实的 end-to-end net benefit。

四个实验最终形成完整证据链：

```text
Experiment 1
page size → effective reuse

Experiment 2
page size → observed fragmentation → direct-I/O penalty

Experiment 3
direct vs kernel → I/O recovery → stall reduction

Experiment 4
kernel I/O → compute interference → end-to-end net benefit / regression
```

本实验组最终目标是确定机制的适用边界，而不是证明 GPU-assisted I/O 始终更快。
