# Experiment 4: GPU Compute Cost and End-to-End Net Benefit

## 1. Objective

本实验定量评估 GPU-assisted I/O 在恢复 fragmented I/O efficiency 的同时占用了多少 GPU execution resource，以及这种资源竞争是否干扰模型的 prefill 与 decode。实验最终判断 Experiment 3 中观察到的 I/O benefit 在扣除 GPU computation interference 后是否仍然具有实际系统价值。

本实验不再研究 GPU-assisted I/O 是否能够恢复带宽。该问题由 Experiment 3 回答。本实验只研究 GPU-assisted I/O 的计算代价以及最终 end-to-end net benefit。

核心机制链为：

```text
GPU-assisted I/O
        ↓
I/O efficiency ↑
        ↓
non-overlapped I/O stall ↓

同时

GPU resource contention ↑
        ↓
model computation efficiency ↓

最终决定

end-to-end net benefit
```

## 2. Research questions

本实验回答以下问题：

1. GPU-assisted I/O 是否会与模型计算竞争 GPU execution resource；
2. 这种 interference 对 prefill 和 decode 的影响是否不同；
3. interference 是否随 I/O intensity 增强；
4. Experiment 3 中 bandwidth 和 I/O stall 的改善能否抵消模型计算性能下降；
5. GPU-assisted I/O 在什么 workload 下具有正的 end-to-end net benefit；
6. 是否存在 I/O benefit 已经趋于饱和，但 GPU computation cost 继续增加的 operating region。

## 3. Experimental variables

本实验主要比较以下三个维度：

- I/O backend：baseline I/O 与 GPU-assisted I/O；
- I/O intensity：低、中、高三个代表性等级；
- computation phase：prefill、decode 和完整 serving。

Page size 不再进行完整 sweep。本实验直接复用 Experiment 3 已确定的 representative operating points，重点保留一个 fragmentation 较弱的对照点、一个 GPU-assisted I/O 收益明显的 trade-off 点，以及一个 fragmentation 较强的压力点。

同一比较组中保持以下条件一致：

- model 与 model revision；
- hardware；
- precision；
- cache capacity；
- cache policy；
- scheduler configuration；
- request sequence；
- input/output length；
- concurrency；
- CPU memory configuration。

## 4. Experiment structure

Experiment 4 分为两层：

1. Controlled GPU interference experiment；
2. End-to-End Net Benefit experiment。

第一层隔离 GPU-assisted I/O 对模型计算本身的影响。第二层将计算代价与 Experiment 3 已确认的 I/O benefit 放回同一 serving workload 中，判断最终系统性能是改善、基本不变还是退化。

## 5. Experiment 4A: Controlled GPU interference

### 5.1 Goal

Controlled experiment 在固定模型计算 workload 的条件下比较以下三种运行状态：

```text
model computation only
model computation + baseline I/O
model computation + GPU-assisted I/O
```

该设计用于区分模型自身的正常计算性能、普通 I/O 引入的影响，以及 GPU-assisted I/O 额外产生的 GPU-side interference。

### 5.2 Prefill interference

Prefill 使用固定输入长度和固定 batch configuration。

每组配置分别执行纯 prefill、与 baseline I/O 并行的 prefill，以及与 GPU-assisted I/O 并行的 prefill。I/O workload 复用 Experiment 3 已验证的 transfer pattern。

实验在保持 prefill workload 不变的情况下逐步提高 I/O intensity。

主要记录：

- prefill throughput；
- prefill execution time；
- GPU compute utilization；
- model kernel execution behavior；
- I/O 与 model computation 的 overlap；
- GPU-assisted I/O 活跃期间的 compute slowdown。

Prefill slowdown 以同一 workload 的 compute-only baseline 为基准定义：

```text
prefill slowdown = assisted-I/O prefill time / compute-only prefill time
```

### 5.3 Decode interference

Decode 使用固定 active request 数、固定 context condition 和固定 output generation window。

每组配置分别执行 decode only、decode + baseline I/O，以及 decode + GPU-assisted I/O。I/O intensity 同样覆盖低、中、高三个代表性等级。

主要记录：

- decode throughput；
- token generation rate；
- per-token latency；
- model kernel execution behavior；
- GPU-assisted I/O 活跃区间中的 decode slowdown。

Decode slowdown 以同一 workload 的 compute-only baseline 为基准定义：

```text
decode slowdown = assisted-I/O decode latency / compute-only decode latency
```

Prefill 与 decode 必须分开测量。两者具有不同的 GPU execution pattern，不能根据一个阶段的 interference 推断另一个阶段。

## 6. I/O intensity design

I/O intensity 不作为新的完整 sweep 维度，而是从前面实验中选择三个有明确系统含义的代表性状态。

### Low I/O intensity

CPU→GPU restoration 较少，I/O 不构成主要 bottleneck。该组作为负对照，用于检查 GPU-assisted I/O 是否在低收益场景引入不必要的 compute cost。

### Medium I/O intensity

I/O 已形成可观测 stall，但尚未完全主导 serving。该区域用于重点比较 I/O stall reduction 与 GPU compute slowdown，是最重要的 trade-off operating region。

### High I/O intensity

大量 CPU-resident cache/state 需要恢复。该区域用于观察 GPU-assisted I/O 的最大 I/O benefit、最大 GPU interference，以及系统是否发生新的 GPU-side bottleneck。

## 7. Controlled interference metrics

### Model computation throughput

分别记录 prefill throughput 与 decode throughput，用于判断 GPU-assisted I/O 是否直接损害模型计算效率。

### Compute slowdown

分别计算 prefill slowdown 与 decode slowdown，并统一使用同一 workload 的 compute-only run 作为基准。

### GPU execution overlap

记录 GPU-assisted I/O 与模型 kernel 的时间重叠关系，确认 I/O 工作是否实际与 computation 并发、并发期间 model kernel 是否减速，以及 GPU-assisted operation 是否出现明显 serialization。

### GPU resource pressure

记录能够反映 GPU execution pressure 的 profiler 指标。GPU utilization 不能单独作为 compute interference 的充分证据，必须与 model throughput、kernel execution behavior 和 overlap 情况联合解释。

## 8. Experiment 4B: End-to-End Net Benefit

### 8.1 Goal

End-to-End experiment 将 I/O stall reduction 和 GPU computation slowdown 放回同一 serving workload 中比较，直接判断 GPU-assisted I/O 是否值得启用。

### 8.2 Workload selection

实验复用 Experiment 3 的 representative workloads，不重新设计新的 workload family。

#### A. Low-fragmentation / low-I/O workload

Baseline I/O 已较高效。该组用于检查 GPU-assisted I/O 是否产生不必要的 regression。

#### B. Trade-off workload

该 workload 同时具有明显 cache reuse、明显 fragmented I/O，以及 Experiment 3 已确认的 bandwidth recovery。该组是 Experiment 4 的主要场景。

#### C. High-fragmentation workload

该 workload 具有较高 I/O pressure。该组用于判断 GPU-assisted I/O 在高压力条件下是否仍然保持正净收益，或是否因为 GPU interference 出现收益上限。

## 9. End-to-End comparison protocol

每个 workload 至少运行 baseline I/O 与 GPU-assisted I/O 两种配置。

两种配置使用完全相同的：

- request trace；
- initial cache state；
- cache capacity；
- page size；
- scheduler；
- random seed。

所有配置使用统一 warm-up 方式并进行多次重复运行。

每次 run 同时记录：

- TTFT；
- request completion time；
- overall throughput；
- prefill throughput；
- decode throughput；
- per-token decode latency；
- non-overlapped I/O stall；
- GPU compute slowdown；
- GPU-assisted I/O active time。

## 10. Net-benefit interpretation

本实验明确区分三个概念。

### I/O benefit

GPU-assisted I/O 相对于 baseline 减少的 critical-path I/O stall。

### Compute cost

GPU-assisted I/O 引起的模型计算额外时间或 throughput degradation。

### Net benefit

最终 end-to-end latency 或 throughput 的实际改善。

机制上可以表示为：

```text
Net benefit
≈ I/O stall reduction
  - additional computation delay
  - other induced overhead
```

该关系仅用于解释。正式结果必须以实际 end-to-end measurement 为准，不能把 profiler 中存在 overlap 的时间项直接相加或相减构造最终 latency。

## 11. Analysis logic

实验重点判断 GPU-assisted I/O 占用的 GPU resource 是否比它消除的 I/O stall 更昂贵。

可能出现以下几类结果：

1. I/O stall 显著下降，compute slowdown 较小，end-to-end performance 明显改善。该区域属于明确的 net-benefit region。
2. I/O stall 显著下降，compute slowdown 同样明显，end-to-end performance 仅小幅改善或基本不变。该结果说明 bottleneck 从 I/O 向 GPU computation 迁移。
3. I/O stall 改善有限，compute slowdown 明显，end-to-end performance 下降。该 workload 不适合启用 GPU-assisted I/O。
4. GPU-assisted I/O 基本不造成 observable compute interference。该结果说明当前 GPU workload 与 I/O execution 能够较好共存。

## 12. Required result views

正式分析至少形成以下四类结果。

### Figure 1: I/O intensity → Prefill slowdown

比较 baseline I/O 与 GPU-assisted I/O 下 prefill computation efficiency 随 I/O pressure 的变化。

### Figure 2: I/O intensity → Decode slowdown

比较 decode 对 GPU-assisted I/O interference 的敏感程度。

### Figure 3: I/O stall reduction vs. compute slowdown

将 Experiment 3 的 I/O benefit 与 Experiment 4 的 compute cost 放在同一 operating-point framework 中比较。

### Figure 4: Workload → End-to-End performance

比较 TTFT、request completion time 与 overall throughput，作为最终 net-benefit 证据。

## 13. Validity checks

正式结果必须满足以下条件：

1. baseline 和 GPU-assisted 配置使用完全相同的 workload；
2. cache hit 和 actual restored state 在可比配置间保持一致；
3. GPU-assisted I/O 必须确认实际处于 active path，而不是 fallback 到 baseline；
4. prefill 与 decode interference 分开统计；
5. GPU utilization 不单独作为 compute interference 证据；
6. model slowdown 必须由实际 throughput 或 execution time 支持；
7. profiler measurement 不得显著改变正常 serving behavior；
8. I/O stall 与 computation overlap 的时间不能重复累计；
9. end-to-end net benefit 必须来自实际 serving measurement，而不是仅由 microbenchmark 推算。

## 14. Final conclusion target

Experiment 4 最终需要回答：

> 在什么 page granularity、I/O pressure 和 serving workload 下，GPU-assisted I/O 减少的 I/O stall 大于它造成的 GPU computation cost。

最终应将 operating region 划分为：

- **Unnecessary region**：I/O 本身不是主要瓶颈，不值得引入 GPU-assisted I/O；
- **Net-benefit region**：I/O compensation 明显大于 GPU computation cost；
- **Interference-limited region**：GPU resource contention 抵消了主要 I/O benefit。

四个实验最终形成完整证据链：

```text
Experiment 1
page size ↓
→ effective cache reuse ↑

Experiment 2
page size ↓
→ actual I/O fragmentation ↑
→ bandwidth efficiency ↓

Experiment 3
GPU-assisted I/O
→ bandwidth recovery
→ non-overlapped I/O stall ↓

Experiment 4
GPU-assisted I/O
→ GPU computation interference
→ end-to-end net benefit / regression
```

该实验组最终不以证明 GPU-assisted I/O 一定更快为目标，而是确定其适用边界和真实系统净收益。