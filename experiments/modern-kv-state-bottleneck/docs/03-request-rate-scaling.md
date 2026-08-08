# Experiment 3: Request-Rate Scaling / Concurrency Pressure

## 1. 实验目标

本实验用于研究在 context length 和 shared-prefix ratio 固定的情况下，随着 serving load 增长，现代模型的 cache/state loading 是否从局部开销演化为系统级瓶颈。

实验主要回答以下问题：

1. request rate 增长时，CPU-GPU state transfer、I/O stall 和 TTFT 如何变化。
2. 系统是否存在明显的负载阈值，在该阈值之后 state loading 开始造成排队、资源竞争和 TTFT 快速恶化。
3. Qwen3.5 与 Gemma 4 在高负载下的瓶颈演化是否一致。

这一实验关注负载放大效应。实验一研究 context scaling，实验二研究 reuse scaling，实验三研究同一 workload 在不同 serving pressure 下是否暴露新的 state bottleneck。

## 2. 实验对象

实验分别使用：

- Qwen3.5-9B
- Gemma 4 12B

两个模型保持各自原生 attention、cache 和 state 机制。

主实验继续使用 A100 40GB，与实验一和实验二保持相同硬件环境。

## 3. 固定 workload

实验固定一个具有代表性的 long-context reuse workload。

主实验设置：

- total context length 固定为 32K；
- shared-prefix ratio 固定为 50%；
- unique suffix 固定为 50%；
- output length 保持固定且较短；
- shared prefix 对应的 cache/state 在实验开始前已经位于 CPU cache 中。

因此，每个请求都具有：

```text
16K shared prefix
+
16K unique suffix
```

这一配置同时具有明显的 prefill computation 和 state loading，适合观察两类成本在高负载下的竞争关系。

如果 32K 在某个模型上无法形成稳定的完整负载曲线，则使用该模型能够稳定运行的最大公共 context length，但同一模型的所有 request-rate 点必须保持相同 workload。

## 4. 核心自变量

本实验将 request arrival rate 作为主要自变量。

不同时 sweep context length 和 shared-prefix ratio。

request rate 从明显低于系统处理能力的低负载开始，逐步提升到接近系统饱和，并继续增加到能够观察明显 queueing 的高负载区域。

正式实验选择约 6 到 8 个负载点，覆盖：

1. 低负载；
2. 中低负载；
3. 中等负载；
4. 接近饱和；
5. 饱和附近；
6. 高于饱和。

具体 requests/s 不预先固定，而是在两种模型上分别根据其实际 serving capacity 确定。

这样可以避免因两个模型的基础处理能力不同而导致不公平比较。

## 5. 负载范围确定

正式实验前首先进行一次短的 capacity calibration。

该阶段估计每个模型在当前 workload 下能够稳定处理的大致最大 request rate。

随后正式实验按照该 capacity 的相对比例设置负载，包括明显低于 capacity、约半负载、中高负载、接近 capacity、capacity 附近和超过 capacity 的配置。

最终比较时同时保留实际 requests/s 和相对于模型 capacity 的 normalized load。

这样能够避免仅仅因为 Qwen3.5 和 Gemma 4 的基础速度不同，就错误判断其中一个模型更容易发生状态瓶颈。

## 6. Concurrency 的处理

本实验不再单独进行完整的 concurrency sweep。

request arrival rate 作为主要负载控制变量，系统中的 active concurrency 随 workload 自然变化，并作为观测指标记录。

这样可以避免同时改变 request rate 与 concurrency 形成大规模二维实验，并避免重复验证相同的 saturation phenomenon。

同时设置一个足够高但固定的 concurrency limit，使正常负载范围内系统不会因为人为并发上限提前受到限制。

因此实验研究的是 arrival pressure 如何自然转化为 active concurrency、资源竞争和排队。

## 7. Cache 条件

主实验使用 warm hierarchical cache。

每组请求的 shared prefix 已经完成 cache/state 构建，并保存在 CPU 侧。

正式请求执行过程中需要：

1. 加载 shared-prefix state；
2. 对 unique suffix 执行 prefill；
3. 进入正常 decode。

所有 request-rate 配置保持相同 cache policy。

实验不改变 cache capacity，不人为制造 cache eviction。

这样可以将重点限制在多个请求同时需要 state loading 时产生的系统压力。

cache locality 和 eviction behavior 留给后续 scheduler/cache-locality 实验研究。

## 8. Baseline

实验保留一个 cold recomputation baseline。

该 baseline 在相同 workload 和 request-rate 条件下不加载已有 shared-prefix state，而是重新计算完整 context。

不需要在所有极端负载点完整重复一套实验，可以选择低负载、接近 saturation 和高负载三个代表性区间进行对照。

这一设计用于判断随着并发压力增加，hierarchical cache 相对于 recomputation 的优势是在扩大、保持稳定，还是逐渐消失。

## 9. 核心测量指标

### A. Throughput

记录：

- offered request rate；
- achieved throughput。

得到：

> Offered load → achieved throughput

当 request rate 继续增加而 throughput 不再明显增长时，可以确定系统进入 saturation 区域。

### B. Active concurrency

记录不同 request rate 下系统中实际同时处理的请求数量。

得到：

> Request rate → active concurrency

这一指标用于描述负载增加如何转化为同时存在的 context/state。

### C. Cache/state memory pressure

记录：

- GPU cache/state memory usage；
- CPU cache/state usage；
- workload 增长时的峰值状态占用。

重点观察高并发是否导致 GPU state residency 明显增长，并进一步产生显存压力。

### D. CPU-GPU traffic

记录单位时间内：

- CPU-GPU state transfer volume；
- sustained transfer bandwidth。

分别观察：

- traffic per request；
- traffic per second。

每请求 transfer volume 理论上应基本稳定，而单位时间 traffic 会随 request rate 增长。

如果单位时间需求不断增长，而实际 transfer throughput 接近平台期，则说明 I/O path 开始接近饱和。

### E. I/O stall

记录每个请求的 state-loading stall，并计算：

\[
\text{I/O Stall Ratio}
=
\frac{\text{I/O stall time}}
{\text{request service time}}
\]

重点观察这一比例是否随着 request rate 增长。

如果低负载下 I/O stall 较小，而高负载下迅速增加，则说明 bottleneck 不是单个请求的 state size 本身，而是并发 state movement 产生的资源竞争。

### F. Queueing delay

单独记录请求进入系统后到真正开始执行之间的等待时间。

这一指标必须与实际 execution latency 分开。

否则高负载下 TTFT 增加之后无法判断请求本身变慢了，还是只是前面的请求排队更多了。

### G. TTFT

记录：

- P50 TTFT；
- P90 TTFT；
- P99 TTFT。

这一实验相比实验一和实验二更需要关注 tail latency。

高负载下 state loading contention 即使对 median 影响有限，也可能首先表现为 P90/P99 的明显恶化。

## 10. TTFT 分解

将 TTFT 至少区分为：

\[
TTFT
=
Queueing
+
Prefill/Execution
+
State\ Loading/Stall
+
Other
\]

实验不要求构建极其细粒度的时间模型，但必须能够区分：

- queueing amplification；
- computation pressure；
- state/I/O pressure。

这是实验三能否真正解释 saturation 的关键。

## 11. 实验执行方式

每个 request-rate 配置持续运行足够长的稳定窗口。

正式统计之前设置 warm-up 阶段，使模型、cache 和 serving runtime 进入稳定状态。

每个负载点重复运行多次。

主要报告：

- P50；
- P90；
- P99；
- median throughput；
- 波动范围。

如果某一 request rate 下请求队列持续增长而无法回落，则将该点定义为 unstable overload，而不是继续无限等待所有请求完成。

## 12. 实验顺序

request rate 不只按照从低到高的单一顺序执行。

正式测试采用交替或随机顺序，例如：

```text
low → medium → low → high → medium → saturation
```

这样可以检查高负载运行之后是否存在残留状态，对后续实验产生系统性影响。

每个负载点之间恢复到统一的初始 cache 和系统状态。

## 13. 最终结果组织

建议形成五组核心图。

### 图 1：Load → Throughput

```text
Offered request rate → Achieved throughput
```

用于确定系统 saturation point。

### 图 2：Load → TTFT

同时绘制：

- P50；
- P90；
- P99。

用于观察 latency cliff 和 tail-latency amplification。

### 图 3：Load → CPU-GPU traffic

同时展示：

- requested transfer rate；
- achieved transfer bandwidth。

用于判断 I/O path 是否逐渐饱和。

### 图 4：Load → I/O stall / Queueing

分别展示：

- I/O stall；
- queueing delay。

用于确定 TTFT 恶化来自哪里。

### 图 5：Load → TTFT composition

选取低、中、接近 saturation、高负载四个代表点，对 TTFT 进行组成分析。

这张图用于解释系统为什么在某个负载以后开始明显恶化。

## 14. 结果判断逻辑

### 情况 A：I/O bottleneck 随负载显现

低负载下 state loading 成本较小，但随着 request rate 提高：

- CPU-GPU bandwidth 接近平台；
- I/O stall 快速增加；
- P90/P99 TTFT 明显恶化；
- throughput 最终饱和。

结论是：

> 现代模型的 state bottleneck 具有明显的 load-dependent 特征。单请求下问题可能并不严重，但在 serving concurrency 增长后会成为系统瓶颈。

这是对 Strata motivation 很重要的支持。

### 情况 B：主要由 computation saturation 导致

request rate 增长后系统进入 saturation，但：

- state transfer 没有接近瓶颈；
- I/O stall ratio 基本稳定；
- 主要增长来自 prefill execution 和 queueing。

结论是：

> 现代模型在该 workload 下的高负载瓶颈主要来自 computation，而不是 state movement。

这种情况下 Strata 类 I/O 优化对整体 capacity 的价值会受到限制。

### 情况 C：hierarchical cache 的优势随负载减弱

低负载下：

\[
TTFT_{warm} < TTFT_{cold}
\]

但随着负载提高，两者差距不断缩小。

结论是：

> prefix reuse 本身仍然有效，但 concurrent state loading 削弱了 hierarchical caching 在高负载 serving 中的实际收益。

这说明问题重点已经从“cache 有没有价值”转变为“cache 能否高效地被加载”。

### 情况 D：模型之间出现不同 saturation behavior

例如一个模型首先遇到 state-transfer saturation，而另一个首先遇到 computation saturation。

此时结论应限定为：

> 现代模型的 serving bottleneck 随负载增长具有明显 model-dependent behavior。

这一实验不能单独把差异归因于 attention architecture，后续模型泛化实验再结合两个模型的 state structure 进行解释。

## 15. 与前两个实验的关系

三组实验形成一个正交设计：

| 实验 | 主要变量 | 回答的问题 |
|---|---|---|
| Experiment 1 | Context length | context 变长是否放大 state bottleneck |
| Experiment 2 | Shared-prefix ratio | reuse 增长的计算收益是否被 loading cost 抵消 |
| Experiment 3 | Request rate | serving load 是否将 state cost 放大为系统瓶颈 |

因此实验三不再 sweep context length、prefix ratio、cache locality 或 scheduler policy。

三个实验共同形成以下因果链：

\[
\text{State size}
\rightarrow
\text{State reuse/loading}
\rightarrow
\text{Concurrent loading pressure}
\rightarrow
\text{TTFT / throughput degradation}
\]

这构成“现代模型上的 KV / 状态瓶颈画像”这一部分的主体证据。
