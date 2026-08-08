# Experiment 3: Request-Rate Scaling / Concurrency Pressure

## 1. 实验目标

本实验用于研究在 context length 和 shared-prefix ratio 固定的情况下，随着 request arrival pressure 增长，现代模型的 cache/state restore cost 是否会从单请求局部开销放大为系统级 bottleneck。

实验主要回答三个问题：

1. offered request rate 增长时，CPU-GPU state traffic、non-overlapped I/O stall、queueing 和 TTFT 如何变化；
2. 系统是否存在明确的 saturation region，在该区域 state restore contention 开始显著放大 tail latency 或限制 throughput；
3. Qwen3.5 与 Gemma 4 在相对于各自 capacity 的相似负载水平下，瓶颈演化是否一致。

本实验的主要自变量是 request arrival rate。Active concurrency 是由 arrival pressure 与 service time 共同产生的观测量，不再作为第二个独立 sweep 变量。

## 2. 实验对象

实验分别使用 Qwen3.5-9B 与 Gemma 4 12B。

主实验继续使用 A100 40GB，并保持与 Experiments 1-2 一致的模型 precision、runtime 与 cache policy。

正式运行前必须确认 CPU-resident hit 对两个模型的目标 cache/state groups 均有效。

## 3. 固定 workload

主实验固定为 long-context reuse workload：

- total context length：32,768 tokens；
- shared prefix：16,384 tokens；
- unique suffix：16,384 tokens；
- shared-prefix ratio：50%；
- cache residency：CPU-resident hit；
- output length：固定且较短；
- input modality：text-only。

这一配置同时保留可观测的 residual prefill computation 与 CPU-GPU state restore demand，适合研究高负载下两类资源的竞争关系。

如果 32K 在其中一个模型上无法形成稳定的完整负载曲线，则正式 cross-model load comparison 使用两个模型都能稳定运行的最大公共 context point。任何降级必须在运行前固定并记录，而不能只对某些高负载点临时改变 context。

## 4. Capacity calibration

正式 sweep 前分别对两个模型执行短的 capacity calibration。

Calibration 用于估计当前固定 workload 下的 sustainable capacity。Sustainable capacity 的定义遵循 `00-measurement-conventions.md`：achieved throughput 能够跟随 offered load，并且请求队列不存在持续增长趋势。

正式 sweep 的相对负载点根据 calibration 结果确定。建议覆盖以下区域：

- 明显低负载；
- 约半负载；
- 中高负载；
- 接近 saturation；
- saturation 附近；
- 轻度 overload。

最终采用约 6-8 个点即可。具体 requests/s 在 calibration 后写入配置并冻结，正式运行过程中不再根据结果临时调整。

同时保存：

- absolute offered requests/s；
- achieved requests/s；
- normalized load。

## 5. Concurrency 的处理

本实验不进行独立 concurrency sweep。

设置足够高且固定的 concurrency ceiling，使正常负载区域不会因人为上限提前截断。实际 active concurrency 随 arrival pressure 自然变化，并作为结果记录。

因此，本实验研究的是：

> arrival pressure → active concurrency/resource contention → queueing/stall → throughput and TTFT

## 6. Cache-residency 条件

主实验使用 **CPU-resident hit**。

每个请求的 shared prefix 已经存在于 CPU/offload tier，需要在请求执行过程中恢复所需 cache/state，然后计算 unique suffix。

为了判断 hierarchical reuse 的高负载价值，选择少量代表性 load points 增加：

### Recompute baseline

不恢复共享 prefix，完整重新计算 context。

建议至少覆盖低负载、接近 saturation 和高负载区域。

### GPU-resident hit control

如果 runtime 能够可靠控制 GPU residency，可在低负载和接近 saturation 的代表点加入该条件，用于区分 CPU restore contention 与 prefix reuse 本身的开销。

这两个控制条件不是新的完整 load sweep。

## 7. 核心测量指标

### 7.1 Offered load 与 achieved throughput

记录：

- offered request rate；
- achieved throughput。

形成：

> Offered load → achieved throughput

当 offered load 继续增加而 achieved throughput 进入平台，同时 queue 持续增长时，将该区域标记为 overload，而不是把它当作稳定 capacity point。

### 7.2 Active concurrency

记录稳定观测窗口中的平均与峰值 active requests。

形成：

> Offered load → active concurrency

这一结果用于说明 arrival pressure 如何转化为同时存在的 execution/cache-state pressure。

### 7.3 Cache/state memory pressure

记录：

- GPU resident cache/state footprint；
- CPU/offload tier footprint；
- 峰值与稳定区间使用量；
- 能够获取时的 state-type breakdown。

重点观察高负载是否导致 GPU residency、allocator pressure 或 restore concurrency 明显增加。

### 7.4 CPU-GPU traffic

记录：

- transferred bytes per request；
- transferred bytes per second；
- achieved transfer bandwidth；
- transfer activity/duration。

理论上固定 workload 下每请求 transfer volume 应在同一 residency policy 下基本稳定。若 bytes/s 继续增长而 achieved bandwidth 接近平台，则表明 I/O path 接近资源上限。

Raw transfer duration 允许与 computation 重叠，不直接作为 TTFT additive component。

### 7.5 Non-overlapped I/O stall

记录真正阻塞 service path 的 restore stall。

主要使用：

```text
service stall ratio = I/O stall / service time
```

同时可以报告：

```text
TTFT stall contribution = I/O stall / TTFT
```

前者用于判断 execution path 是否越来越 I/O-bound，后者用于观察 end-to-end 影响。

### 7.6 Queueing delay

单独记录请求 arrival 到 service 开始之间的等待时间。

Queueing 必须与 execution/service latency 分开，否则高负载下无法判断 TTFT 恶化来自请求本身变慢，还是系统已经进入排队放大阶段。

### 7.7 TTFT

记录：

- P50 TTFT；
- P90 TTFT；
- P99 TTFT。

Experiment 3 将 tail latency 作为核心结果，因为资源 contention 往往先反映在 P90/P99，而不是 median。

## 8. TTFT decomposition

统一使用：

```text
TTFT = queueing + service time
service time = compute-path time + non-overlapped I/O stall + other service overhead
```

Transfer duration 是资源活动区间，在与 computation overlap 时不能再次加进 service time。

任何无法严格拆分的时间段都放入 `other service overhead`，而不是强行归类到 I/O 或 computation。

## 9. 实验执行方式

每个 request-rate 配置包含 warm-up 与固定的稳定观测窗口，并进行多次重复运行。

正式结果报告：

- achieved throughput；
- P50/P90/P99 TTFT；
- queueing distribution；
- service stall ratio；
- transfer bandwidth；
- active concurrency；
- 波动范围。

如果某个 load point 出现持续增长的 queue，则该点标记为 `unstable overload`。该点仍保留 raw data，但不与稳定点混在一起计算 steady-state latency。

## 10. 实验顺序与状态恢复

正式测试不只按照从低到高的单一顺序执行。

采用交替或随机化顺序，降低 thermal state、allocator history 或 cache residue 对曲线的系统性影响。

每个配置开始前恢复到同一 cache-residency 初始条件，并确认目标 prefix 位于规定 tier。

## 11. 最终结果组织

实验形成五组核心结果。

### 图 1：Offered Load → Achieved Throughput

用于识别 sustainable region、saturation region 与 overload。

### 图 2：Normalized / Absolute Load → P50/P90/P99 TTFT

绝对 load 用于描述真实 capacity，normalized load 用于跨模型比较接近各自 capacity 时的行为。

### 图 3：Load → CPU-GPU Traffic

展示 bytes/s、achieved bandwidth 和必要的 per-request transfer volume。

### 图 4：Load → I/O Stall and Queueing

分别展示 service stall ratio 与 queueing delay，避免把 I/O contention 和排队放大混为一谈。

### 图 5：Representative Load → TTFT Composition

选择低负载、中高负载、接近 saturation 和 overload 前缘的代表点，展示 queueing、compute path、non-overlapped I/O stall 和 other overhead。

## 12. 结果判断逻辑

### 情况 A：I/O bottleneck 随负载显现

低负载下 CPU restore cost 较小，但随着 request rate 提高，transfer bandwidth 接近平台、service stall ratio 上升、tail TTFT 明显恶化，并最终限制 achieved throughput。

该结果说明 state bottleneck 具有明显 load-dependent 特征。

### 情况 B：主要由 computation saturation 导致

如果系统进入 saturation，但 achieved transfer bandwidth 仍有余量、service stall ratio 基本稳定，而 computation/service time 与 queueing 主导 TTFT，则说明当前 workload 的主要高负载瓶颈不是 state movement。

### 情况 C：Hierarchical reuse 的优势随负载减弱

如果低负载下 CPU-resident hit 明显优于 recompute，但接近 saturation 后两者差距缩小，同时 CPU restore contention 增强，则说明 hierarchical cache 仍有基础价值，但其高负载收益受到 state movement 限制。

### 情况 D：两个模型的 saturation path 不同

如果一个模型首先进入 state/I/O pressure，而另一个首先进入 computation pressure，则结论限定为 serving bottleneck evolution 具有 model-dependent 特征。

本实验不把该差异直接归因于 attention architecture。

## 13. 与前两个实验的关系

| Experiment | Primary variable | Question |
|---|---|---|
| 1 | Context length | 长 context 是否放大 state pressure |
| 2 | Shared-prefix ratio | reuse 收益是否被 restore cost 抵消 |
| 3 | Request arrival rate | serving load 是否将 state cost 放大为系统瓶颈 |

三个实验共同形成：

```text
state behavior
→ reuse/restore cost
→ concurrent restore pressure
→ TTFT / throughput degradation
```

统一 measurement definitions 见 [00-measurement-conventions.md](00-measurement-conventions.md)。
