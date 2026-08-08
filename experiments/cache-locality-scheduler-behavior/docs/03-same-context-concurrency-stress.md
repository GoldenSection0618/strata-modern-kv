# Experiment 3: Same-Context Concurrency Stress

## 1. 实验目标

本实验用于研究多个请求在短时间内集中访问同一 reusable context 时，现代 serving 系统是否会出现明显的 cache reuse failure 和 scheduler contention。

实验固定总体 request arrival rate，只改变同一 context 请求在时间上的聚集程度，并观察高并发热点是否导致 delay hit、redundant prefill、重复 cache loading、queueing 和 TTFT 恶化。

本实验主要回答三个问题：

1. 同一 context 的并发请求增加后，已有 cache reuse opportunity 是否仍然能够被有效利用；
2. delay-hit mitigation 是否能够阻止多个请求在同一 cache/state 尚未 ready 时重复执行 prefill 或其他重复工作；
3. 完整 scheduler 是否能够在高共享、高并发 workload 下保持稳定的 TTFT、throughput 和 fairness。

Experiment 1 研究 locality × overall load 的一般关系。

Experiment 2 研究不同 scheduler mechanism 的贡献。

Experiment 3 不重新进行这两类 sweep，而是单独研究 **hot-context concurrency** 这一特殊但重要的压力场景。

所有配置遵循 [`00-common-conventions.md`](00-common-conventions.md)。

## 2. 核心实验变量

本实验的主要自变量为 **same-context fan-in**。

Same-context fan-in 表示在一个 reusable context 尚未完成所需 cache/state restore 或 reuse preparation 时，同时到达并希望复用该 context 的请求数量。

实验设置四个 concurrency level。

| Level | 含义 |
|---|---|
| C0 | 同一 context 请求基本不重叠，作为 serialized control |
| C1 | 少量请求发生 overlap |
| C2 | 明显的同-context 并发 |
| C3 | 接近当前稳定 serving 条件下能够形成的高并发热点 |

具体 fan-in 数量在实验前 calibration 中确定。

C0-C3 的划分以实际 overlap 和系统稳定性为依据，而不是跨模型固定使用同一绝对请求数。

C3 必须仍处于可以完成稳定测量的区域，不通过系统整体 overload 人为制造 scheduler failure。

## 3. 固定总体负载

本实验不通过提高总体 request arrival rate 来增加 same-context concurrency。

所有 C0-C3 workload 使用相同的平均 offered request rate。

该 request rate 选择 Experiment 1 中已经确认的 **High but stable** load point。

不同 concurrency level 只改变同一 context 请求在局部时间窗口中的聚集程度。

当某个 context 的多个请求被压缩到更短时间窗口时，后续请求间隔相应调整，使整个 trace 的长期平均 arrival rate 保持一致。

因此主实验保持：

```text
overall offered load = fixed
context reuse opportunity = fixed
request count = fixed
same-context temporal concentration = variable
```

这种设计用于把 hot-context concurrency 与普通系统 saturation 分离。

## 4. Workload 构造

实验建立固定的 reusable-context pool。

每个 context 对应多个请求。

同一个 context group 内的 reusable prefix/context 完全一致，suffix/query 不同，output request 不完全相同。

实验不使用大量完全重复的 identical requests，因为目标是研究 context reuse，而不是重复请求缓存。

不同 context group 使用匹配的 context length 和请求数量，使某一个热点 context 不因内容长度特殊而产生系统性偏差。

## 5. Concurrency trace 构造

所有 concurrency workload 使用相同的 logical request set。

不同 workload 只改变同一 context 请求的 arrival timestamp。

### C0: Serialized

同一 context 的请求被充分错开。

后续请求到达时，前一个请求触发的 cache/state preparation 已经完成。

该条件表示理想情况下不存在明显 same-context race。

### C1: Low fan-in

少量同-context 请求在 cache/state preparation 尚未结束时发生 overlap。

该条件用于观察 scheduler pathology 开始出现的位置。

### C2: Medium fan-in

更多请求集中进入同一个 reuse window。

该条件用于观察 delay hit、redundant prefill 和 queueing 是否开始明显放大。

### C3: High fan-in

同一 context 在很短的时间窗口内收到大量请求。

该条件用于测试 scheduler 对 hot-context burst 的稳定性。

不同 concurrency trace 使用完全相同的：

- context pool；
- 每个 context 的访问次数；
- input/output length distribution；
- total request count；
- total theoretical reusable-token volume；
- average offered request rate。

## 6. Cache initial state

主实验使用 **warm reusable context，但目标 state 在正式 burst 开始时不处于 GPU-ready 状态** 的条件。

目标 context 已经具有可以被重新利用的 cache/state，因此后续请求理论上不需要重新完成整个 prefix computation。

同时，该 state 需要经历当前系统真实的 restore / loading / preparation path。

这样才能触发 Experiment 3 真正需要研究的问题：

```text
reusable state exists
        ↓
first request begins preparing/restoring it
        ↓
other requests for same context arrive
        ↓
scheduler decides wait / reuse / recompute
```

如果 context 从一开始就完整 GPU-resident，大部分 delay-hit 问题不会被暴露。

正式运行前必须通过 runtime observable behavior 验证目标 state 的初始 residency 与 restore/preparation path，不能仅根据配置推断其状态。

## 7. GPU-resident control

实验增加一个 GPU-resident control。

选择代表性的 C0 与 C3 workload，在正式请求开始前保证目标 reusable context 已处于可直接使用的 GPU-resident 状态。

该 control 用于区分高并发本身造成的 queueing，与 cache/state 尚未 ready 时同-context 高并发造成的 delay-hit pathology。

如果 C3 在 restore-required 条件下明显恶化，而 GPU-resident C3 中该现象大幅减弱，则能够更有力地说明问题来自 cache preparation race，而不是普通的请求并发。

GPU-resident control 不进行完整 C0-C3 sweep。

## 8. Scheduler configurations

本实验不重新进行 Experiment 2 的完整四级 component ablation。

主实验使用三个 scheduler configuration。

| Configuration | 作用 |
|---|---|
| S0 Baseline | 建立 hot-context pathology baseline |
| S1 Delay-hit mitigation | 直接测试同-context race 的核心机制 |
| S3 Full scheduler | 测试完整 control plane 在压力场景下的最终表现 |

Balanced batching 和 stall hiding 的独立贡献已经由 Experiment 2 分析。

Experiment 3 重点判断以下序列在 same-context concurrency 增强时是否能够逐步保持系统稳定：

```text
Baseline
→ Delay-hit mitigation
→ Full scheduler
```

## 9. 主实验矩阵

主实验采用四个 concurrency levels 与三个 scheduler configurations 的交叉设计，共 12 个主要条件。

| Same-context fan-in | Baseline | Delay-hit mitigation | Full scheduler |
|---|---:|---:|---:|
| C0 | ✓ | ✓ | ✓ |
| C1 | ✓ | ✓ | ✓ |
| C2 | ✓ | ✓ | ✓ |
| C3 | ✓ | ✓ | ✓ |

另外增加少量 GPU-resident control。

每个 condition 进行多次独立重复测量。

## 10. 实验前 calibration

Calibration 首先确定 Experiment 1 中 High but stable 的 request-rate operating point。

然后在固定 overall arrival rate 下逐渐提高 same-context fan-in。

Calibration 记录：

- 实际 overlap request count；
- effective concurrency；
- achieved request rate；
- backlog；
- active-request preemption；
- OOM 或 runtime instability。

C3 选择为能够形成明显 hot-context pressure，但仍不会使整个系统进入持续 overload 的最大代表性区域。

任何依赖全局 saturation 才能形成的 concurrency level 不进入主实验。

## 11. 正式实验流程

每轮实验首先建立规定的 reusable-context residency。

随后验证目标 context 的 state 当前确实需要经过预定的 restore / preparation path。

系统加载固定 request trace，并按照 C0-C3 对应 timestamps 发送请求。

不同 scheduler configuration 使用完全相同的 trace。

每轮实验记录目标 context 的 state availability、restore / preparation transition、concurrent waiter、cache hit / miss decision、recomputation 和 request completion。

正式实验重复多次。

不同 configuration 和 concurrency level 的执行顺序交替或随机化。

## 12. Reuse realization 指标

Experiment 3 的核心不是普通 request-level cache hit rate，而是理论可复用工作究竟有多少真正被复用了。

实验记录：

- theoretical reusable token/state volume；
- realized reused token/state volume；
- redundant recomputation volume；
- redundant prefill；
- reuse realization ratio。

定义概念上满足：

```text
reuse realization
=
actually reused reusable work
/
theoretically reusable work
```

如果 concurrency 增大后 theoretical reuse 不变，而 realized reuse 明显下降，则说明高并发正在破坏 reuse opportunity。

## 13. Delay-hit 行为

实验记录：

- delay-hit event count；
- affected request count；
- affected token/state volume；
- requests arriving while matching state is being prepared；
- requests successfully deferred；
- requests incorrectly falling back to recomputation；
- redundant prefill volume。

核心分析链为：

```text
same-context fan-in increases
        ↓
more requests arrive during state preparation
        ↓
delay-hit opportunity increases
        ↓
wait/reuse OR redundant recomputation
```

S0 与 S1 的主要区别必须通过这一机制链解释。

## 14. Cache / restore behavior

实验记录目标 reusable context 的：

- GPU residency transition；
- CPU-resident reuse；
- restore count；
- restore volume；
- duplicate restore activity；
- cache/state preparation duration；
- non-overlapped I/O stall。

理想情况下，多个并发请求访问同一 reusable context 时，不应产生与请求数量近似线性增长的重复 restore 或重复 prefill。

如果 C3 中同一 context 产生明显重复 data movement，则该行为作为单独的 scheduler/cache-coordination pathology 报告。

## 15. Queueing 与请求等待

实验记录：

- queueing delay；
- scheduler deferral time；
- within-burst waiting time；
- maximum waiting time；
- queue-age distribution。

Delay-hit mitigation 本质上可能主动让部分请求等待 cache ready。

因此 waiting time 增加本身不等价于 regression。

需要判断这种等待是否换来了更少的 redundant computation，并最终改善 TTFT 或系统 throughput。

## 16. 用户可见性能

实验统一报告：

- P50 TTFT；
- P90 TTFT；
- P99 TTFT；
- throughput；
- request completion time；
- TPOT 或等价 decode latency。

Same-context burst 特别关注 tail TTFT。

即使平均 TTFT 基本稳定，高并发热点仍可能使 burst 中后到达的请求产生严重 tail latency。

## 17. Burst-level 指标

除全局指标外，Experiment 3 单独保存每个 hot-context burst 的统计。

每个 burst 记录：

- fan-in；
- first-request TTFT；
- median TTFT；
- last-request TTFT；
- burst completion span；
- redundant prefill；
- realized reuse；
- restore activity。

这样能够观察同一组请求内部性能如何随请求位置变化。

例如：

```text
request 1 triggers restore
request 2 arrives during restore
request 3 arrives during restore
...
```

如果 baseline 中后续请求的 TTFT 和 redundant work 随 fan-in 快速增长，而 delay-hit mitigation 后明显缓解，则能够形成比全局平均指标更直接的证据。

## 18. 分析一：Baseline 对 concurrency 的敏感性

首先只分析 S0。

比较：

```text
C0 → C1 → C2 → C3
```

判断随着 same-context fan-in 增加：

- delay hit 是否增加；
- realized reuse 是否下降；
- redundant prefill 是否增加；
- duplicate restore 是否出现；
- P99 TTFT 是否恶化。

这一分析回答 Baseline scheduler 是否存在真正的 hot-context concurrency pathology。

如果 S0 从 C0 到 C3 基本稳定，则说明现代 runtime 已经能够较好处理同-context coordination，Experiment 3 的后续 scheduler 收益预期应相应降低。

## 19. 分析二：Delay-hit mitigation

主要比较 S0 与 S1，重点分析 C2 和 C3。

判断 delay-hit mitigation 是否增加合理 deferral、减少 premature miss、减少 redundant prefill、提高 reuse realization，并改善 TTFT tail 或 throughput。

如果 S1 只是增加 waiting time，但没有减少 redundant work，则该机制在当前 runtime 下没有实现预期价值。

## 20. 分析三：Full scheduler

主要比较 S1 与 S3。

重点判断 delay-hit 问题被处理以后，balanced batching 与 stall hiding 是否还能进一步改善高并发热点。

观察 exposed I/O stall、GPU utilization、queueing、throughput 与 TTFT tail。

如果 S1 已经获得绝大部分收益，而 S3 几乎没有进一步改善，则说明 hot-context workload 的主要 scheduler bottleneck 是 delay hit。

该结果属于有价值的机制结论。

## 21. 分析四：GPU-resident control

比较 restore-required 与 GPU-resident 条件，重点使用 C3。

如果：

```text
restore-required C3
→ delay hit / redundant work sharply increases

GPU-resident C3
→ same pathology largely disappears
```

则说明问题主要来自 reusable state preparation 与并发请求之间的协调。

如果 GPU-resident C3 仍然出现相似 TTFT 和 queueing deterioration，则说明主要瓶颈可能是一般性 high-concurrency contention，而不能完全归因于 delay hit。

## 22. Fairness 与 starvation 检查

Scheduler 不允许通过长期推迟某些 hot-context requests 来获得更高 aggregate throughput。

实验记录：

- maximum request waiting time；
- P99 queueing；
- starvation event；
- burst 内最慢请求；
- request completion ordering。

任何出现明显 starvation 的 configuration 都必须单独报告。

即使 aggregate throughput 更高，也不能直接视为稳定的 scheduler improvement。

## 23. 实验控制条件

除 same-context fan-in 和 scheduler configuration 外，主要条件保持固定。

包括：

- model identifier / revision；
- hardware；
- precision；
- cache dtype；
- cache hierarchy；
- cache capacity；
- page/cache policy；
- I/O backend；
- context pool；
- context length；
- suffix/input distribution；
- output length distribution；
- total request count；
- theoretical reuse opportunity；
- average offered arrival rate。

本实验不 sweep：

- context length；
- cache capacity；
- page size；
- I/O backend；
- overall request rate；
- hardware。

这些变量分别由其他实验组研究。

## 24. 结果组织

Experiment 3 至少形成以下结果。

### Figure A: Concurrency pathology curve

横轴为：

```text
C0 → C1 → C2 → C3
```

纵轴分别展示：

- delay hit；
- redundant prefill；
- reuse realization；
- duplicate restore。

比较 S0、S1 和 S3。

### Figure B: User-visible performance

展示：

- P50/P90/P99 TTFT；
- throughput；
- request completion time。

观察 scheduler-level pathology 是否真正传导到 serving performance。

### Figure C: Burst behavior

选择代表性的 C3 burst，展示 burst 内不同请求的 arrival、wait、restore/reuse、prefill 与 TTFT。

该结果用于直接展示 hot-context race 的实际执行行为。

### Figure D: Residency control

比较：

```text
restore-required C3
vs
GPU-resident C3
```

用于区分 cache-preparation contention 与一般性 concurrency contention。

## 25. 结果判断逻辑

### 情况 A：Baseline 随 fan-in 明显恶化，delay-hit mitigation 有效

随着 C0-C3，baseline delay hit、redundant prefill 和 tail TTFT 明显增加。

S1 显著减少 redundant work，并提高 realized reuse。

该结果说明 Strata 所针对的 same-context delay-hit problem 在现代 workload 中仍然存在。

### 情况 B：Baseline 出现 pathology，但完整 scheduler 的额外收益有限

S1 已经消除大部分问题，S3 相对 S1 增益较小。

该结果说明同-context 高并发的主要问题是 delay-hit coordination，而不是 batch balancing 或 stall hiding。

### 情况 C：内部 pathology 存在，但端到端影响有限

Concurrency 增加导致 delay hit 或 redundant work 增加，但 TTFT / throughput 基本不变。

该结果说明当前系统仍有足够资源吸收这些额外工作。

不能据此声称 scheduler optimization 具有显著 serving benefit。

### 情况 D：GPU-resident 条件下问题消失

Restore-required C3 明显恶化，而 GPU-resident C3 基本稳定。

该结果能够较强地支持性能问题来自 cache/state preparation 与并发请求之间的 coordination，而不是并发本身。

### 情况 E：所有 concurrency level 均稳定

S0 在 C0-C3 下都能够保持较高 reuse realization，并且几乎没有 redundant prefill 或异常 tail latency。

该结果说明当前 runtime 已经能够较好处理 same-context concurrency。

Strata 原本针对的这一 scheduler pathology 在现代系统中的重要性已经下降。

## 26. 实验边界

Experiment 3 只研究 **same-context concurrency**。

本实验不重新研究一般 locality × arrival-rate surface，也不重新执行完整 scheduler component ablation。

Experiment 1 已经确定普通 workload 下问题出现在哪里。

Experiment 2 已经确定各 scheduler mechanism 分别解决什么问题。

Experiment 3 提供一个针对 hot shared context 的压力验证。

这些结果最终由 Experiment 4 统一形成 scheduler operating region。
