# Experiment 3: Same-Context Concurrency Stress

## 1. 实验目标

本实验专门研究多个请求在短时间内访问同一 context 时，现代 serving runtime 是否仍然出现 Strata 所定义的 delay hit。

Delay hit 的核心条件是：第一个相关请求发生 cache miss 后，该 context 尚未 resolve；后续同-context 请求在这个 resolve window 内到达。如果 scheduler 未正确协调，这些请求可能被再次视为 miss，并产生 redundant prefill / recomputation。

本实验固定长期平均 offered request rate，只改变同一 context 在 resolve window 内的并发聚集程度，从而把 hot-context overlap 与普通全局 overload 分离。

本实验主要回答：

1. same-context fan-in 增大后，cold-miss resolve 是否产生更多 delay hit 与 redundant work；
2. delay-hit mitigation 是否能通过合理 deferral 提高 reuse realization；
3. full scheduler 是否在 delay hit 被处理后仍能改善 residual loading / queueing；
4. CPU-resident restore 是否形成类似的 coordination problem，以及它与 cold-miss delay hit 有何区别。

所有配置遵循 [`00-common-conventions.md`](00-common-conventions.md)。

## 2. 主实验场景：Cold-miss resolve

主实验使用 `cold-miss` 状态，而不是把 CPU restore 作为 delay hit 的定义前提。

每个目标 context 在对应 burst 开始前尚未 materialize 为可复用 cache/state。

第一个请求触发该 context 的实际 prefill / state construction。后续同-context 请求被安排在这一 cache resolve window 内以构造 C1–C3 fan-in。

概念流程为：

```text
first request misses
        ↓
context computation / cache materialization is in progress
        ↓
more requests for the same context arrive
        ↓
wait for resolve and reuse OR duplicate work
```

这与 Strata 对 delay hit 的原始定义一致，也避免把 cache eviction、CPU hierarchy 和 host-transfer latency混入主因果链。

## 3. 主要自变量：Same-context fan-in

Same-context fan-in 表示一个目标 context 的首次 miss 尚未 resolve 时，到达并引用该 context 的请求数量。

实验设置四个 level。

| Level | 定义 |
|---|---|
| C0 | 同-context 请求基本不落入同一 resolve window，作为 serialized control |
| C1 | 少量请求在 resolve window 内 overlap |
| C2 | 明显的同-context overlap |
| C3 | 高 fan-in burst，但系统整体仍处于 stable-serving 区域 |

具体 fan-in 数量不跨模型固定，而由 calibration 根据实际 cache resolve time 与系统稳定性确定。

C3 不允许通过全局持续 overload 人为制造。

## 4. 固定长期总体负载

C0–C3 使用相同长期平均 offered request rate。

默认使用 Experiment 1 已经确认的 `High but stable` operating point，因为该区域更容易观察 delay hit，同时仍能与 capacity saturation 分离。

提高 fan-in 时，只压缩同一 context group 内请求的局部到达间隔，并在其他时间段补偿，使完整 trace 的：

- total request count；
- trace duration；
- average offered request rate；
- context pool；
- per-context request count；
- input/output length distribution；
- theoretical reusable volume

保持一致。

实验同时记录瞬时 burst rate，因为长期平均 rate 一致并不意味着局部负载一致。局部 burstiness 是本实验有意控制的变量之一。

## 5. Workload 构造

实验建立固定 context pool。

每个 context group 包含多个共享同一 prefix/context 的请求。Suffix/query 与 output generation 不完全相同，避免把任务退化为 identical-request memoization。

不同 context group 使用匹配的 prefix length 和访问次数，避免单个 context 因长度特殊而主导结果。

每条请求保存：

- context group ID；
- arrival timestamp；
- fan-in burst ID；
- input/output token length；
- eligible reusable token/state volume；
- request position within burst。

第一个 cold-miss request 本身不计入“理论可被前序 cache 复用”的 volume。Reuse realization 的分母只统计在首次 resolve 后理论上可以通过等待复用而避免重复工作的后续请求部分。

## 6. Cache resolve time

每个 burst 实际测量 cache resolve time，而不是只根据设定值推断。

定义：

```text
cache resolve time
=
matching context first becomes safely reusable
-
first unresolved miss is accepted
```

开始和结束事件必须来自 runtime observable behavior 或 instrumentation。

Experiment 3 不把 cache resolve time 作为新的完整 sweep axis。它主要作为解释 fan-in 与 delay-hit probability 的 observed variable。

如果需要复验 Strata Figure 12 的方向，可以在主结果后补充极少量 timing-sensitivity points，但不能因此扩展成第三个大规模维度。

## 7. Scheduler configurations

主实验使用三个 configuration：

| Configuration | 作用 |
|---|---|
| S0 Baseline | 测量未处理的 same-context delay hit |
| S1 Delay-hit mitigation | 验证 defer-until-resolved 机制 |
| S3 Full scheduler | 检查其他 scheduler stages 是否还有额外价值 |

Experiment 2 已负责 balanced batching 与 stall hiding 的独立 attribution，因此 Experiment 3 不重复完整 S0–S3 component matrix。

所有 scheduler configuration 必须通过 Experiment 2 的 semantic capability gate。

## 8. 主实验矩阵

主矩阵为 4 fan-in levels × 3 scheduler configurations，共 12 conditions。

| Fan-in | Baseline | Delay-hit mitigation | Full scheduler |
|---|---:|---:|---:|
| C0 | ✓ | ✓ | ✓ |
| C1 | ✓ | ✓ | ✓ |
| C2 | ✓ | ✓ | ✓ |
| C3 | ✓ | ✓ | ✓ |

每个 condition 进行多次独立重复测量。

## 9. GPU-ready control

增加少量 `gpu-ready` control。

选择 C0 与 C3，在 burst 开始前保证目标 context 已经 GPU-ready。

该 control 用于区分：

- burst concurrency 本身造成的 queueing/contention；
- unresolved miss 导致的 delay-hit-specific pathology。

如果 cold-miss C3 明显恶化，而 gpu-ready C3 中 redundant work 消失且 tail latency 大幅缓解，则支持 delay-hit interpretation。

GPU-ready control 不需要完整 C0–C3 sweep。

## 10. CPU-restore extension

在 full hierarchical restore 已通过 validation 时，增加少量 `cpu-restore` control。

目标 context 已经存在于 CPU tier，但 burst 开始时尚未 GPU-ready。第一个相关请求触发 restore，其他请求可能在 restore resolve window 内到达。

建议只运行代表性的 C0 与 C3，并比较 S0、S1。

该扩展回答：delay-hit coordination 是否也出现在 host restore 尚未完成时。

CPU-restore 结果与 cold-miss 主结果分开报告。它不能用来重新定义 delay hit，也不能在 partial hierarchy 状态下并入 full-state结论。

## 11. Calibration

Calibration 分两步执行。

### 11.1 固定 stable load

复用 Experiment 1 的 High stable operating point，并确认当前 cold-miss trace 在 baseline 下不会形成持续 backlog。

### 11.2 确定 C0–C3

测量目标 prefix 长度下的实际 cache resolve time，然后逐步压缩同-context arrival spacing。

记录：

- observed fan-in；
- cache resolve time；
- instantaneous burst rate；
- effective concurrency；
- backlog；
- achieved request rate；
- active-request preemption；
- runtime instability。

C3 选择为能够形成明显 overlap，但不依赖全局 overload 的最高代表性区域。

## 12. 正式实验流程

每轮实验：

1. 恢复目标 resolve mode；
2. 验证 cold-miss / gpu-ready / cpu-restore 状态符合设计；
3. 启动固定 trace；
4. 记录首次 miss、resolve transition、后续 same-context arrival、scheduler decision、reuse/recomputation 和 completion events；
5. 保存 per-request 与 per-burst metrics；
6. 重复测量并随机化 configuration 执行顺序。

不同 scheduler configuration 使用完全相同的 exact arrival timestamps。

## 13. Delay-hit / reuse metrics

记录：

- unresolved same-context arrivals；
- delay-hit event count；
- affected request/token/state volume；
- deferred request count；
- deferral duration；
- redundant prefill / recomputation；
- realized reused volume；
- reuse realization ratio；
- cache resolve time。

Reuse realization 概念定义为：

```text
actually reused eligible work
/
theoretically reusable work after the initial miss
```

如果 fan-in 增大而 theoretical reusable work 保持不变，但 realized reuse 下降，则说明 concurrency 正在破坏 reuse opportunity。

## 14. Queueing 与 serving metrics

统一记录：

- queueing delay；
- scheduler deferral time；
- P50/P90/P99 TTFT；
- request completion time；
- throughput；
- TPOT / decode latency；
- maximum waiting time；
- starvation event。

Delay-hit mitigation 可能有意增加某些请求的短期等待，因此 waiting time 增加本身不等价于 regression。判断依据是等待是否换来 redundant work reduction，并最终改善或至少不恶化 relevant SLO。

## 15. Burst-level metrics

每个 hot-context burst 单独保存：

- observed fan-in；
- cache resolve time；
- first-request TTFT；
- median / last-request TTFT；
- burst completion span；
- delay-hit count；
- redundant prefill；
- reuse realization；
- scheduler deferral；
- restore activity when cpu-restore mode is used。

Burst timeline 是本实验的重要证据，因为它可以直接显示第一个 miss 与后续请求的重叠关系。

## 16. 分析一：Baseline fan-in sensitivity

只看 S0，比较 C0→C3。

判断 fan-in 增大后：

- unresolved same-context arrivals 是否增加；
- delay hit 是否增加；
- redundant work 是否增加；
- reuse realization 是否下降；
- P99 TTFT / burst completion span 是否恶化。

如果 S0 在 C0–C3 下保持稳定，则说明当前 runtime 已经较好地协调同-context miss，Strata-style delay-hit optimization space 收缩。

## 17. 分析二：Delay-hit mitigation

比较 S0 vs S1，重点分析 C2/C3。

理想证据链为：

```text
more same-context overlap
        ↓
baseline duplicate work
        ↓
S1 defers later requests until resolve
        ↓
redundant work decreases
        ↓
reuse realization / tail TTFT improves
```

如果 S1 只增加等待而不减少 duplicate work，则该 mechanism 在当前 runtime 下没有实现预期价值。

## 18. 分析三：Full scheduler

比较 S1 vs S3。

Cold-miss delay hit 被处理后，如果 remaining bottleneck 主要来自普通 compute/queueing，则 full scheduler 可能没有额外收益。

如果同时存在 host loading 或 residual stall，S3 可能进一步改善 TTFT / throughput。

该比较只判断额外价值，不重新归因 balanced batching / stall hiding 的独立机制。

## 19. 分析四：Controls

### GPU-ready

如果 gpu-ready C3 仍出现严重 queueing 但没有 redundant prefill，则说明其中一部分 tail latency 来自 burst concurrency 本身，而不是 delay hit。

### CPU-restore

如果 cold-miss 与 cpu-restore 都出现 same-context coordination failure，则说明 unresolved-context coordination 问题跨 compute-resolve 与 load-resolve 两类路径存在。

如果只有 cpu-restore 出现问题，则主要瓶颈更接近 hierarchical-cache loading coordination，而不是一般 cold-miss delay hit。

## 20. 结果输出

至少形成：

1. C0–C3 的 delay-hit / redundant-work / reuse-realization curve；
2. S0 vs S1 vs S3 的 P50/P90/P99 TTFT 与 throughput；
3. 代表性 C3 burst timeline；
4. cold-miss C3 vs gpu-ready C3 control；
5. 可用时增加 cold-miss vs cpu-restore C3 comparison；
6. cache resolve time 与 observed fan-in summary。

## 21. 结果判断逻辑

### A. Baseline 随 fan-in 明显恶化，S1 有效

该结果说明 Strata 定义的 same-context delay hit 在现代 runtime 中仍然存在。

### B. Baseline 有 pathology，但 S1 已获得绝大部分收益

该结果说明 hot-context 场景主要由 delay-hit coordination 主导，full scheduler 的其他阶段增量有限。

### C. 内部 duplicate work 增加但端到端影响有限

说明 pathology 存在但不是当前 dominant bottleneck，不能仅根据内部 metric 声称显著 serving value。

### D. GPU-ready control 同样恶化

说明大量性能下降来自一般 burst concurrency / queueing。Delay-hit-specific claim 必须只基于 cold-miss 相对 gpu-ready 的额外部分。

### E. C0–C3 均稳定

说明当前 runtime 已经能够较好处理 unresolved same-context requests。Strata delay-hit mechanism 的 operating region 在当前 stack 上明显缩小。

## 22. 实验边界

Experiment 3 只研究 same-context concurrency 与 unresolved-context coordination。

Experiment 1 已建立一般 cache-distance × load surface。Experiment 2 已做 scheduler component attribution。Experiment 4 负责把本实验的 hot-context规则与一般 workload operating map 合并。
