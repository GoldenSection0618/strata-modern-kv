# Experiment 2: Scheduler Component Ablation

## 1. 实验目标

本实验用于分离 Strata 三类 scheduler optimization 在现代 hybrid LLM serving workload 中的实际贡献。

实验在 Experiment 1 已经定位出的 representative workloads 上，逐步加入 delay-hit mitigation、balanced batching 和 bubble filling / stall hiding，并观察 scheduler-level pathology 与端到端 serving performance 的变化。

本实验主要回答三个问题：

1. delay-hit mitigation 是否主要通过减少 delay hit 和 redundant prefill 获得收益；
2. balanced batching 是否主要通过改善 computation 与 cache loading 的组合关系减少 loading-bound behavior；
3. bubble filling / stall hiding 是否能够进一步利用不可避免的 I/O stall，并将 GPU idle time 转化为有效计算。

本实验不重新扫描 Experiment 1 的完整 locality × arrival-rate 空间，而是在少量预先确定的 workload 上完成机制消融。

所有配置遵循 [`00-common-conventions.md`](00-common-conventions.md)。

## 2. 实验设计原则

本实验保持模型、硬件、cache hierarchy、cache capacity、I/O backend 和 workload trace 不变，只改变 scheduler mechanism。

Page granularity、GPU-assisted I/O、CPU cache capacity 和其他 data-plane optimization 不进入本实验 sweep。

所有 scheduler 配置使用同一个经过验证的 cache 与 I/O configuration，从而避免把 I/O backend 改善错误归因于 scheduler。

实验主要在系统仍能稳定服务的负载区域进行。

Experiment 1 中的 Overload workload 只作为补充诊断条件，不作为 scheduler 优化价值的主要证据，因为持续 backlog 本身会主导 queueing delay。

## 3. Representative workload 选择

Experiment 2 不预先根据预期结果指定固定 workload，而是在 Experiment 1 完成后按照预先规定的规则选择 representative points。

选择过程必须在运行任何 scheduler ablation 之前完成并冻结。

至少选择以下四类 workload。

### W0: Control workload

选择 Experiment 1 中 scheduler pathology 较弱的稳定 workload。

该 workload 应具有较低的 delay hit、redundant prefill 和 I/O stall，并且系统没有明显 queueing pressure。

W0 用于检查 scheduler optimization 在“不需要优化”的情况下是否引入额外开销或 latency regression。

### W1: Delay-hit-sensitive workload

选择 Experiment 1 中 delay hit 和 redundant prefill 最明显，同时仍处于稳定服务区域的 workload。

优先选择具有较强 locality、较短 reuse distance，并且多个相关请求容易在 cache 尚未 ready 时到达的 workload。

W1 主要用于评估 delay-hit mitigation。

### W2: Loading-balance-sensitive workload

选择具有明显 CPU→GPU cache loading activity，并且 baseline 中存在较高 I/O stall，但 delay hit 不占主导的 workload。

该 workload 应具有足够的请求异质性，使不同请求之间存在不同的 loading/computation 组合机会。

W2 主要用于评估 balanced batching。

### W3: Residual-stall-sensitive workload

选择 baseline 中存在明显 I/O stall、较高 stall variance 或 GPU idle interval 的 workload。

W3 用于判断即使经过更合理的 batch formation，是否仍然存在无法通过 batching 消除的 loading bubble，以及 bubble filling 是否能够利用这些空闲区间。

四个 workload 必须来自 Experiment 1 已经实际测量的 workload space。

每个 workload 保存对应的 Experiment 1 run identifier、locality condition、arrival-rate level、reuse-distance distribution 和选择依据。

不得在看到 Experiment 2 的性能结果后重新选择更有利的 workload。

## 4. 主消融配置

主实验使用四个 scheduler configuration。

| Configuration | Delay-hit mitigation | Balanced batching | Stall hiding |
|---|---:|---:|---:|
| S0 Baseline | × | × | × |
| S1 Delay-hit | ✓ | × | × |
| S2 Delay-hit + Balance | ✓ | ✓ | × |
| S3 Full scheduler | ✓ | ✓ | ✓ |

该设计与三个机制的处理顺序保持一致。

S0 → S1 用于观察 delay-hit mitigation 的增量效果。

S1 → S2 用于观察已经解决明显 delay-hit 后，balanced batching 能否进一步降低 loading-bound behavior。

S2 → S3 用于观察经过 batch balancing 后剩余的 I/O stall 是否还能通过 bubble filling 被利用。

四个 configuration 在 W0–W3 上全部运行。

主实验因此包含 4 个 representative workloads × 4 个 scheduler configurations，共 16 个主要实验条件。

## 5. 补充 leave-one-out 验证

单纯依赖逐步加入机制存在 attribution 风险。

某个机制的收益可能依赖前面的机制，因此 S1 → S2 或 S2 → S3 的增量不能完全等价于该组件自身的独立价值。

本实验增加少量 targeted leave-one-out validation，但不进行完整三机制 factorial sweep。

在对应机制最敏感的 representative workload 上补充：

- Full − Delay-hit mitigation；
- Full − Balanced batching；
- Full − Stall hiding。

每个 leave-one-out configuration 只在与该机制直接相关的一个或两个 workload 上运行。

该验证用于检查主消融中的机制归因，而不是形成第二套完整实验矩阵。

## 6. Workload 与运行条件

每个 representative workload 使用 Experiment 1 中已经冻结的 request trace。

同一个 workload 在不同 scheduler configuration 下保持以下条件一致：

- request set；
- request arrival timestamps；
- request ordering；
- context / prefix distribution；
- input length distribution；
- output length distribution；
- reuse opportunity；
- reuse-distance structure。

Scheduler 不得通过主动改变 offered workload 来获得性能优势。

如果某个 scheduler configuration 改变了实际 admission rate、effective concurrency 或其他 workload condition，该变化必须被记录，并在结果解释中明确区分。

## 7. 实验执行流程

### 第一阶段：Scheduler configuration validation

正式运行前验证四种 scheduler configuration 可以独立、稳定地启用。

实验确认每个 configuration 只改变预期的 scheduler mechanism，不同时改变 cache policy、I/O backend、batch-size limit 或其他系统参数。

实验确认关闭某个组件后不存在 silent fallback 到相似策略的情况。

### 第二阶段：执行主消融实验

依次在 W0、W1、W2 和 W3 上运行 S0–S3。

每个 condition 进行多次独立重复测量。

不同 scheduler configuration 的运行顺序采用交替或随机方式，避免机器长期状态变化系统性偏向某种 scheduler。

每轮正式实验重新建立规定的 cache 初始状态和 workload state。

所有 run 保存完整 configuration identifier 与 repetition index。

### 第三阶段：执行 targeted leave-one-out

根据预先确定的 workload-to-mechanism mapping，在对应 workload 上运行 Full-minus-component configuration。

这一阶段只验证主实验的因果归因，不重新寻找新的最佳 workload。

## 8. Delay-hit mitigation 观测设计

Delay-hit mitigation 的核心评价不能只依赖 throughput。

实验重点观察：

- delay-hit event；
- delay-hit affected request/token volume；
- redundant prefill；
- deferred request count；
- deferred request waiting time；
- realized cache reuse；
- queueing delay；
- TTFT distribution；
- throughput。

核心证据链为：

```text
delay-hit mitigation enabled
        ↓
fewer premature cache misses
        ↓
lower redundant prefill
        ↓
higher realized reuse
        ↓
TTFT / throughput change
```

如果 S1 相对 S0 提高 throughput，但 delay hit 和 redundant prefill 基本没有变化，则不能直接把性能提升归因于 delay-hit mitigation 的预期机制。

## 9. Balanced batching 观测设计

Balanced batching 的重点不是进一步提高 cache hit rate，而是改善一个 batch 中 computation 与 loading 的组合关系。

实验重点观察：

- batch loading requirement；
- batch computation amount；
- loading-bound batch fraction；
- cache-loading stall；
- non-overlapped I/O stall；
- GPU utilization；
- queueing delay；
- TTFT；
- throughput。

同时记录 request waiting behavior，检查为了构造更加平衡的 batch 是否导致部分请求被长期推迟。

核心证据链为：

```text
balanced batching enabled
        ↓
better loading / computation composition
        ↓
fewer severely loading-bound batches
        ↓
lower exposed I/O stall
        ↓
TTFT / throughput change
```

Balanced batching 不要求显著减少 cache loading volume。

如果 loading volume 基本相同，但 non-overlapped stall 明显下降，仍然属于符合机制预期的正结果。

## 10. Bubble filling / stall hiding 观测设计

Bubble filling 的作用对象是经过前两个机制后仍然存在的 residual loading stall。

实验重点观察：

- residual I/O stall；
- GPU idle interval；
- successfully filled bubble time；
- inserted useful computation；
- GPU utilization；
- prefill TTFT；
- throughput；
- decode latency / TPOT。

由于 bubble filling 可能通过插入其他 work 利用等待时间，本实验同时观察 decode-side latency。

不能通过明显损害已有 decode request 的 latency 来换取更好的 aggregate throughput。

核心证据链为：

```text
residual loading stall
        ↓
bubble filling
        ↓
useful computation during otherwise idle interval
        ↓
lower exposed GPU idle time
        ↓
throughput / TTFT change
```

如果 S3 相对 S2 的 throughput 提升来自更高的有效计算占用，同时 residual exposed stall 减少，则能够支持 stall-hiding mechanism 的解释。

## 11. 用户可见性能指标

所有 configuration 统一报告：

- throughput；
- P50 TTFT；
- P90 TTFT；
- P99 TTFT；
- queueing delay distribution；
- TPOT 或等价 decode latency metric。

平均 TTFT 不单独作为主要 latency 指标。

Scheduler 可能改善平均吞吐但恶化部分请求等待，因此 tail latency 与 throughput 必须同时报告。

## 12. Scheduler 安全性指标

为了防止 scheduler 通过牺牲公平性获得表面性能提升，同时记录：

- maximum request waiting time；
- queue-age distribution；
- starvation event；
- achieved request rate；
- effective concurrency；
- active-request preemption。

任何明显改变 workload admission 或导致 request starvation 的 configuration 均需要单独解释。

## 13. 分析一：Delay-hit mitigation

主要比较 S0 与 S1。

重点分析 W1，同时使用 W0 作为 control。

如果 W1 中 delay hit 和 redundant prefill 显著下降，并进一步改善 TTFT 或 throughput，而 W0 中收益很小，则说明 delay-hit mitigation 的收益具有明确 workload dependency。

如果 W0 和 W1 都没有明显 delay hit，则不能仅根据 throughput 波动声称该机制有效。

## 14. 分析二：Balanced batching

主要比较 S1 与 S2。

重点分析 W2。

判断 balanced batching 是否减少 loading-bound batch 和 exposed I/O stall。

同时检查 cache hit 和 transfer volume。

如果 transfer volume 基本不变，而 I/O stall 和 TTFT 下降，则说明收益主要来自 scheduling overlap，而不是意外改变了 cache behavior。

## 15. 分析三：Bubble filling

主要比较 S2 与 S3。

重点分析 W3。

判断 residual I/O stall 是否被有效计算覆盖，以及 GPU idle time 是否下降。

同时检查 decode latency，避免 throughput improvement 建立在明显恶化 decode QoS 的基础上。

## 16. 分析四：完整 scheduler 的互补性

最后比较 S0、S1、S2 和 S3。

分析三个组件是基本独立、相互补充、收益高度重叠，还是存在负面 interaction。

如果完整 scheduler 的收益主要由单个机制贡献，其余机制在现代 workload 中基本没有增量价值，则直接报告该结果。

实验不要求三个机制都取得明显收益。

## 17. Control workload regression 检查

W0 是本实验不可删除的 control workload。

如果 W0 本身不存在明显 delay hit、loading imbalance 或 I/O stall，则 scheduler optimization 理论上不应取得明显收益。

实验重点检查：

- TTFT 是否增加；
- queueing 是否增加；
- throughput 是否下降；
- scheduler overhead 是否出现。

如果优化机制在 W0 上造成明显 regression，则需要在最终 scheduler operating region 中明确排除这一 workload 区域。

## 18. 结果组织

本实验至少形成以下结果。

### Figure A: Scheduler progressive ablation

对 W0–W3 分别展示：

```text
Baseline
→ + Delay-hit mitigation
→ + Balanced batching
→ + Stall hiding
```

并同时报告 throughput 与 TTFT 的变化。

### Figure B: Mechanism-level attribution

分别展示：

- delay-hit mitigation → delay hit / redundant prefill；
- balanced batching → loading-bound batch / I/O stall；
- stall hiding → residual stall / GPU idle。

该图用于建立 mechanism metric 与 end-to-end performance 之间的证据链。

### Figure C: Latency safety

展示各 scheduler configuration 下：

- P50/P90/P99 TTFT；
- TPOT；
- queueing tail。

该图用于确认 throughput 收益没有建立在明显 latency regression 上。

### Workload × mechanism summary

最终形成：

| Workload | Dominant baseline pathology | Delay-hit mitigation | Balanced batching | Stall hiding | Full scheduler |
|---|---|---|---|---|---|
| W0 Control | weak | ... | ... | ... | ... |
| W1 Delay-hit-sensitive | delay hit / redundant prefill | ... | ... | ... | ... |
| W2 Loading-sensitive | loading imbalance | ... | ... | ... | ... |
| W3 Stall-sensitive | residual I/O stall | ... | ... | ... | ... |

表中的判断依据来自 mechanism metric 与 end-to-end metric，而不是单独根据 speedup 分类。

## 19. 结果判断逻辑

### 情况 A：三种机制表现出清晰分工

Delay-hit mitigation 主要降低 redundant prefill，balanced batching 主要降低 loading-bound stall，stall hiding 主要降低 residual GPU idle。

完整 scheduler 获得进一步端到端收益。

该结果说明三阶段 control-plane 设计在现代 workload 中仍然基本成立。

### 情况 B：只有部分机制仍然有效

例如 delay-hit mitigation 仍然明显有效，但 balanced batching 或 stall hiding 增量很小。

该结果说明现代模型、runtime 或 I/O 系统已经改变 scheduler bottleneck 的组成。

后续 Experiment 4 应缩小这些机制的有效 operating region，而不是继续把所有 scheduler optimization 视为同等重要。

### 情况 C：机制指标改善，但端到端收益有限

Scheduler pathology 明显减少，但 TTFT 和 throughput 改善很小。

该结果说明这些 pathology 当前不是 dominant performance bottleneck。

不能仅根据内部指标改善声称 scheduler optimization 具有显著 serving value。

### 情况 D：Full scheduler 出现负面 interaction

单独组件产生收益，但完整组合收益降低，或者 tail latency 恶化。

该结果说明不同 scheduler mechanism 之间存在 interaction，需要通过 leave-one-out result 定位冲突来源。

### 情况 E：所有 scheduler mechanism 收益都很弱

W1–W3 中内部 pathology 与端到端性能都基本不变。

该结果说明原系统所针对的 control-plane bottleneck 在当前现代 serving stack 中已经明显减弱。

该结论本身构成有效实验结果。

## 20. 实验边界

Experiment 2 只研究 scheduler component attribution。

本实验不重新 sweep locality × arrival rate，不研究 extreme same-context concurrency，也不确定完整 scheduler 的全局 operating region。

Locality × load 的完整基础空间已经由 Experiment 1 建立。

Same-context concurrency 由 Experiment 3 单独研究。

最终 workload-to-mechanism operating region 由 Experiment 4 综合 Experiments 1–3 得出。
