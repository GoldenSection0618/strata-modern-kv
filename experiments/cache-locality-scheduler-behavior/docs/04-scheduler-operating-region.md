# Experiment 4: Scheduler Operating Region

## 1. 实验目标

本实验用于确定 Strata scheduler optimization 在现代 hybrid LLM serving 系统中的实际有效范围。

实验复用 Experiments 1–3 已经获得的 baseline profiling、component ablation 和 same-context concurrency 结果，并只补充少量边界 workload。

本实验最终回答四个问题：

1. 哪些 workload 中 baseline scheduler 已经足够，不需要额外 scheduler optimization；
2. 哪些 workload 中 delay-hit mitigation、balanced batching 或 stall hiding 能够产生稳定收益；
3. 完整 scheduler 的收益主要集中在哪些 locality 与 load 区域；
4. same-context concurrency 是否构成一个需要特殊处理的独立 workload region。

本实验不重新验证每个 scheduler mechanism 的内部原理。机制 attribution 已经由 Experiment 2 完成。

## 2. 实验总体思路

Experiment 4 以 Experiment 1 建立的 locality × arrival-rate space 作为基础坐标系。

主要 workload space 表示为：

```text
cache locality
        ×
system load
```

Experiment 3 的 same-context concurrency 作为特殊 workload dimension 单独叠加分析，而不扩展成完整三维 sweep。

因此本实验形成两类结果：

1. 一张一般 workload 下的 locality × load scheduler operating map；
2. 一组 hot-context concurrency 下的补充 operating rules。

这种设计避免形成 locality × load × fan-in 的大规模组合实验。

## 3. 数据来源

Experiment 4 优先复用前三组已经获得的数据。

### Experiment 1 提供

- locality × arrival-rate baseline surface；
- delay hit；
- redundant prefill；
- queueing delay；
- I/O stall；
- TTFT；
- throughput。

这些结果用于确定 scheduler pathology 出现在哪里。

### Experiment 2 提供

- delay-hit mitigation 的增量收益；
- balanced batching 的增量收益；
- stall hiding 的增量收益；
- full scheduler 的总体收益；
- control workload 上的 regression 情况。

这些结果用于确定不同 mechanism 应该被映射到哪些 pathology。

### Experiment 3 提供

- same-context fan-in 对 reuse realization 的影响；
- high-concurrency 下的 delay hit；
- redundant prefill；
- tail TTFT；
- delay-hit mitigation 与 full scheduler 的收益。

这些结果用于形成 hot-context workload 的特殊规则。

## 4. Operating region 定义

本实验不根据单个 throughput speedup 判断 scheduler 是否有效。

一个 scheduler mechanism 被认为在某个 workload region 中具有实际价值，需要同时满足以下方向性的证据：

1. 对应的 baseline pathology 明确存在；
2. scheduler mechanism 能够改善该 pathology；
3. 改善能够传导到 TTFT、throughput 或 tail latency；
4. 没有产生明显的 fairness、queueing 或 decode-side regression；
5. 结果在重复实验中保持稳定。

因此 operating region 表示：

> 某个 scheduler mechanism 在什么 workload 条件下既具有机制上的作用对象，又能够产生实际 serving benefit。

## 5. 初始 workload region 划分

根据 Experiment 1 的结果，将 locality × load space 初步划分为若干 region。

### R0：Low-pressure region

系统处于 Low load，或者 Medium load 下仍有明显资源余量。

Delay hit、redundant prefill、I/O stall 和 queueing 均较弱。

该区域用于判断 scheduler optimization 是否基本没有必要。

### R1：Locality-sensitive region

随着 reuse distance 增大，delay hit、redundant prefill 或 realized reuse 出现明显变化。

该区域说明 request ordering 已经开始影响 cache reuse efficiency。

### R2：Load-sensitive region

Locality 不一定很差，但系统接近 High load 后 queueing、batch imbalance 或 I/O stall 明显上升。

该区域用于评估 balanced batching 和 stall hiding 的主要价值。

### R3：Combined-pressure region

系统同时具有较差 locality 和较高 load。

该区域中多种 scheduler pathology 同时存在。

该区域是 full scheduler 最可能产生明显总体收益的 workload。

### R4：Saturation region

系统进入持续 backlog 或明显 overload。

该区域单独标记。

R4 不直接纳入 scheduler 正常 operating region，因为此时性能主要受到系统 capacity 限制。

如果 scheduler 在 R4 中仍然产生收益，可以作为 robustness 结果报告，但不能据此扩大其正常适用范围。

## 6. Region 划分方式

Region 不根据预设的 Min / Shuffle / Max 或 Low / Medium / High 标签机械确定。

Experiment 1 的实际 measurement 用于决定 region boundary。

例如：

- 如果 Shuffle 与 Min distance 实际表现接近，则两者可以处于同一 region；
- 如果 High load 仍然没有明显 queueing 或 stall，则不能因为名称是 High 就自动划入压力区域；
- 如果 Medium load 已经出现明显 pathology，则 operating boundary 可以提前出现。

因此 workload label 用于实验设计，actual observed behavior 用于最终 region classification。

## 7. 初始 scheduler map 构建

首先不运行新的实验。

实验根据 Experiments 1–3 的已有结果建立 provisional scheduler map。

每个 workload point 记录：

- locality；
- offered / achieved load；
- reuse distance；
- baseline pathology；
- dominant pathology；
- delay-hit mitigation benefit；
- balanced batching benefit；
- stall-hiding benefit；
- full-scheduler benefit；
- latency regression；
- validity status。

然后按照 dominant pathology 对已有 point 分类。

## 8. Scheduler mechanism mapping

### Delay-hit mitigation region

该机制的候选 operating region 满足：

- reusable state 存在；
- matching state 尚未 ready 时会出现新的相关请求；
- baseline 存在 delay hit 或 redundant prefill；
- mitigation 后 realized reuse 提高。

该 region 主要与 locality、reuse timing 和 same-context overlap 相关。

### Balanced batching region

该机制的候选 operating region 满足：

- cache/state loading activity 明显；
- baseline 存在 loading-bound batch；
- loading/computation ratio 在请求间具有差异；
- balanced batching 能降低 exposed I/O stall。

该 region 主要与较高 serving load 和 workload heterogeneity 相关。

### Stall-hiding region

该机制的候选 operating region 满足：

- 前两阶段 scheduler 后仍存在 residual I/O stall；
- GPU 存在可利用 idle interval；
- 系统中同时存在可以插入的有效计算工作。

如果 workload 本身没有明显 residual stall，则该机制不应被划入有效区域。

### Full scheduler region

Full scheduler 的 operating region 满足：

- 至少一种 scheduler pathology 明确存在；
- 多种 mechanism 具有互补收益，或者完整配置比最有效单组件进一步改善性能；
- tail latency 和 fairness 不发生明显 regression。

## 9. Boundary-point selection

初始 scheduler map 建立后，只对结论最不确定的边界区域补充实验。

Boundary point 的选择规则在补充实验开始前冻结。

优先选择以下三类点。

### B1：No-benefit → Benefit boundary

选择 scheduler 从几乎没有收益转变为稳定收益的相邻 workload。

该实验用于确定 operating region 的起始位置。

### B2：Single-mechanism → Full-scheduler boundary

选择单个 mechanism 已经有效，但完整 scheduler 是否进一步有价值仍不明确的 workload。

该实验用于判断是否真的需要完整 scheduler。

### B3：Benefit → Saturation boundary

选择 High stable 与 Overload 之间的边界 workload。

该实验用于区分：

- scheduler 无法继续改善；
- 系统已经进入 capacity-limited regime。

## 10. 补充实验规模

Experiment 4 不进行新的完整网格搜索。

建议选择约 4–6 个 boundary points。

每个 boundary point 只运行必要的 scheduler configuration。

例如：

```text
Baseline
vs
candidate mechanism
vs
Full scheduler
```

如果已有 Experiment 2 数据已经足够判断某个 mechanism，则不重复运行。

因此 Experiment 4 的新增 workload 数量明显小于 Experiments 1–3。

## 11. Boundary refinement

对于每个 boundary point，在原有 locality 或 load 条件附近增加一个较小扰动。

例如：

```text
Point A
↓
slightly higher load
```

或：

```text
Point A
↓
slightly worse locality
```

该设计用于判断 operating boundary 是否稳定。

如果 scheduler benefit 只存在于一个极窄的 workload point，而相邻条件立即消失，则该区域不应被解释为稳定 operating region。

## 12. Scheduler benefit 指标

每个 workload point 统一报告：

### Scheduler pathology

- delay hit；
- redundant prefill；
- reuse realization；
- loading-bound batch；
- exposed I/O stall；
- GPU idle。

### End-to-end performance

- throughput；
- P50 TTFT；
- P90 TTFT；
- P99 TTFT；
- request completion time；
- TPOT 或等价 decode metric。

### Safety

- queueing tail；
- maximum request waiting time；
- starvation；
- active-request preemption；
- achieved request rate。

## 13. Benefit classification

每个 scheduler × workload point 最终被分类为四种状态之一。

### Effective

对应 pathology 明显改善，并产生稳定的端到端收益，没有明显 regression。

### Mechanism-only

内部 scheduler pathology 得到改善，但端到端性能收益有限。

该状态说明机制有效，但当前 pathology 不是 dominant bottleneck。

### Neutral

内部指标和端到端性能都基本没有变化。

该 workload 不属于该机制的有效 operating region。

### Regressive

scheduler 导致 queueing、tail latency、decode latency、fairness 或 throughput 明显恶化。

该 workload 明确排除在对应机制的 operating region 之外。

## 14. 分析一：Locality 方向

在固定 load 条件下，从高 locality 向低 locality 比较。

分析：

```text
Min distance
→ Shuffle
→ Max distance
```

以及必要的 boundary refinement point。

重点判断：

- delay-hit mitigation 的收益是否随 locality 恶化增强；
- full scheduler 是否在某个 reuse-distance 区域开始产生稳定收益；
- locality 极差以后 cache reuse opportunity 是否已经低到 scheduler 无法继续发挥作用。

因此 scheduler benefit 不一定随 locality 单调增加。

可能出现：

```text
high locality
scheduler unnecessary

medium locality
scheduler valuable

extremely poor locality
little reusable state remains
scheduler value decreases
```

这种非单调结果需要保留，而不能强行解释为 locality 越差 scheduler 越有价值。

## 15. 分析二：Load 方向

在固定 locality 下比较：

```text
Low
→ Medium
→ High
→ Overload
```

重点判断：

- balanced batching 在什么 load 开始产生稳定收益；
- stall hiding 在什么 load 下开始具有足够 residual stall；
- scheduler benefit 是否在系统接近 saturation 后停止增长。

最终需要区分三个阶段：

```text
resource-rich
→ scheduler-sensitive
→ capacity-limited
```

## 16. 分析三：Locality × Load interaction

重点分析四个 corner regions：

| | Low load | High load |
|---|---|---|
| High locality | A | B |
| Low locality | C | D |

A 用于表示 scheduler optimization 的最低需求区域。

B 用于判断高 load 是否单独足以产生 scheduler value。

C 用于判断较差 locality 是否在低负载下仍然能够被系统余量吸收。

D 用于观察 locality 和 load 同时恶化后的完整 scheduler 价值。

如果 D 的收益显著大于 B 和 C，则说明 locality 与 load 存在明显 interaction。

如果 B 和 C 已经分别解释全部收益，则不需要强行声称存在交互作用。

## 17. Same-context concurrency operating rule

Experiment 3 的结果不扩展成完整 locality × load × fan-in surface。

Experiment 4 将 hot-context workload 单独形成一条 operating rule。

例如最终可能得到：

```text
normal fan-in
→ use general locality × load scheduler map

high same-context fan-in
→ prioritize delay-hit mitigation

high fan-in + high global load
→ full scheduler becomes valuable
```

该规则必须由 Experiment 3 的 C0–C3 结果支持。

如果现代 runtime 对 same-context concurrency 已经具有良好 coordination，则不单独建立 hot-context scheduler region。

## 18. Regression region

Experiment 4 必须明确标记 scheduler 不值得启用的 workload。

重点包括：

- low load；
- pathology 很弱；
- cache 已经 GPU-resident；
- I/O stall 很低；
- scheduler deferral 带来的 queueing 大于避免的 redundant work。

这些结果不能只写成“收益较小”。

如果 scheduler 产生实际 regression，应明确标记为：

```text
scheduler-not-recommended region
```

这对于形成完整 operating region 与证明机制边界同样重要。

## 19. Model-specific 与 general conclusion

Experiment 4 首先在本实验组的 primary model / primary hardware 上形成 scheduler operating region。

不在本实验中重新进行完整 cross-model × hardware sweep。

后续 Model and Hardware Generalization 实验选择这里得到的代表性 workload region，包括：

- no-benefit point；
- clear-benefit point；
- boundary point；
- hot-context point。

然后在第二模型或第二硬件上验证 region-level conclusion 是否保持。

因此 Experiment 4 输出的是：

> primary system 上的完整 operating map。

后续 generalization 实验回答：

> 这张 map 的方向性边界是否跨模型和硬件稳定。

## 20. 实验执行流程

### 第一阶段：汇总 Experiments 1–3

整理所有有效 run。

统一 workload identifier、scheduler configuration、metrics 和 validity status。

建立 provisional workload × mechanism table。

### 第二阶段：建立 provisional operating map

根据 baseline pathology 与 scheduler benefit 对已有 workload point 分类。

形成初始 locality × load map。

标记：

- clear effective region；
- clear neutral region；
- regression region；
- uncertain boundary region。

### 第三阶段：冻结 boundary points

只从 uncertain boundary region 选择 4–6 个代表性 workload。

Selection rule 与 workload configuration 在补充实验前冻结。

### 第四阶段：运行 boundary validation

运行必要的 Baseline、candidate mechanism 和 Full scheduler。

每个 point 进行多次重复测量。

### 第五阶段：形成最终 operating region

整合所有实验结果。

将每个 workload region 分类为：

- scheduler unnecessary；
- delay-hit mitigation dominated；
- balanced batching dominated；
- stall-hiding dominated；
- full scheduler beneficial；
- capacity limited；
- scheduler regressive。

## 21. 结果组织

### Figure A：Locality × Load Operating Map

横轴表示 locality / reuse distance。

纵轴表示 normalized system load。

在二维空间中标记：

- scheduler unnecessary；
- delay-hit-sensitive；
- balance-sensitive；
- stall-sensitive；
- full-scheduler-beneficial；
- saturation。

这是 Experiment 4 的核心结果。

### Figure B：Mechanism Benefit Map

分别为：

- delay-hit mitigation；
- balanced batching；
- stall hiding

形成三张小型 operating map。

该结果用于展示不同 mechanism 的适用范围并不完全相同。

### Figure C：Boundary Validation

展示 selected boundary points 周围：

```text
baseline
candidate mechanism
full scheduler
```

的 TTFT 与 throughput。

该结果用于证明 operating-region boundary 不是由偶然的单点结果产生。

### Figure D：Same-context concurrency rule

展示 C0–C3 下：

- Baseline；
- Delay-hit mitigation；
- Full scheduler。

用于形成 hot-context workload 的补充 operating rule。

### Table：Final Scheduler Decision Matrix

最终形成：

| Workload condition | Dominant pathology | Recommended mechanism | Expected value |
|---|---|---|---|
| Low load + good locality | None | Baseline | Scheduler optimization unnecessary |
| High locality + delay-hit overlap | Delay hit | Delay-hit mitigation | Reduce redundant work |
| High load + loading imbalance | Exposed loading stall | Balanced batching | Improve overlap |
| Residual I/O stall | GPU bubble | Stall hiding | Utilize idle compute |
| Poor locality + high stable load | Multiple | Full scheduler | Largest combined benefit |
| Same-context high fan-in | Delay-hit race | Delay-hit / Full | Preserve reuse |
| Sustained overload | Capacity saturation | No scheduler claim | Capacity limited |

表中最终内容必须由实际实验结果填写，不能把上述预期关系直接作为实验结论。

## 22. 结果判断逻辑

### 情况 A：存在清晰 operating region

Scheduler 在 Low load 下收益有限，在某些 locality × load 条件下开始稳定产生收益，并在 saturation 前形成清晰有效区域。

该结果说明 Strata scheduler mechanism 在现代系统中仍然有效，但其价值具有明确 workload dependency。

### 情况 B：只有少数机制保留明显 operating region

例如 delay-hit mitigation 仍有明确区域，而 balanced batching 和 stall hiding 大部分 workload 中接近 neutral。

该结果说明现代 serving stack 已经改变了 scheduler bottleneck 的组成。

最终结论应明确缩小 Strata scheduler optimization 的适用范围。

### 情况 C：Scheduler pathology 存在，但 operating region 很小

部分 workload 内部指标能够改善，但只有非常有限区域能够转化为端到端收益。

该结果说明现代系统已经能够吸收大多数原论文所关注的 scheduler overhead。

### 情况 D：Full scheduler 几乎始终优于 baseline

如果从 Medium 到 High stable load 的多个 locality condition 中都出现稳定收益，同时没有低负载 regression，则说明完整 scheduler 仍然具有较宽的 operating region。

后续 generalization 应重点验证这一结果是否跨模型与硬件保持。

### 情况 E：Scheduler 主要在 saturation 后才产生收益

如果明显收益只存在于持续 Overload 区域，则不能据此声称 scheduler 在正常 serving workload 中仍具有重要价值。

该结果更接近 overload robustness，而不是正常 operating-region benefit。

## 23. 实验边界

Experiment 4 是本组 scheduler 实验的 synthesis experiment。

本实验不重新进行：

- 完整 locality × arrival-rate sweep；
- 完整 scheduler component ablation；
- 完整 same-context concurrency sweep；
- model × hardware generalization。

本实验只补充决定 operating-region boundary 所必需的少量 workload。

Experiments 1–4 最终形成完整逻辑：

```text
Experiment 1
问题在哪里
        ↓
Experiment 2
什么机制解决什么问题
        ↓
Experiment 3
高共享 context 压力下是否仍然成立
        ↓
Experiment 4
什么 workload 下应该使用什么 scheduler mechanism
```

Experiment 4 的最终产物不是另一个 speedup figure，而是一张能够概括 scheduler applicability、mechanism attribution 和 workload boundary 的 operating map。
