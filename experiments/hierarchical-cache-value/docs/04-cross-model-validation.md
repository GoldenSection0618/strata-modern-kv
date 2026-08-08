# Experiment 4: Cross-Model Hierarchical Cache Validation

## 1. 实验目标

本实验用于验证 Experiments 1–3 得到的 hierarchical cache 结论在不同现代 hybrid 模型上是否保持稳定。

实验不重新进行完整的 GPU cache pressure sweep 或 prefix reuse sweep，而是从前述实验中选择具有代表性的 workload regime，在 Qwen3.5-9B 与 Gemma 4 12B 上进行 matched validation。

本实验主要回答三个问题：

1. hierarchical cache 的基础收益是否同时存在于两种不同现代模型上；
2. cache pressure 和 prefix reuse 对 hierarchy 收益的影响方向是否具有跨模型稳定性；
3. 两个模型的 hierarchy value region 是否存在明显差异，以及这些差异能否由实际 cache/state behavior 解释。

本实验只验证跨模型稳定性，不根据两个模型之间的性能差异直接声称 attention architecture 是差异产生的因果因素。

## 2. 实验对象

实验分别使用：

- Qwen3.5-9B；
- Gemma 4 12B。

两个模型均保持各自原生的 cache/state 组织方式，不将其人为转换为统一的 dense-attention KV cache。

实验关注的是每个模型在自身原生 serving path 下 hierarchical cache 是否有价值，而不是比较两个模型谁具有更高的绝对性能。

因此跨模型比较主要使用：

- cache pressure regime；
- reuse regime；
- relative recomputation reduction；
- relative TTFT benefit；
- relative throughput gain。

不直接使用绝对 cache size、TTFT 或 throughput 作为跨模型优劣判断。

## 3. 实验设计原则

Experiment 4 不重新执行 Experiments 1–3 的全部参数组合。

前述实验已经分别研究：

```text
Experiment 1: hierarchy 是否存在基础收益
Experiment 2: GPU cache pressure 如何影响收益
Experiment 3: prefix reuse 如何影响收益
```

Experiment 4 只选择能够代表这些结论的少量配置进行跨模型验证。

代表性配置必须按照预先定义的规则选择，不能在观察第二个模型结果之后重新挑选有利实验点。

这样可以避免通过选择性配置人为制造跨模型一致性。

## 4. Representative Regime 选择

本实验至少选择三个代表性 regime。

### Regime A: Low-value control

该配置代表 hierarchical cache 理论收益较低的区域。

配置满足：

- GPU cache pressure 较低；
- prefix reuse 较低或中等；
- GPU-only 已经能够覆盖主要 reusable working set。

该配置作为 negative control。

实验预期 hierarchy 在该区域不会产生明显性能收益。

### Regime B: Value onset

该配置代表 Experiments 2 或 3 中 hierarchical cache 开始出现稳定收益的区域。

配置满足：

- GPU cache 已经产生稳定 eviction；
- 被驱逐状态存在实际 reuse；
- hierarchical cache 能够获得稳定 CPU hit；
- CPU-GPU traffic 尚未完全成为系统瓶颈。

该 regime 是 Experiment 4 最重要的验证点。

它用于判断 hierarchical cache 从“没有必要”转变为“值得使用”的现象是否能够在两个模型上同时观察到。

### Regime C: High-value / high-pressure

该配置代表 hierarchy 具有较强使用动机的区域。

配置满足：

- GPU cache 明显无法覆盖 working set；
- prefix reuse 较高；
- GPU-only 中存在明显 recomputation。

该配置用于观察两个模型在 hierarchy 需求较强时是否都能够从 CPU tier 获益，以及收益最终是否受到 CPU-GPU restore cost 限制。

## 5. Regime 的跨模型匹配方式

两个模型不强制使用完全相同的绝对 GPU cache size。

不同模型的 cache/state footprint 和 retention behavior 不同。如果简单设置相同的 GPU cache GB 数量，两个模型实际承受的 cache pressure 可能完全不同。

因此跨模型实验按照系统状态匹配，而不是按照绝对容量匹配。

例如 Regime B 在两个模型上都应代表：

> GPU cache 已出现稳定 eviction，但尚未进入严重 thrashing。

Regime C 在两个模型上都应代表：

> GPU cache 明显无法覆盖 active reusable working set。

GPU cache budget 可以根据 Experiment 2 的实际测量结果分别确定。

每个模型都需要报告该 regime 对应的真实 GPU hit、eviction 和 working-set coverage，证明两个配置处于可比较的 pressure region。

## 6. Prefix Reuse 的跨模型匹配

两个模型使用相同结构的 workload trace。

跨模型保持：

- 相同的请求数量；
- 相同的 prefix group structure；
- 相同的 prefix reuse frequency；
- 相同的 reused request proportion；
- 尽可能一致的 input/output token scale。

由于 tokenizer 和模型实现不同，不要求 raw text 编码后得到完全相同的内部表示。

跨模型比较以实际测得的 reuse opportunity 和 reused token/state proportion 为准。

这样可以保证比较的是相似的 workload reuse structure，而不是表面相同的文本输入。

## 7. 对照配置

每一个 representative regime 都分别运行两种 cache architecture。

### GPU-only

系统只使用 GPU cache。

GPU eviction 后状态失效，后续重复访问需要重新计算。

### GPU + CPU hierarchical cache

GPU 使用与 GPU-only 相同的 cache budget。

GPU eviction 后允许 reusable state 保存在 CPU tier，并在后续访问时恢复。

最终实验矩阵为：

```text
Qwen3.5-9B
├── Regime A
│   ├── GPU-only
│   └── Hierarchical
├── Regime B
│   ├── GPU-only
│   └── Hierarchical
└── Regime C
    ├── GPU-only
    └── Hierarchical

Gemma 4 12B
├── Regime A
│   ├── GPU-only
│   └── Hierarchical
├── Regime B
│   ├── GPU-only
│   └── Hierarchical
└── Regime C
    ├── GPU-only
    └── Hierarchical
```

Experiment 4 只完成少量代表性配置，不重新复制完整 parameter sweep。

## 8. Cache Initial State

主实验统一使用 warm-cache steady-state。

每个 regime 在正式测量前执行与其 workload 相匹配的 cache population 阶段。

Warm-up 不计入正式结果。

使用 warm steady-state 可以避免 cold-start initialization 对模型间比较产生额外干扰，并使实验重点集中于长期 serving 状态下 hierarchy 的稳定价值。

Experiment 1 已经单独研究 cold-cache 与 warm-cache，因此本实验不重复该维度。

## 9. 实验执行过程

首先从 Experiments 2 和 3 的结果中按照预定义标准确定 Regime A、B、C。

随后分别在两个模型上建立对应的 matched pressure 和 reuse condition。

每个 regime 首先验证实际 workload 和 cache behavior 是否符合预期。

需要确认：

- actual prefix reuse；
- GPU cache pressure；
- GPU eviction；
- GPU hit；
- CPU hit opportunity。

只有实际系统状态满足 regime 定义，该 run 才进入正式跨模型比较。

每一个有效配置进行多次独立重复实验。

GPU-only 与 hierarchical 使用完全相同的 workload trace。

不同模型分别进行独立 warm-up 和 cache initialization。

正式实验记录完整 runtime 和 workload metadata。

## 10. 核心测量指标

### 10.1 Cache / State Footprint

记录每个模型在代表性 workload 下的实际 cache/state footprint。

能够分项时分别报告不同 state type。

该指标用于解释为什么相同 workload structure 在两个模型上可能产生不同 GPU cache pressure。

Experiment 4 不要求两个模型具有相同 state size。

### 10.2 GPU Cache Hit 与 Eviction

记录：

- GPU cache hit；
- GPU eviction；
- GPU-resident reusable state。

这些指标用于验证两个模型是否确实处于预期的 Low-value、Value-onset 或 High-pressure regime。

### 10.3 CPU Cache Hit

Hierarchical 配置记录 CPU cache hit。

重点比较 CPU tier 对两个模型有效 reuse 的贡献。

如果两个模型拥有相似 reuse workload，但 CPU hit 明显不同，则进一步结合各自 state footprint 和 GPU residency behavior 分析。

### 10.4 Recomputation

记录 GPU-only 与 hierarchical 配置下的 recomputation。

跨模型主要比较：

```text
relative recomputation reduction
=
(recompute_GPU-only - recompute_hierarchical)
/
recompute_GPU-only
```

相对指标比绝对计算时间更适合判断 hierarchy 是否在两个模型中发挥相同系统作用。

### 10.5 CPU-GPU Traffic

记录 hierarchical cache 引入的 CPU-GPU traffic。

同时记录 CPU restore 所对应的有效 reused state。

该结果用于判断不同模型中，为避免相同程度的 recomputation，需要支付多少额外 data movement cost。

### 10.6 TTFT

记录 GPU-only 与 hierarchical cache 的 TTFT。

每个模型分别报告 median、P90 和 P99。

跨模型主要比较 relative TTFT improvement：

```text
relative TTFT improvement
=
(TTFT_GPU-only - TTFT_hierarchical)
/
TTFT_GPU-only
```

不使用两个模型之间的绝对 TTFT 差值作为 hierarchy 优劣判断。

### 10.7 Throughput

记录 steady-state throughput。

跨模型主要比较：

```text
throughput gain
=
throughput_hierarchical
/
throughput_GPU-only
- 1
```

该指标用于判断 hierarchy 是否在两个模型上都能够把减少的 recomputation 转化为实际 serving capacity。

## 11. 第一层结果：方向一致性

首先不比较收益大小，只检查结果方向。

对于三个 regime，分别判断两个模型是否表现为：

| Regime | 预期现象 |
|---|---|
| A | hierarchy 收益很小或不存在 |
| B | hierarchy 开始产生稳定收益 |
| C | hierarchy 收益进一步增强，或受到 transfer 限制 |

如果两个模型都表现出相同的基本趋势，则说明 hierarchical cache value 随 cache pressure 和 reuse 改变的基本规律具有一定跨模型稳定性。

这一结论比要求两个模型获得相同百分比的加速更加合理。

## 12. 第二层结果：收益大小比较

在确认方向之后，再比较 hierarchy 的收益幅度。

形成：

> Regime → relative TTFT improvement

和：

> Regime → throughput gain

两个模型分别绘制。

如果两者趋势相同但收益大小不同，则结合：

- cache/state footprint；
- GPU hit；
- CPU hit；
- recomputation reduction；
- restore traffic；

解释差异来源。

收益大小不同本身不代表某种 architecture 更适合 hierarchical cache。

## 13. 第三层结果：机制一致性

Experiment 4 最重要的分析不是只确认两条性能曲线是否相似，而是验证相同的机制链是否存在。

对于每个模型分别检查：

```text
GPU pressure / reuse increase
            ↓
more reusable state misses GPU
            ↓
CPU-tier hit increases
            ↓
recomputation decreases
            ↓
CPU-GPU traffic increases
            ↓
TTFT / throughput changes
```

如果两个模型都呈现这条基本因果链，则 hierarchy 的系统作用具有较强的跨模型证据。

如果某个模型在其中某个阶段发生中断，则需要明确指出其 bottleneck 所在位置。

## 14. 结果判断逻辑

### 情况 A：两个模型表现出一致 value region

如果两个模型均表现为：

- Low pressure / low reuse 下收益有限；
- eviction 与 reuse 增加后 hierarchy 开始产生收益；
- 高压力下收益继续增加或受到 transfer 限制；

则可以得出：

> hierarchical cache 的基础价值并非只存在于单个模型，其收益主要受 GPU cache pressure、reuse opportunity 和 restore cost 的共同控制。

### 情况 B：趋势一致，但收益大小明显不同

如果两个模型的 value onset 方向一致，但 TTFT 或 throughput gain 差异明显，则说明 hierarchical cache 的总体机制具有跨模型稳定性，而收益幅度具有 model-dependent 特征。

此时需要使用实际 cache/state footprint、GPU residency 和 transfer behavior 解释差异。

不能仅凭模型名称或 attention 类型进行因果归因。

### 情况 C：一个模型明显受益，另一个模型基本无收益

如果在 matched workload regime 下只有一个模型获得明显收益，则需要首先检查：

1. 两个模型实际 GPU cache pressure 是否真正匹配；
2. CPU tier 是否实际缓存了可复用 state；
3. 两个模型的 reusable state volume 是否相近；
4. CPU-GPU restore 是否成为其中一个模型的主要瓶颈；
5. runtime 是否完整支持对应模型的 cache/state restore path。

只有排除这些系统因素后，才能将结果报告为模型相关差异。

仍然不能仅凭两种模型的比较证明差异由 attention architecture 单独导致。

### 情况 D：两个模型都几乎无收益

如果即使在 High-pressure / High-reuse regime 下两个模型都没有明显 TTFT 或 throughput 收益，则说明在当前现代模型与平台上，hierarchical cache 至少在当前实现和 workload 范围内缺乏明显的端到端价值。

随后需要结合 CPU hit 和 recomputation reduction 区分：

- reusable state 本身已经显著减少；
- GPU cache 已能覆盖主要 reuse；
- CPU restore cost 过高；
- runtime hierarchy implementation 效率不足。

这几种解释需要分别报告。

## 15. 与 Experiments 1–3 的关系

四组实验形成完整逻辑链：

```text
Experiment 1
Hierarchical cache 有没有基础收益？
        ↓
Experiment 2
收益在什么 GPU cache pressure 下出现？
        ↓
Experiment 3
收益需要什么程度的 prefix reuse？
        ↓
Experiment 4
上述规律能否跨现代模型保持稳定？
```

Experiment 4 不增加新的 workload 自变量。

它只验证前三个实验得到的主要结论是否能够从一个模型推广到另一个模型。

## 16. 实验边界

本实验不进行完整的 context length sweep。

本实验不进行完整的 GPU cache budget sweep。

本实验不进行完整的 prefix reuse sweep。

本实验不改变 scheduler strategy。

本实验不比较不同硬件平台。

硬件差异留给后续独立的模型与硬件泛化实验处理。

Experiment 4 只负责 cross-model validation。

## 17. 最终结果组织

Experiment 4 最终形成一张核心跨模型表格：

| Regime | Model | GPU pressure | CPU-tier contribution | Recomputation reduction | TTFT improvement | Throughput gain |
|---|---|---:|---:|---:|---:|---:|
| A | Qwen3.5 | ... | ... | ... | ... | ... |
| A | Gemma 4 | ... | ... | ... | ... | ... |
| B | Qwen3.5 | ... | ... | ... | ... | ... |
| B | Gemma 4 | ... | ... | ... | ... | ... |
| C | Qwen3.5 | ... | ... | ... | ... | ... |
| C | Gemma 4 | ... | ... | ... | ... | ... |

同时形成一张核心图：

> Representative workload regime → hierarchical cache relative benefit

分别展示两个模型。

最终结论应明确区分：

1. 哪些规律在两个模型上都成立；
2. 哪些收益幅度具有明显 model-dependent 特征；
3. 哪些结果只能作为关联观察，不能归因于 attention architecture。

本实验最终回答：

> **Hierarchical context caching 在现代 hybrid 模型上的价值是否具有跨模型稳定性，以及其适用边界是否主要由 cache pressure、reuse 和 data movement cost 决定。**
