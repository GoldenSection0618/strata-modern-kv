# Experiment 1: Cross-model Mechanism Generalization

## 1. 实验目标

本实验在固定 A100 40GB 的条件下，对 Qwen3.5-9B 与 Gemma 4 12B 执行统一的代表性系统测试。

本实验用于验证三类结论：

1. 两种现代 hybrid model 上是否仍然存在可观测的 cache/state capacity、data movement 或 scheduling bottleneck；
2. 模型变化后，主要 bottleneck 的位置与严重程度是否发生变化；
3. Hierarchical Cache、I/O Optimization 与 Scheduler Optimization 所针对的机制是否仍然能够产生与其机制目标一致的收益。

本实验只形成 cross-model robustness conclusion 和 serving-state behavior correlation。模型之间的差异不直接归因为 attention architecture。

所有正式比较遵循 [`00-common-conventions.md`](00-common-conventions.md) 与仓库根目录 [`../../../docs/TECHNICAL_BASELINE.md`](../../../docs/TECHNICAL_BASELINE.md)。

## 2. 实验总体设计

实验固定使用 A100 40GB 作为参考硬件，避免模型变量与硬件变量同时变化。

Qwen3.5-9B 与 Gemma 4 12B 使用相同的 experiment software revision、统一的 workload-generation rule、统一的 measurement rule 和统一的结果处理流程。

实验不重新执行前五组实验的完整 parameter sweep。正式测试从已经通过 validity gate 的前置实验中冻结少量 representative points，并在两个模型上执行 matched validation。

实验只选择能够覆盖以下机制链的代表性 workload：

```text
serving-state pressure
        ↓
cache / hierarchy behavior
        ↓
I/O or scheduling pressure
        ↓
optimization effect
        ↓
serving performance
```

同一模型内部先完成 baseline bottleneck profile，再验证对应 optimization 是否改变了目标中间机制，最后进行跨模型对照。

## 3. Serving state 的统一定义

实验统一使用 `serving state` 描述推理期间需要持续保存、复用、迁移或参与容量管理的模型状态。

Qwen3.5-9B 与 Gemma 4 12B 的 state composition 分别记录，不假定两者等价于普通 dense-attention KV cache，也不假定两种模型具有相同的 state scaling behavior。

每个模型至少记录：

- total reusable serving-state footprint；
- runtime 可观测的各类 state group；
- GPU-resident state；
- CPU-resident state when hierarchy is enabled；
- state eviction；
- state restore；
- recomputation；
- CPU-GPU transferred bytes when applicable。

跨模型比较同时保留 absolute footprint 与 normalized pressure indicator。

只有在恢复了跳过对应 prefix computation 所需的全部 state group 时，才把一次 hierarchy reuse 记为 full hierarchical hit。Partial state restore 单独标记，不与完整 hierarchy result 混合。

## 4. Representative workload 结构

Experiment 1 使用三类代表性 workload。三类 workload 分别针对 state pressure、reuse/I/O 和 locality/scheduling，不进行不必要的 full-factorial combination。

### 4.1 Scenario A: State Pressure

该场景逐步提高 reusable serving-state pressure，使 GPU cache 从容量充足区域进入明显受限区域。

该场景保持较稳定的 prefix reuse structure 与 arrival pattern，使主要变化来自 state working-set pressure。

每个模型分别设置 Low、Medium 与 High 三个相对 pressure region。Pressure region 根据该模型在当前 runtime 下的实际 reusable-state capacity 和 observed occupancy 定义，而不是机械使用完全相同的绝对 cache bytes。

该场景主要记录：

- GPU state occupancy；
- reusable-state eviction；
- recomputation；
- CPU-tier activity when enabled；
- CPU-GPU traffic；
- non-overlapped stall；
- TTFT；
- request/token throughput。

该场景用于建立两个模型各自的 baseline state-pressure surface，并确认 state pressure 是否仍然能够传导为 serving degradation。

### 4.2 Scenario B: Reuse and I/O Pressure

该场景使用具有明确 shared-prefix/context reuse 的 workload，并产生稳定的 reusable-state restore demand。

正式比较至少包含：

1. reference / baseline cache path；
2. validated hierarchical cache path；
3. validated hierarchical cache + I/O optimization path。

三种配置使用相同 logical workload、相同 GPU reusable-state budget 与相同 CPU-tier policy。

该场景主要记录：

- effective reusable hit volume；
- avoided recomputation；
- CPU-GPU transferred bytes；
- restore frequency；
- sustained transfer efficiency；
- non-overlapped I/O stall；
- TTFT；
- throughput。

本场景不重新进行完整 page-size sweep。Page Granularity and GPU-Assisted I/O 组已经验证的 granularity / backend 只在 representative configuration 上复用。

该场景用于判断 `reuse → restore traffic → stall → serving effect` 的因果链在两个模型上是否仍然成立，以及 I/O optimization 的收益是否与实际 restore pressure 对应。

### 4.3 Scenario C: Locality and Scheduling Pressure

该场景构造具有代表性 cache locality 与 request competition 的 workload，并保持 context/state pressure 在一个预先冻结的范围内。

正式比较至少包含：

1. validated baseline/reference scheduler；
2. validated optimized scheduler。

Scheduler Optimization 使用前置 scheduler experiment 已经验证过的最终 representative mechanism configuration，不在本实验重新进行完整 component ablation。

两个模型分别在相对轻载、中载与高载区域运行。Load region 根据各模型 baseline serving capacity 的相对 operating region 冻结，而不是机械使用完全相同的 requests/s。

该场景主要记录：

- reuse realization；
- delay-hit behavior when supported；
- redundant prefill / recomputation；
- queueing delay；
- I/O stall；
- scheduler idle/stall behavior；
- TTFT；
- throughput。

该场景用于判断 scheduler 所针对的 workload pathology 在两个模型上是否仍然存在，以及优化收益是否随 pathology severity 同方向变化。

## 5. 实验矩阵

主实验矩阵控制在以下范围：

| Dimension | Setting |
|---|---|
| Model | Qwen3.5-9B / Gemma 4 12B |
| Hardware | A100 40GB fixed |
| Workload family | State Pressure / Reuse and I/O / Locality and Scheduling |
| Pressure / load | representative Low / Medium / High regions |
| Optimization | only the baseline and validated mechanism configurations required by the current workload |
| Repetition | multiple independent runs after warm-up |

实验不对所有 workload family、pressure level、optimization switch 与 runtime control 做完整笛卡尔积。

每类 workload 只改变当前研究问题需要的主要变量，其余条件保持冻结。Representative points 使用 versioned identifiers 保存。

## 6. 跨模型 workload matching

两种模型使用相同的 logical workload definition，并分别通过各自 tokenizer materialize 实际 request trace。

每个 trace 保存：

- logical request identifier；
- context/prefix identifier；
- input token count；
- shared/reusable token count；
- output target；
- realized output length；
- arrival timestamp；
- reuse/revisit metadata；
- workload family；
- pressure/load region；
- trace seed / config hash。

跨模型比较检查实际 token-length distribution 和 offered work。若 tokenizer 或模型行为导致两种模型的 realized token work 明显不同，则补充 matched-token、matched-work 或 relative-pressure control，而不直接把性能差异解释为模型机制差异。

## 7. 控制变量

同一 representative point 的 paired comparison 尽可能保持以下条件一致：

- GPU hardware；
- CPU / NUMA / host-memory placement；
- experiment code revision；
- runtime family 与 validated mechanism semantics；
- numerical precision policy；
- cache/state dtype policy；
- logical request trace；
- prefix reuse structure；
- locality structure；
- output target distribution；
- cache initial-state protocol；
- generation settings；
- measurement boundary；
- measurement window；
- random seed / trace identifier。

主实验不分别为两个模型进行针对结果的后验调优。

如果某一模型必须改变关键 system semantics 才能运行，则该 point 不进入严格 cross-model paired comparison，并单独记录 capability boundary。

## 8. Pressure 与 load 的定义

Absolute state footprint 和 baseline capacity 在两个模型之间可能不同，因此 Experiment 1 使用 relative operating region 作为主要跨模型 matching 方法。

State pressure 使用实际 reusable-state occupancy、working-set size 与 eviction onset 建立 Low、Medium、High 区域。

Serving load 使用 baseline throughput/queueing behavior 建立轻载、中载和接近饱和区域。

Relative region 的 calibration 在 optimized result 产生前完成并冻结。

同一模型内部的所有 system configurations 使用相同的 frozen pressure/load points。不能为 optimized configuration 重新选择更有利的 load point。

## 9. 核心指标

### 9.1 State layer

记录：

- total serving-state footprint；
- per-state-group footprint when observable；
- GPU-resident state；
- CPU-resident state；
- eviction；
- restore；
- recomputation。

### 9.2 Cache and I/O layer

记录：

- cache / reusable-state hit volume；
- effective reused tokens or equivalent reusable work；
- CPU-GPU transferred bytes；
- restore throughput / transfer efficiency；
- non-overlapped I/O stall。

### 9.3 Scheduler layer

记录 runtime 能够可靠暴露的：

- delay-hit / unresolved reuse behavior；
- redundant prefill / redundant computation；
- queueing delay；
- scheduler stall / idle behavior；
- batch behavior when needed for attribution。

### 9.4 Serving layer

统一记录：

- request throughput；
- token throughput；
- P50 / P90 / P99 TTFT；
- request completion time；
- GPU utilization。

Serving metrics 用于确认 mechanism-level change 是否真正传导到系统性能，不单独作为跨模型机制结论的唯一依据。

## 10. System configuration rule

Experiment 1 不使用一个固定的五配置矩阵覆盖所有 workload。

State Pressure 场景主要用于建立 baseline bottleneck profile，并根据前置 hierarchy experiment 选择必要的 reference / hierarchical comparison。

Reuse and I/O 场景只比较能够隔离 hierarchy 与 I/O effect 的代表性配置。

Locality and Scheduling 场景只比较经过 semantic gate 的 reference scheduler 与 optimized scheduler configuration。

这种设计避免为了形式统一而加入与当前 mechanism 无关的 system configuration，同时保留每个 conclusion 需要的 attribution chain。

## 11. 实验执行流程

每个模型首先完成统一的模型加载、runtime initialization 与 warm-up。

随后执行 capability validation，确认目标 state group、cache/hierarchy path、I/O path 或 scheduler mechanism 在当前模型上具有预期语义。

正式 run 按冻结的 representative-point list 执行。

同一 representative point 的 paired configurations 使用相同 logical trace、arrival schedule、cache initial-state rule 与 measurement boundary。

每个 configuration 进行多次独立重复测量。不同 configuration 和 model 的执行顺序交替或随机化，减少机器长期状态变化带来的系统偏差。

每次 run 保存完整 metadata 与 validity status。

## 12. Run validity conditions

进入主 cross-model result 的 run 必须满足：

- model checkpoint 与 revision 已记录；
- runtime version/commit 已记录；
- hardware/driver/CUDA/runtime metadata 完整；
- target mechanism 通过对应 capability gate；
- target state groups 被正确识别；
- full hierarchy conclusion 对应完整 state restore；
- workload trace 与冻结的 representative point 一致；
- actual token/work summary 已记录；
- cache/state budget 与 current comparison contract 一致；
- arrival/load protocol 未因 runtime behavior 被动态修改；
- measurement window 完整；
- instrumentation 未发生静默失败；
- 未发生破坏 paired comparison 语义的 fallback、OOM 或 runtime failure。

不满足条件的 run 保留 raw result，并标记为 `partial`、`unsupported` 或 `invalid`。

## 13. 重复测量与统计

每个正式 configuration 在 warm-up 后进行多次独立重复运行。

主结果同时报告 absolute measurement、中心值、run-to-run variability 与必要 uncertainty summary。

跨模型 optimization comparison 使用相对于同一模型自身 baseline 的 normalized effect。

对于 throughput 等越高越好的指标，可使用：

```text
relative_gain = (optimized - baseline) / baseline
```

对于 latency、stall、recomputation 等越低越好的指标，统一报告 relative reduction 或使用预定义 sign convention。

Normalized effect 必须始终保留对应 absolute baseline 与 optimized measurement。

## 14. 结果组织

结果首先分别为 Qwen3.5-9B 与 Gemma 4 12B 建立 baseline bottleneck profile。

随后分别验证：

```text
observed bottleneck
        ↓
target optimization enabled
        ↓
mechanism-level observable changes
        ↓
serving metric changes
```

最终跨模型比较重点包括：

1. state-pressure onset 与 severity；
2. hierarchy 对 eviction/recomputation 与 serving performance 的影响；
3. I/O pressure 与 I/O optimization 的 normalized effect；
4. scheduling pathology 与 scheduler optimization 的 normalized effect；
5. mechanism-level change 与 TTFT/throughput change 是否保持一致解释。

## 15. 结果判断逻辑

### Stable

两个模型上均出现相同类型的 bottleneck，目标 optimization 均改变对应中间机制，并产生方向一致的 serving effect。该结果支持较强的 cross-model robustness conclusion。

### Weakened

两个模型上机制均存在，但其中一个模型的 bottleneck severity 和 optimization gain 明显更弱。该结果支持机制仍成立但重要程度下降的结论。

### Model-sensitive

两个模型的 bottleneck profile 或 optimization effect 明显不同，并且差异能够与实际 serving-state behavior 或 workload interaction 对应。该结果标记为 model-sensitive，不进一步声明 architecture-level causality。

### Boundary case

机制只在部分 pressure/load region 成立。该结果用于定义 Strata-style optimization 的现代适用边界，不视为实验失败。

### Capability-limited

某一模型无法验证完整 state restore、I/O path 或 scheduler semantics。该结果记录为 implementation/capability boundary，不解释为系统机制本身失效。

### Inconclusive

测量波动、workload mismatch 或 capability uncertainty 足以影响结论。该 comparison 标记为 `inconclusive`，不强行输出 stable 或 unstable 判断。

## 16. 最终输出

Experiment 1 至少形成一张 cross-model mechanism matrix：

| Mechanism | Qwen3.5-9B | Gemma 4 12B | Cross-model conclusion |
|---|---|---|---|
| State-capacity pressure | severity / operating region | severity / operating region | stable / weakened / model-sensitive / boundary |
| Hierarchical cache | normalized effect + mechanism evidence | normalized effect + mechanism evidence | same categories |
| I/O pressure | severity / stall evidence | severity / stall evidence | same categories |
| I/O optimization | normalized effect + transfer/stall evidence | normalized effect + transfer/stall evidence | same categories |
| Scheduling pressure | pathology / queueing evidence | pathology / queueing evidence | same categories |
| Scheduler optimization | normalized effect + scheduler evidence | normalized effect + scheduler evidence | same categories |

最终结论不比较 Qwen3.5-9B 与 Gemma 4 12B 谁具有更高 absolute throughput，而是判断 Strata 的问题机制与解决机制在两个现代模型上是否仍然形成完整、可解释的证据链。

## 17. 实验边界

本实验固定 A100 40GB，不用于形成 cross-hardware conclusion。

A100 与 L40 的结论稳定性由 Experiment 2 单独验证。

完整 `2 models × 2 GPUs` 的 serving-level validation 由 Experiment 3 完成。

本实验不重新执行前五组的全部参数扫描，也不重新进行完整 scheduler component ablation 或 page-granularity sweep。前置实验已经验证的 representative mechanism configuration 作为本实验输入。
