# Experiment 4: Cross-Model Hierarchical Cache Validation

## 1. 实验目标

本实验用于验证 Experiments 1–3 在 primary model 上得到的主要 hierarchical-cache 规律能否在第二个现代 hybrid 模型上复现。

Experiment 4 不重新执行完整 GPU cache-pressure sweep 或 prefix-reuse sweep。它只选择少量在前三个实验中预先定义的代表性 operating points，在第二个模型上进行 matched validation。

本实验主要回答三个问题：

1. GPU reusable-cache pressure 增加后 hierarchy 从低价值区进入有价值区的方向性规律是否跨模型成立；
2. 在固定 capacity pressure 下，prefix reuse 增加后 hierarchy 收益开始出现的规律是否跨模型成立；
3. 两个模型的收益幅度和机制链是否存在 model-dependent 差异。

本实验不根据两个模型的差异直接声称 attention architecture 是唯一因果因素。

所有配置遵循 [`00-common-conventions.md`](00-common-conventions.md)。

## 2. 实验对象与执行角色

Experiments 1–3 在一个 validated primary model 上完成完整 sweep。Experiment 4 在第二个模型上进行少量 A100 40GB matched validation。

默认角色为：

- primary sweep candidate：Qwen3.5-9B；
- secondary validation candidate：Gemma 4 12B。

该角色不是固定结论，而取决于 full-hierarchy runtime validation。

如果 Qwen3.5 无法验证 attention KV 与 Gated DeltaNet recurrent state 的完整 CPU restore，则不能把其 partial offload 结果作为 full-hierarchy primary sweep。此时可以由 Gemma 4 承担 primary sweep。只有当 Qwen3.5 后续通过完整 validation gate 时，它才能作为 second-model full-hierarchy validation；否则跨模型结论明确标记为 unsupported，而不是使用 partial hierarchy 填补结果。

本实验只使用 text-only requests。

## 3. Representative points 的选择原则

代表性点必须在查看 second model 的性能结果之前，由 Experiments 2 和 3 的 primary-model 结果与预先定义规则确定。

Experiment 4 至少包含三个 validation points。

### V0: Low-value control

V0 取自 Experiment 2 的 Low pressure region，并保持 Experiment 2 的固定 reuse 条件。

该点用于验证在 GPU 可以覆盖主要 reusable working set 时，额外 CPU tier 的端到端收益应当较小。

V0 是 negative control，不要求 second model 获得完全相同的数值，只要求实际系统状态确实处于 low-pressure region。

### V1: Capacity value-onset validation

V1 对应 Experiment 2 中 hierarchy 开始产生稳定端到端收益附近的 pressure region。

Second model 不强制使用与 primary model 相同的绝对 GPU cache GB 数值，而是通过 calibration 建立相似的系统状态：

- 已出现稳定 reusable-state eviction；
- active-request preemption 为 0；
- GPU-only recomputation 已开始增加；
- hierarchical 配置存在可验证 CPU-tier hit。

该点验证的是 **capacity pressure → hierarchy value onset** 的方向性规律。

### V2: Reuse value-onset validation

V2 使用 Experiment 3 固定的 representative GPU pressure region，并选择 primary model 上 reuse benefit 开始稳定出现附近的 revisit level。

Second model 使用相同的 workload construction rule：eligible revisit slots、request ordering、prefix length distribution 和 locality structure保持一致，只按相同定义调整 revisit fraction。

该点验证的是 **prefix reuse opportunity → hierarchy value onset** 的方向性规律。

### Optional V3: High-value stress point

如果实验成本允许，可以增加一个 high-pressure / high-reuse stress point，用于判断收益在更强 hierarchy demand 下继续增加、进入平台或受 restore stall 限制。

V3 不是完成 Experiment 4 的必要条件，也不能为了得到更漂亮的跨模型一致性而事后新增。

## 4. 跨模型匹配原则

跨模型匹配采用 **observed operating regime matching**，而不是绝对参数匹配。

两个模型的 cache/state footprint、retention behavior 和 allocator layout 不同，因此相同 GPU cache GB 数值不代表相同 capacity pressure。

每个 validation point 都需要报告 second model 的实际：

- GPU cache occupancy；
- GPU hit / eviction volume；
- CPU hit volume；
- active-request preemption；
- reusable working-set coverage 的可观测 proxy。

只有这些指标与目标 regime 一致，该配置才被认为是有效 matched point。

## 5. Workload 匹配

两个模型使用相同结构的 text-only workload。

保持：

- 相同 request count；
- 相同 prefix-group construction；
- 相同 eligible revisit slots；
- 相同 request ordering；
- 相同 revisit fraction 定义；
- 相同 output-length target；
- 相同 offered-load regime。

由于 tokenizer 不同，不要求 raw text 产生完全相同 token sequence。跨模型需要记录实际 token lengths，并尽可能保持 input/output token scale 和 reusable-prefix token proportion 可比。

如果 tokenization 导致实际 workload 偏离目标 regime，应重新生成匹配文本或调整 trace，而不是忽略差异。

## 6. 对照配置

每一个 validation point 都严格配对运行：

- **GPU-only**；
- **GPU + CPU hierarchical cache**。

两种 architecture 使用相同 second-model GPU cache budget、workload trace、offered load 和 scheduler policy。

CPU tier 保持足够容量，使 CPU eviction 不成为未控制变量。

## 7. Cache initial state

主实验统一使用 warm-cache steady-state。

每个 validation point 先执行匹配的 cache-population phase，并验证实际 residency/occupancy，然后进入正式测量。

Cold-cache 已由 Experiment 1 独立研究，Experiment 4 不重复该维度。

## 8. Validity conditions

每个 second-model validation run 必须满足：

- full-hierarchy state restore 通过数值与 residency 验证；
- GPU-only 与 hierarchical 的 GPU budget 完全一致；
- active-request preemption 为 0；
- CPU tier 不发生未控制 capacity eviction；
- validation point 的实际 pressure / reuse state 与预定义 regime 匹配；
- 配置选择发生在 second-model 性能结果分析之前；
- partial hierarchy 不进入 full-hierarchy cross-model comparison。

如果 second model 无法满足 full-hierarchy validation，则 Experiment 4 的结论为 runtime capability limitation，不用替代机制伪造跨模型结果。

## 9. 实验执行过程

首先从 Experiments 2 和 3 的 primary-model结果中冻结 V0、V1、V2 的选择规则。

随后在 second model 上进行 calibration，使每个 point 达到目标 observed regime。

每个 point 分别执行 GPU-only 与 hierarchical 配置，多次独立重复。

配对 architecture 的执行顺序交替或随机化。

所有 run 保存完整 metadata，包括 model revision、runtime commit、state-group validation status、GPU/CPU cache budget、workload identifier、actual token lengths、revisit fraction、pressure metrics 和 repetition index。

## 10. 核心测量指标

### 10.1 Cache / state behavior

记录：

- cache/state footprint by state group when available；
- GPU hit volume；
- GPU eviction volume；
- CPU hit volume；
- full / partial restore status。

### 10.2 Recomputation

记录 GPU-only 与 hierarchical 配置的 recomputation，并计算 relative recomputation reduction：

```text
relative recomputation reduction
=
(recompute_GPU-only - recompute_hierarchical)
/
recompute_GPU-only
```

### 10.3 Data movement

记录 CPU-GPU restore volume、transfer activity 和 non-overlapped restore stall。

### 10.4 TTFT

记录 median、P90 和 P99 TTFT，并计算：

```text
relative TTFT improvement
=
(TTFT_GPU-only - TTFT_hierarchical)
/
TTFT_GPU-only
```

### 10.5 Throughput

记录 steady-state throughput，并计算：

```text
throughput gain
=
throughput_hierarchical / throughput_GPU-only - 1
```

绝对值与相对值同时保留。

## 11. 第一层分析：方向一致性

首先判断三个 representative points 的方向是否一致。

| Point | 要验证的规律 |
|---|---|
| V0 | GPU reusable cache 充足时 hierarchy 收益有限 |
| V1 | capacity pressure 增加后 hierarchy 开始产生收益 |
| V2 | 在固定 pressure 下 reuse 增加后 hierarchy 开始产生收益 |

如果 second model 在满足对应 observed regime 时呈现相同方向，则说明前三个实验的主要系统规律具有一定跨模型稳定性。

不要求两个模型拥有相同百分比的收益。

## 12. 第二层分析：机制一致性

对于每个模型检查完整机制链：

```text
reusable-state pressure / revisit opportunity
                ↓
GPU miss or eviction
                ↓
validated CPU-tier hit
                ↓
avoided recomputation
                ↓
restore traffic / non-overlapped stall
                ↓
TTFT / throughput change
```

如果两个模型最终收益方向相同，但链条中的中间变量差异明显，则报告为 model-dependent mechanism strength，而不是只比较端到端加速比。

## 13. 结果判断逻辑

### 情况 A：V0、V1、V2 方向均一致

说明 hierarchical cache 的基础价值边界不是只存在于 primary model。Capacity pressure 与 reuse opportunity 对 hierarchy value 的方向性影响具有跨模型证据。

### 情况 B：方向一致但收益幅度不同

说明 hierarchy 的基本机制具有跨模型稳定性，但具体收益受 model-specific cache/state footprint、GPU residency、recomputation cost 和 restore behavior 影响。

### 情况 C：某个 validation point 不一致

首先检查 observed pressure、actual reuse、state-group coverage、restore stall 和 tokenization workload 是否真正匹配。

只有排除这些系统性差异后，才能报告为 model-dependent result。仍然不能由两模型比较直接证明 attention architecture 是唯一原因。

### 情况 D：Second model 无法完成 full hierarchy

如果 runtime 无法验证 second model 的完整 state restore，则不做 full-hierarchy cross-model performance claim。

该结果被记录为 runtime-support boundary，并保留 primary-model evidence。项目后续可以在 runtime 支持完善后重新完成 validation。

## 14. 与项目级 Generalization 的关系

Experiment 4 只处理 **A100 上的 cross-model validation**。

项目第六组 Model and Hardware Generalization 直接复用这里已经完成的 A100 representative results，并只增加完成 2 × 2 model × hardware matrix 所需的 L40 representative runs。

相同 A100 配置不得为了“第六组实验”再重复运行一遍，除非 runtime、model revision 或其他关键条件已经改变，需要重新建立可比基线。

## 15. 实验边界

本实验不进行完整 context-length sweep。

本实验不进行完整 GPU cache-budget sweep。

本实验不进行完整 prefix-reuse sweep。

本实验不改变 scheduler strategy，也不研究 cache-distance/locality effect。

本实验不比较不同硬件平台。

最终目标是验证 Experiments 1–3 的关键方向性结论是否能够从一个现代 hybrid model 推广到另一个，而不是重新复制前三组实验。
