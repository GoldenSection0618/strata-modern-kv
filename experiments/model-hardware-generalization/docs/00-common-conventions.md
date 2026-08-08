# Common Conventions for Model and Hardware Generalization

本文件定义 Model and Hardware Generalization 组的统一比较口径。具体实验文档只能在明确说明原因时覆盖这些约束。

## 1. Generalization objective

本组验证前五组实验得到的 systems conclusions 是否跨模型与硬件保持方向性稳定。

本组不追求不同模型或 GPU 获得相同 absolute throughput，也不把两个模型之间的差异解释为 attention architecture 的单因素因果效应。

主要比较对象是：

- bottleneck 是否仍然存在；
- bottleneck location / severity 是否变化；
- optimization direction 是否一致；
- optimization relative benefit 是否稳定；
- mechanism-level observable 与 end-to-end effect 是否保持一致的解释链。

## 2. Representative-point selection

本组不重新执行前五组完整 parameter sweep。

Representative points 从已经通过 validity gate 的前置实验中冻结，至少覆盖：

- neutral / control condition；
- clear cache or hierarchy pressure condition；
- clear I/O pressure condition；
- clear scheduling pressure condition；
- representative operating boundary；
- representative end-to-end workload when applicable。

Selection rule、point identifiers 与 representative workload identifiers 必须在对应 generalization optimized results 生成前写入 versioned configuration。

不能根据 generalization run 的结果后验删除负结果或替换 representative point。

## 3. Serving-state terminology

跨模型比较统一使用 `serving state` 作为 umbrella term。

Qwen3.5-9B 与 Gemma 4 12B 的 state composition 不假定一致。能够观测时分别报告 attention KV、local/sliding-window KV、recurrent/linear-attention state，以及 runtime 暴露的其他必要 reusable state group。

`Full hierarchical hit` 只有在恢复了跳过对应 prefix computation 所需的全部 state group 时成立。只恢复部分 state 的情况标记为 partial hierarchy。

当 runtime 为 attention KV、Mamba/recurrent state、SWA 等不同 state group 分别分配 GPU 或 CPU pool 时，实验必须记录 configured budget 与 resolved allocation。一个 aggregate `GPU reusable-state budget` 或 `CPU-tier budget` 不能替代 state-group-level allocation metadata。

Configured `hicache-size`、`hicache-ratio` 或其他总量参数只表示配置意图。跨模型比较使用 runtime 实际解析后的 allocation、occupancy 与 state-group behavior 判断 pressure，不从单个配置值推断真实容量。

## 4. Model comparison rules

Experiment 1 固定 A100 40GB，避免模型变量与硬件变量同时变化。

两种模型使用：

- 相同 experiment software revision；
- 相同 runtime family / mechanism semantics，或经过明确验证的语义等价 path；
- 相同 numerical precision policy；
- 相同 workload-generation rule；
- 相同 measurement rule；
- 相同 logical reuse / locality / arrival structure。

Token-level workload 在模型 tokenizer 下分别 materialize，并记录实际 input/output token counts。跨模型比较使用实际 realized workload metadata，而不是只使用原始文本字符长度。

模型容量、state footprint 和 baseline saturation capacity 不同，因此 pressure/load 档位优先按相对 operating region 定义，而不是机械使用完全相同的绝对 cache bytes 或 requests/s。

Generic model-family support 不满足实验 capability gate。Qwen3.5-9B 与 `google/gemma-4-12B-it` 都必须在 exact pinned runtime 上验证 target checkpoint execution 和目标 state path。

## 5. Hardware comparison rules

Experiment 2 才把 A100 40GB 与 L40 48GB 作为主要变量。

硬件比较必须记录实际：

- GPU form factor；
- CPU-GPU interconnect / PCIe topology；
- CPU model；
- NUMA placement；
- host-memory policy；
- driver 与 CUDA/runtime；
- GPU usable memory budget。

GPU 型号与 nominal memory size 不能替代这些 metadata。

硬件维度主要比较 bottleneck location、optimization direction 与 normalized effect。Absolute throughput 差异本身不构成 generalization failure。

Hardware comparison 同时保留：

- `same_workload`：相同 logical workload 下的 deployment behavior；
- `matched_pressure`：只在必要时补充的相对 pressure / saturation control。

两种 comparison type 必须使用不同 identifiers，并在结果中分开解释。

如果 A100 与 L40 位于不同 host platform，则结论使用 platform-level comparison，不把全部差异归因于 GPU silicon。

## 6. End-to-End comparison rules

Experiment 3 使用 `2 models × 2 GPUs` 的完整组合，但不重新执行前置实验的完整 mechanism sweep。

Primary matrix 使用：

- Long-context reuse；
- Short-context control；
- Mixed workload；
- Low / Medium / High operating regions；
- Baseline / Full Configuration。

Full Configuration 只包含已经通过前置 capability / validity gate 的 mechanisms。

中间 configuration 只用于预定义触发条件下的 targeted attribution，不扩展为所有 workload 上的完整五配置矩阵。

Mixed workload 必须同时保存 overall 与 request-class-level performance。Aggregate gain 不能掩盖 long-context 或 short-context class 的 material tail-latency regression。

Short-context control 的 material-regression / equivalence rule 优先复用来源 End-to-End Serving 实验中已经冻结且语义完全匹配的判定规则。若 workload、runtime 或 measurement contract 已变化，则必须在查看 generalization optimized results 前重新版本化冻结 decision margin，不能事后决定“多大差异算 regression”。

## 7. Normalization

每个模型和硬件组合都保留 absolute measurements。

跨组合比较主要使用相对于同一组合自身 baseline 的 normalized delta，例如：

```text
relative_gain = (optimized - baseline) / baseline
```

对于 latency、stall、recomputation 等越低越好的指标，报告 reduction 或使用统一的 sign convention，并在 processed metadata 中明确记录。

任何 normalized result 必须可以回溯到对应 absolute baseline 与 optimized measurements。

## 8. Workload matching

Paired comparison 必须冻结：

- logical request trace；
- reuse / revisit structure；
- locality class；
- input/output target distribution；
- offered-load definition；
- cache initial-state protocol；
- measurement boundary；
- random seed / trace identifier。

如果模型 tokenizer 或模型能力导致实际 token counts 不同，必须记录 realized token distribution，并根据实验目的使用 matched-token、matched-work 或 relative-pressure control。

不能把不同实际 work volume 下的性能差异直接解释为模型或硬件机制差异。

## 9. Runtime and capability gate

所有 run 遵循仓库根目录 [`../../../docs/TECHNICAL_BASELINE.md`](../../../docs/TECHNICAL_BASELINE.md) 的 mandatory runtime validation gates。

跨模型 comparison 只有在 target checkpoint、目标 state group、cache/hierarchy path、I/O path 和 scheduler mechanism 的语义已经验证时才进入对应 mechanism conclusion。

如果 capability 只在某一模型或某一硬件组合上成立，则结果明确标记 capability boundary，不能把缺失实现解释为机制失效。

Full Configuration 只有在其组成 mechanisms 均通过当前 model × platform combination 的 capability gate 时才进入 Experiment 3 主结果。

对于 Qwen3.5，full hierarchy 必须同时验证 attention KV 与 recurrent/Gated-DeltaNet state。若 runtime 只恢复其中一类 state，则该 hierarchy capability 为 `partial`。

## 10. Repetition and uncertainty

正式 configuration 在统一 warm-up 之后进行多次独立重复测量。

Paired runs 尽量使用相同 trace identifiers，并交替或随机化执行顺序，降低机器长期状态变化造成的系统偏差。

主结果报告中心值、run-to-run variability 与必要 uncertainty summary。单次运行结果不用于形成 cross-model 或 cross-hardware robustness conclusion。

## 11. Validity status

每个 run 至少使用以下状态之一：

- `valid`；
- `partial`；
- `unsupported`；
- `invalid`。

Failed、negative、partial 与 unsupported results 保留 raw record，不静默删除。

`unsupported` 表示当前 capability 无法建立所需比较，不等于 mechanism 的负结果。

## 12. Interpretation categories

Mechanism-level robustness 使用：

- `stable`：关键 bottleneck 与 optimization direction 在比较对象间保持一致；
- `weakened`：机制仍存在，但 severity 或收益明显减弱；
- `model_sensitive`：结果随模型明显变化；
- `hardware_sensitive`：结果随硬件明显变化；
- `boundary_case`：只有部分 operating region / combination 成立，或 boundary 明显移动；
- `inconclusive`：测量精度、capability 或匹配条件不足以支持判断。

Experiment 3 的 end-to-end summary 使用：

- `stable_generalization`；
- `model_sensitive`；
- `hardware_sensitive`；
- `boundary_shifted`；
- `throughput_latency_tradeoff`；
- `cross_class_tradeoff`；
- `unsupported`；
- `inconclusive`。

这些类别必须由 absolute measurements、normalized effects 与 mechanism observables 共同支持。Trade-off category 不能被简单归入 positive system gain。
