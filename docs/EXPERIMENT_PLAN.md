# Experiment Plan

## 1. Evaluation objective

本项目重新评估 Strata 的主要 systems claims 在现代 hybrid LLM 与本项目可用 GPU 平台上是否仍然成立。

实验不机械复现 Fig.1–15，而是保留原论文的关键 causal questions，并重新设计 workload、model、runtime 与 hardware validation。

仓库统一使用 `KV/state` 作为 umbrella term。Attention KV、local/sliding-window KV、recurrent/linear-attention state 在 runtime 可观测时必须分项报告。

当前模型、runtime、硬件和集群事实见 [`TECHNICAL_BASELINE.md`](TECHNICAL_BASELINE.md)。

## 2. Experiment groups

### 2.1 Modern KV / state bottleneck profiling

**Question:** Strata 研究的 cache/state loading bottleneck 在现代 hybrid model 上还剩多少？

包含四个实验：

1. context-length scaling；
2. shared-prefix scaling；
3. request-rate scaling；
4. cross-model synthesis / matched validation。

主要观察 cache/state footprint、prefill/service computation、CPU-GPU transfer、non-overlapped I/O stall、queueing、TTFT 与 throughput。

详细设计位于 `experiments/modern-kv-state-bottleneck/`。

---

### 2.2 Hierarchical cache value

**Question:** 当 reusable GPU cache capacity 受限时，把 reusable cache/state 扩展到 CPU 是否仍然值得？

包含四个实验：

1. GPU-only vs GPU + CPU hierarchical baseline；
2. GPU cache-pressure scaling；
3. prefix-reuse scaling；
4. second-model matched validation。

主要证据链为：GPU eviction → validated CPU-tier hit → avoided recomputation → restore traffic/stall → TTFT/throughput。

Full hybrid-state restore 无法验证时，结果必须标记为 `partial` 或 `unsupported`。

详细设计位于 `experiments/hierarchical-cache-value/`。

---

### 2.3 Page granularity and GPU-assisted I/O

**Question:** Fine-grained cache 是否仍然带来 I/O fragmentation，GPU-assisted I/O 能否恢复 transfer efficiency，并且其 GPU compute cost 是否值得？

包含四个实验：

1. page size vs cache reuse；
2. page size vs I/O efficiency；
3. standard-copy vs GPU-assisted I/O；
4. GPU interference and net serving benefit。

Configured page size、prefix-match granularity、physical block size 与 actual transfer size 不得混为同一变量。

SGLang HiCache 是主要 mechanism candidate，但必须先通过当前集群的 non-Docker build/runtime gate。

详细设计位于 `experiments/page-granularity-gpu-assisted-io/`。

---

### 2.4 Cache locality and scheduler behavior

**Question:** Strata 的 cache-aware scheduler 在现代 workload 中还在哪些条件下有价值，各阶段分别解决什么问题？

这一组不把 cache distance 简化为“locality 越差，scheduler pathology 越严重”。Strata 原论文中，minimum cache distance 代表相同 context 请求连续出现，locality 最高，此时 delay hit 更明显；maximum cache distance 降低 delay-hit likelihood，但更容易产生 CPU-tier loading pressure。现代实验需要重新验证这两个方向是否仍然成立。

包含四个实验：

1. **Locality × Arrival Rate Baseline Profiling**。使用 baseline scheduler，在 Min distance / Shuffle / Max distance × Low / Medium / High / Overload 上建立 delay-hit 与 host-loading 两类 mechanism-specific surface。
2. **Scheduler Component Ablation**。在 Experiment 1 冻结的 W0–W3 representative workloads 上执行 progressive sequence：Baseline → +Delay-hit mitigation → +Balanced batching → +Stall hiding = Full scheduler，并在语义允许时做少量 targeted attribution。
3. **Same-Context Concurrency Stress**。以 cold-miss resolve 为主场景，固定长期平均 offered load，只改变 unresolved context resolve window 内的 same-context fan-in。GPU-ready 是 concurrency control，CPU-restore 仅作为 full hierarchy 可验证时的扩展。
4. **Scheduler Operating Region**。复用前三组结果，只补充少量 boundary points，形成 mechanism-specific operating map 与最终 scheduler decision matrix。

核心指标包括：

- cache resolve time / same-context fan-in；
- delay hit / redundant work / reuse realization；
- host restore / non-overlapped I/O stall；
- batch load / compute ratio / bundle hit；
- GPU idle / filled-bubble time；
- queueing；
- P50/P90/P99 TTFT；
- TPOT；
- throughput。

Scheduler mechanism 必须通过 semantic capability gate。当前 upstream runtime 中名字相近的 scheduler option 不能自动视为 Strata 三阶段机制的等价实现。

详细设计位于 `experiments/cache-locality-scheduler-behavior/`。

---

### 2.5 End-to-end serving

**Question:** 前面验证过的机制组合起来是否产生实际 serving gain，并且不损害普通短请求？

本组包含三个分实验。Load scaling 是三组实验的共同控制维度，不单独拆成第四个实验。

#### Experiment 1. Long-context reuse serving

验证项目最主要的 cache/state reuse scenario，观察机制组合后是否真正改善 throughput、TTFT 与 request completion time。

#### Experiment 2. Short-context serving regression

在缺乏显著 long-context reuse 的普通短请求上检查 fixed overhead、serving-capacity regression 与 tail-latency regression。

#### Experiment 3. Mixed workload serving

同时包含：

- long shared contexts；
- ordinary short requests；
- different output lengths；
- different cache-distance / locality patterns。

Mixed workload 同时报告 overall 与 request-class-level performance，避免 aggregate throughput 掩盖 long-context / short-context 之间的 cross-class interference。

系统统一比较以下五种配置：

1. **Baseline**；
2. **Hierarchical Cache**；
3. **Hierarchical Cache + I/O Optimization**；
4. **Hierarchical Cache + Scheduler Optimization**；
5. **Full Configuration**。

配置 3 与配置 4 是 parallel attribution branches，不是严格的逐层累加关系。配置 3 使用 baseline/reference scheduler，配置 4 使用 baseline/reference I/O path。Full Configuration 才同时启用经过验证的 hierarchy、I/O 与 scheduler mechanisms。

核心指标：

- request / token throughput；
- P50/P90/P99 TTFT；
- request completion time；
- GPU utilization。

TPOT 或等价 decode-latency metric 在需要解释 output-length / decode interference 时作为辅助指标记录，不作为本组统一核心指标。

只有在前面对应机制已经通过 capability/validity gate 时，才把该机制纳入 Full Configuration。

详细设计位于 `experiments/end-to-end-serving/`。

---

### 2.6 Model and hardware generalization

**Question:** 前五组得到的方向性结论能否跨模型与 GPU 平台保持？

本组包含三个分实验，不把前五组实验机械重复四遍。

#### Experiment 1. Cross-model Mechanism Generalization

固定 A100 40GB，对 Qwen3.5-9B 与 Gemma 4 12B 执行统一的 representative workloads。

该实验同时建立两个模型各自的 baseline bottleneck profile，并验证 hierarchical cache、I/O optimization 与 scheduler optimization 的 mechanism-level effect 是否跨模型保持。

跨模型比较主要使用 relative state/load region、相对自身 baseline 的 normalized effect 与 mechanism observable。Absolute throughput 作为必要原始结果保留，但不作为 cross-model robustness 的唯一判断依据。

模型差异只支持 cross-model robustness 与 serving-state behavior correlation，不支持 attention architecture 的单因素因果归因。

#### Experiment 2. Cross-hardware Conclusion Stability

固定模型与冻结的 representative points，在 A100 40GB 与 L40 48GB 上验证 bottleneck location、optimization direction 和 relative benefit 是否稳定。

硬件比较同时保留两种语义。Same-workload comparison 使用相同 logical workload 观察真实 deployment-level bottleneck shift。只有当两个平台落入明显不同的 capacity / saturation region 时，才补充少量 matched-pressure controls 判断 mechanism 本身是否仍然成立。

硬件比较不要求相同 absolute throughput。正式分析记录实际 GPU form factor、CPU-GPU topology、CPU/NUMA placement、host-memory policy、driver、CUDA/runtime 与 usable GPU memory budget。若两个 GPU 所在 host platform 不同，结论使用 platform-level comparison，不把全部差异归因于 GPU silicon。

#### Experiment 3. End-to-End Generalization

在最终冻结的少量 representative workloads 上完成 `2 models × 2 GPUs` 的交叉验证。

主实验覆盖 Long-context reuse、Short-context control 与 Mixed workload，并在 Low / Medium / High operating regions 上以 Baseline 与 Full Configuration 作为主要 comparison pair。Mixed workload 同时报告 overall 与 request-class-level performance。

主矩阵不重新执行完整 mechanism ablation。只有当 Full Configuration 出现异常收益、regression、throughput-latency trade-off、cross-class interference，或与 Experiments 1–2 的机制预测不一致时，才执行少量 targeted attribution runs。

代表性矩阵为：

| Model | A100 40GB | L40 48GB |
|---|---:|---:|
| Qwen3.5-9B | reference / matched validation | representative validation |
| Gemma 4 12B | reference / matched validation | representative validation |

Representative points 从前五组已经通过 validity gate 的结果中冻结，至少覆盖：

- neutral/control point；
- clear cache/hierarchy pressure condition；
- clear I/O pressure condition；
- clear scheduler pressure condition；
- representative operating boundary；
- representative end-to-end workload when applicable。

Selection rule 与 representative-point identifiers 必须在 generalization optimized results 生成前版本化冻结。不能在看到跨模型或跨硬件结果后反向替换验证点。

A100 上已经存在且 experiment contract 完全匹配的结果可以复用。若 tokenizer、runtime、cache/state budget、load definition、measurement boundary 或其他关键 comparison semantics 不同，则需要重新执行 matched run，不能仅根据模型/硬件名称复用。

详细设计位于 `experiments/model-hardware-generalization/`。Shared comparison conventions 与 Experiments 1–3 的详细设计均已完成。Measured generalization runs 仍需等待前置实验通过 validity gate，并在正式执行前冻结 representative points、representative workloads 与对应 execution contracts。

## 3. Experimental dependency chain

六组实验形成以下逻辑：

```text
现代模型上 bottleneck 还存在吗？
        ↓
hierarchical cache 本身值得吗？
        ↓
I/O inefficiency 从哪里来，修复代价是多少？
        ↓
哪些 workload 需要 cache-aware scheduling？
        ↓
机制组合后是否产生 end-to-end gain？
        ↓
这些方向性结论能否跨模型与硬件保持？
```

如果某个前置 premise 在现代模型/runtime 上不成立，后续结果必须解释为 conditional engineering gain，而不是继续声称原始 bottleneck 广泛存在。

## 4. Reproducibility principles

每个正式实验至少记录：

- exact model identifier / revision；
- hardware、CPU-GPU/NUMA topology；
- driver、CUDA/runtime、serving-engine version/commit；
- precision / cache dtype；
- workload identifier、token-length convention 与 exact trace configuration；
- cache residency / state policy；
- cache granularity controls；
- I/O backend、host layout、write policy when applicable；
- scheduler mechanism capability status；
- random seed；
- raw measurement source；
- processing / plotting version。

运行时默认值只要可能影响结论，就必须解析后显式写入 metadata。

Raw measurement、processed data 与 figures/tables 保持可追溯关系。

Failed、negative、partial 和 unsupported results 不静默删除。

## 5. Interpretation discipline

- 不把 configured feature 当作已经验证的 capability。
- 不把 cache hit 当作 end-to-end benefit 的充分证据。
- 不把 transfer duration 与 computation time 直接相加，异步 overlap 必须单独处理。
- 不把两个模型的差异解释为单一 architecture causal effect。
- 不复制原论文中的 hardware/model-dependent threshold 作为现代平台默认参数。
- 不在看到 optimized result 后反向选择更有利的 workload 或 boundary point。
- 不把相同 request rate 下 workload composition 或 output-length distribution 的变化解释为单一变量因果效应，除非同时控制 offered work / token volume 或提供 matched-load control。

## 6. Scope discipline

项目不以完整复现 Strata Fig.1–15 为目标。

历史 figure 只有在建立 baseline、验证 mechanism、解释 causal chain 或提供必要 reference 时才复现或重构。

同时不为了压缩实验而删除关键的 mechanism attribution、regression check、boundary validation 或 generalization evidence。
