# Results

本目录用于存放 “Model and Hardware Generalization” 的实验结果。

所有结果遵循 [`../docs/00-common-conventions.md`](../docs/00-common-conventions.md)，并保持以下可追溯关系：

```text
raw measurements
    ↓
processed / normalized results
    ↓
robustness matrices / figures / tables
```

## Raw results

Raw results 保存每次 run 的原始 measurement payload 与 metadata，不被 processing scripts 覆盖。

Metadata 至少包含：

- experiment ID；
- representative-point / representative-workload identifier；
- comparison type；
- model identifier 与 revision；
- hardware platform 与 GPU form factor；
- CPU-GPU topology、CPU、NUMA placement 与 host-memory policy；
- driver、CUDA/runtime 与 PyTorch build；
- serving runtime version / commit / build source；
- system configuration；
- precision 与 cache/state dtype；
- GPU reusable-state budget；
- CPU-tier budget when applicable；
- workload family；
- logical trace identifier；
- tokenizer/materialized trace identifier；
- input/output token summary；
- reuse/locality summary；
- request-class composition when applicable；
- pressure/load region 与 calibration identifier；
- offered request/token/work summary；
- achieved request throughput；
- achieved token throughput；
- cache/state initial condition；
- repetition index；
- capability status；
- validity status 与 invalid reason；
- targeted-attribution trigger when applicable。

Raw measurement payload 尽可能保留：

- serving-state footprint by observable state group；
- GPU/CPU state residency；
- cache/state hit volume；
- eviction；
- restore；
- recomputation；
- CPU-GPU transferred bytes；
- transfer efficiency；
- non-overlapped I/O stall；
- delay-hit / redundant-work observables when supported；
- queueing time；
- scheduler idle/stall behavior；
- per-request request class when applicable；
- per-request TTFT；
- request completion time；
- completed request/token accounting；
- GPU utilization samples；
- runtime errors and fallback events。

## Processed results

Processed results 从 raw measurements deterministic 生成，并保留 source run identifiers。

Processed data 至少支持：

- absolute baseline measurements；
- absolute optimized / full measurements；
- normalized gain / reduction；
- run-to-run variability；
- uncertainty summary；
- state-pressure profile；
- hierarchy effect summary；
- I/O effect summary；
- scheduler effect summary；
- model-level robustness comparison；
- hardware/platform-level robustness comparison；
- same-workload vs matched-pressure comparison；
- end-to-end Baseline vs Full comparison；
- overall and request-class-level mixed-workload aggregation；
- targeted-attribution result when triggered；
- final model × hardware robustness matrix。

Invalid / partial / unsupported runs 不删除。主 aggregation 只包含满足对应 experiment validity requirements 的 runs。

## Experiment 1 outputs

Cross-model Mechanism Generalization 至少形成：

1. Qwen3.5-9B 与 Gemma 4 12B 各自的 baseline bottleneck profile；
2. State Pressure 场景下 state occupancy、eviction、recomputation、stall、TTFT 与 throughput 的对照；
3. Reuse and I/O 场景下 hierarchy 与 I/O optimization 的 absolute 和 normalized effect；
4. Locality and Scheduling 场景下 scheduler optimization 的 absolute 和 normalized effect；
5. 两个模型的 actual token/work matching summary；
6. full / partial hierarchy capability summary；
7. mechanism-level observable 与 serving-level performance 的 evidence chain；
8. cross-model mechanism matrix。

核心 cross-model matrix 使用以下结构：

| Mechanism | Qwen3.5-9B | Gemma 4 12B | Cross-model conclusion |
|---|---|---|---|
| State-capacity pressure | ... | ... | ... |
| Hierarchical cache | ... | ... | ... |
| I/O pressure | ... | ... | ... |
| I/O optimization | ... | ... | ... |
| Scheduling pressure | ... | ... | ... |
| Scheduler optimization | ... | ... | ... |

Mechanism conclusion 使用 [`../docs/00-common-conventions.md`](../docs/00-common-conventions.md) 中定义的类别。

## Experiment 2 outputs

Cross-hardware Conclusion Stability 至少形成：

1. 每个 representative point 的 A100 与 L40 absolute measurements；
2. 每个平台内部 optimization 相对自身 baseline 的 normalized effect；
3. `same_workload` comparison 下的 bottleneck location、pressure、mechanism observable 与 serving effect；
4. 必要时的 `matched_pressure` control，以及实际 pressure-matching summary；
5. GPU form factor、CPU-GPU topology、CPU/NUMA、host-memory policy、driver、CUDA/runtime 等 platform metadata summary；
6. capacity / saturation boundary shift；
7. unsupported / partial capability boundary；
8. cross-hardware mechanism matrix。

`same_workload` 与 `matched_pressure` 结果必须分开保存和绘图，不允许把两种 comparison semantics 混成一个平均值。

核心 matrix 使用以下结构：

| Model | Mechanism | Comparison type | A100 bottleneck | L40 bottleneck | A100 normalized effect | L40 normalized effect | Conclusion |
|---|---|---|---|---|---:|---:|---|
| Qwen3.5-9B | Hierarchical Cache | same_workload / matched_pressure | ... | ... | ... | ... | ... |
| Qwen3.5-9B | I/O Optimization | same_workload / matched_pressure | ... | ... | ... | ... | ... |
| Qwen3.5-9B | Scheduler | same_workload / matched_pressure | ... | ... | ... | ... | ... |
| Gemma 4 12B | ... | ... | ... | ... | ... | ... | ... |

如果两个 GPU 所在 host platform 不同，matrix 与 figure caption 必须明确使用 platform-level comparison 口径。

## Experiment 3 outputs

End-to-End Generalization 的 primary matrix 按以下维度组织：

```text
model × hardware × workload × load region × system configuration
```

其中 primary system configuration 只包含 Baseline 与 Full Configuration。

### Long-context reuse

至少形成：

1. 四种 model × hardware 组合的 request/token throughput；
2. P50/P90/P99 TTFT；
3. request completion time；
4. GPU utilization；
5. Full-vs-Baseline normalized effect；
6. reuse realization、recomputation、CPU-GPU traffic、I/O stall 与 queueing evidence。

### Short-context control

至少形成：

1. 四种组合的 Baseline vs Full throughput 与 latency；
2. low / medium / high load 下的 fixed-overhead / saturation-regression summary；
3. material regression / no-material-regression / inconclusive 所需 absolute measurement 与 uncertainty；
4. 无效 CPU-tier activity、scheduler overhead 或其他 regression attribution evidence when observed。

### Mixed workload

至少形成：

1. overall request/token throughput 与 P50/P90/P99 TTFT；
2. long-context request class 的 throughput、TTFT 与 completion time；
3. short-context request class 的 throughput、TTFT 与 completion time；
4. cross-class interference summary；
5. reuse、queueing、I/O stall、scheduler/batch behavior 等解释指标。

Aggregate performance 不能替代 request-class-level performance。Overall throughput improvement 如果伴随任一主要 request class 的 material tail-latency regression，必须报告为 `cross_class_tradeoff`。

### Targeted attribution

只有 primary matrix 满足预定义 trigger 时才产生 targeted-attribution result。

每个 attribution result 必须保存：

- trigger condition；
- target model × platform × workload point；
- 最小中间 configuration set；
- 被验证的 mechanism hypothesis；
- absolute / normalized measurement；
- attribution conclusion。

Targeted attribution 不与 primary matrix 混合 aggregation，也不能在看到结果后扩展成任意的 post-hoc parameter search。

### Final robustness matrix

Experiment 3 最终至少形成：

| Model | GPU / Platform | Workload | Load region | Baseline | Full Configuration | Throughput gain | TTFT change | Mechanism evidence | Conclusion |
|---|---|---|---|---|---|---:|---:|---|---|
| Qwen3.5-9B | A100 | Long | ... | ... | ... | ... | ... | ... | ... |
| Qwen3.5-9B | L40 | Long | ... | ... | ... | ... | ... | ... | ... |
| Gemma 4 12B | A100 | Long | ... | ... | ... | ... | ... | ... | ... |
| Gemma 4 12B | L40 | Long | ... | ... | ... | ... | ... | ... | ... |

Final conclusion 使用 `stable_generalization`、`model_sensitive`、`hardware_sensitive`、`boundary_case`、`throughput_latency_tradeoff`、`cross_class_tradeoff`、`unsupported` 或 `inconclusive`。

## Result interpretation

跨模型结果不能只比较 absolute throughput。

如果两个模型的 absolute performance 差异明显，但 bottleneck location、mechanism evidence 与 normalized optimization direction 一致，则该结果仍然支持 cross-model robustness。

如果某一模型的 bottleneck 明显减弱，同时目标 optimization 的收益也同步下降，则该结果支持 mechanism importance weakening。

如果两个模型差异明显，只报告 model sensitivity 与实际 serving-state behavior 的关联，不把差异进一步解释为 attention architecture 的单因素因果效应。

跨硬件结果优先区分 resource/boundary shift 与 mechanism failure。Same-workload 下收益消失但 matched-pressure 下重新出现，说明 operating boundary 移动，不等于 mechanism 被否定。

如果 A100 与 L40 host platform 不同，结果只支持 platform-level generalization，不能从观测差异反推出 GPU silicon 的单因素因果效应。

如果 Full Configuration 提高 throughput 但稳定恶化 P99 TTFT 或 completion time，则结果标记为 `throughput_latency_tradeoff`，不写成无条件整体提升。

如果某一模型、硬件或 mechanism 只支持 partial hierarchy，或无法验证目标 scheduler/I/O semantics，则对应 point 作为 capability boundary 报告，不进入完整 robustness comparison。

## Figures and tables

正式 figure/table 只从 processed data 生成，不手工录入最终数值。

所有 relative metrics 必须保留 underlying absolute measurements 与 uncertainty。

`same_workload`、`matched_pressure`、`end_to_end_primary` 与 `targeted_attribution` 必须使用可区分的 labels / identifiers。

正式结果必须能够追溯到：

```text
figure/table
    ↓
processed dataset + processing config/commit
    ↓
raw run identifiers
    ↓
run metadata + capability/validity status
```

## Storage policy

大体积 profiler dump、模型权重与可重新生成的大型中间文件不提交到 Git。

需要长期保留的外部结果记录 external path/object identifier、checksum、generating run identifier、runtime/processing version 与 retention note。
