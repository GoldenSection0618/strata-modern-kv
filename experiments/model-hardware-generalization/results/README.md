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
- representative-point identifier；
- model identifier 与 revision；
- hardware、CPU-GPU topology、driver、CUDA/runtime；
- serving runtime version / commit；
- system configuration；
- precision 与 cache/state dtype；
- GPU reusable-state budget；
- CPU-tier budget when applicable；
- workload family；
- logical trace identifier；
- tokenizer/materialized trace identifier；
- input/output token summary；
- reuse/locality summary；
- pressure/load region；
- offered request/token/work summary；
- achieved request throughput；
- achieved token throughput；
- cache/state initial condition；
- repetition index；
- capability status；
- validity status 与 invalid reason。

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
- per-request TTFT；
- request completion time；
- GPU utilization samples；
- runtime errors and fallback events。

## Processed results

Processed results 从 raw measurements deterministic 生成，并保留 source run identifiers。

Processed data 至少支持：

- absolute baseline measurements；
- absolute optimized measurements；
- normalized gain / reduction；
- run-to-run variability；
- uncertainty summary；
- state-pressure profile；
- hierarchy effect summary；
- I/O effect summary；
- scheduler effect summary；
- model-level robustness comparison；
- later hardware-level robustness comparison；
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

Conclusion 只使用预定义类别，例如 `stable`、`weakened`、`model_sensitive`、`boundary_case`、`inconclusive`。

## Result interpretation

跨模型结果不能只比较 absolute throughput。

如果两个模型的 absolute performance 差异明显，但 bottleneck location、mechanism evidence 与 normalized optimization direction 一致，则该结果仍然支持 cross-model robustness。

如果某一模型的 bottleneck 明显减弱，同时目标 optimization 的收益也同步下降，则该结果支持机制 importance weakening。

如果两个模型差异明显，只报告 model sensitivity 与实际 serving-state behavior 的关联，不把差异进一步解释为 attention architecture 的单因素因果效应。

如果某一模型只支持 partial hierarchy 或无法验证目标 scheduler/I/O semantics，则对应 point 作为 capability boundary 报告，不进入完整 mechanism comparison。

## Figures and tables

正式 figure/table 只从 processed data 生成，不手工录入最终数值。

所有 relative metrics 必须保留 underlying absolute measurements 与 uncertainty。

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
