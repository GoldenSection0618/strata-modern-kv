# Model and Hardware Generalization

本目录用于验证前五组实验得到的关键 systems conclusions 是否能够跨现代 hybrid model 与不同 GPU 平台保持稳定。

本组不重新执行前五组实验的完整参数空间，而是使用统一的 representative-point selection rule，从已经通过 validity gate 的实验中冻结少量代表性配置，并在模型与硬件维度上进行 matched validation。

## Scope

本部分包含三个分实验：

1. **Cross-model Mechanism Generalization**：固定参考硬件，比较 Qwen3.5-9B 与 Gemma 4 12B，验证 bottleneck、serving-state behavior 与关键 optimization effect 是否跨模型稳定。
2. **Cross-hardware Conclusion Stability**：固定模型与代表性配置，在 A100 40GB 与 L40 48GB 上验证 bottleneck location、optimization direction 与 relative benefit 是否稳定。
3. **End-to-End Generalization**：在代表性 workload 上完成 `2 models × 2 GPUs` 的最终交叉验证，判断前两组机制结论能否转化为稳定的 serving-level gain。

本组不把模型差异直接解释为 attention architecture 的单因素因果效应。跨模型结果只支持 robustness conclusion，以及 cache/state behavior 与系统收益之间的关联分析。

Experiments 1–3 均已完成详细方案设计。

## Directory structure

```text
model-hardware-generalization/
├── README.md
├── docs/
│   ├── 00-common-conventions.md
│   ├── 01-cross-model-mechanism-generalization.md
│   ├── 02-cross-hardware-conclusion-stability.md
│   └── 03-end-to-end-generalization.md
├── code/
│   └── README.md
└── results/
    └── README.md
```

- `docs/00-common-conventions.md`：本组统一比较口径、representative-point selection、normalization 与 validity requirements。
- `docs/01-cross-model-mechanism-generalization.md`：Experiment 1 的详细实验方案。
- `docs/02-cross-hardware-conclusion-stability.md`：Experiment 2 的 A100/L40 same-workload 与 matched-pressure 跨硬件稳定性实验方案。
- `docs/03-end-to-end-generalization.md`：Experiment 3 的 `2 models × 2 GPUs` end-to-end robustness、representative workload 与 targeted attribution 实验方案。
- `code/`：workload materialization、matched-run orchestration、profiling、validation、processing 与 plotting code。
- `results/`：raw measurements、processed results、robustness matrix、figures、tables 与结论摘要。

## Experiment logic

```text
Experiment 1
模型变化后，Strata 的问题机制与解决机制是否仍然成立？
        ↓
Experiment 2
硬件变化后，前述方向性结论是否仍然稳定？
        ↓
Experiment 3
模型与硬件同时变化时，最终 serving gain 是否仍然成立？
```

三个实验分别对应 mechanism robustness、hardware robustness 与 end-to-end robustness。不会为了形成完整笛卡尔积而机械重复前五组实验。

## Primary model × hardware matrix

| Model | A100 40GB | L40 48GB |
|---|---:|---:|
| Qwen3.5-9B | reference / matched validation | representative validation |
| Gemma 4 12B | reference / matched validation | representative validation |

Experiment 1 固定 A100 40GB。Experiment 2 才引入 L40 48GB 作为硬件变量。Experiment 3 使用最终冻结的少量 representative workloads 完成四种组合的交叉验证。

Experiment 3 的主矩阵以 Baseline 与一个预先冻结的 **common Full Configuration** 为主。Common Full Configuration 必须在四种 model × hardware 组合上具有相同的机制集合和经过验证的等价语义。某一组合如果无法支持该共同 feature set，则该组合对这一 full-system cross-product 标记为 `unsupported`，不能静默删除某个 mechanism 后仍使用同一个 Full Configuration 名称。

如果项目希望额外展示每个组合各自能运行的最佳配置，可以作为 `best_validated_configuration` 补充结果单独报告，但不能用它替代 common Full Configuration 的跨组合 robustness comparison。

Experiment 3 不机械重复所有中间 system configurations。只有在主结果出现异常、trade-off 或与前置机制预测不一致时，才执行 targeted attribution runs。

## Execution discipline

所有实验遵循 [`docs/00-common-conventions.md`](docs/00-common-conventions.md) 和仓库根目录 [`docs/TECHNICAL_BASELINE.md`](../../docs/TECHNICAL_BASELINE.md)。

Representative points 与 representative workloads 必须依据前置实验的预定义 selection rule 冻结，不能在看到跨模型或跨硬件结果后反向选择更有利的点。

跨模型比较优先使用相对自身 baseline 的 normalized effect 与 mechanism-level observable，而不是直接比较 absolute throughput。

跨硬件比较同时保留 same-workload deployment behavior 与必要的 matched-pressure control。前者用于观察相同实际 workload 在不同平台上的 bottleneck shift，后者用于判断在相近 operating pressure 下 mechanism 本身是否保持稳定。

Experiment 3 的 primary load grid 按模型分别从 A100 baseline calibration 冻结。对同一个模型，A100 与 L40 使用相同 arrival/load schedule，从而保留 hardware same-workload comparison。不同模型之间不要求相同 requests/s，跨模型主要比较 normalized effect 与对应 operating region。L40 若因容量或 serving capacity 差异落入明显不同的 pressure region，只补充少量 matched-pressure control，并与 primary result 分开报告。

End-to-End Generalization 同时报告 absolute serving performance 与 normalized Full-vs-Baseline effect。Mixed workload 必须保留 request-class-level performance，避免 aggregate metrics 掩盖 cross-class interference。

如果某一模型或硬件组合无法通过 runtime/state capability gate，则对应结果标记为 `partial`、`unsupported` 或 `invalid`，不通过改变语义不同的配置强行补齐矩阵。
