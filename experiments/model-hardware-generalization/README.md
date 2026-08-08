# Model and Hardware Generalization

本目录用于验证前五组实验得到的关键 systems conclusions 是否能够跨现代 hybrid model 与不同 GPU 平台保持稳定。

本组不重新执行前五组实验的完整参数空间，而是使用统一的 representative-point selection rule，从已经通过 validity gate 的实验中冻结少量代表性配置，并在模型与硬件维度上进行 matched validation。

## Scope

本部分包含三个分实验：

1. **Cross-model Mechanism Generalization**：固定参考硬件，比较 Qwen3.5-9B 与 Gemma 4 12B，验证 bottleneck、serving-state behavior 与关键 optimization effect 是否跨模型稳定。
2. **Cross-hardware Conclusion Stability**：固定模型与代表性配置，在 A100 40GB 与 L40 48GB 上验证 bottleneck location、optimization direction 与 relative benefit 是否稳定。
3. **End-to-End Generalization**：在代表性 workload 上完成 `2 models × 2 GPUs` 的最终交叉验证，判断前两组机制结论能否转化为稳定的 serving-level gain。

本组不把模型差异直接解释为 attention architecture 的单因素因果效应。跨模型结果只支持 robustness conclusion，以及 cache/state behavior 与系统收益之间的关联分析。

目前已完成 Experiment 1 的详细方案设计。Experiments 2–3 在对应 representative points 与 execution contract 冻结后补充。

## Directory structure

```text
model-hardware-generalization/
├── README.md
├── docs/
│   ├── 00-common-conventions.md
│   └── 01-cross-model-mechanism-generalization.md
├── code/
│   └── README.md
└── results/
    └── README.md
```

- `docs/00-common-conventions.md`：本组统一比较口径、representative-point selection、normalization 与 validity requirements。
- `docs/01-cross-model-mechanism-generalization.md`：Experiment 1 的详细实验方案。
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

Experiment 1 固定 A100 40GB。Experiment 2 才引入 L40 48GB 作为硬件变量。Experiment 3 使用最终冻结的少量代表性 workload 完成四种组合的交叉验证。

## Execution discipline

所有实验遵循 [`docs/00-common-conventions.md`](docs/00-common-conventions.md) 和仓库根目录 [`docs/TECHNICAL_BASELINE.md`](../../docs/TECHNICAL_BASELINE.md)。

Representative points 必须依据前置实验的预定义 selection rule 冻结，不能在看到跨模型或跨硬件结果后反向选择更有利的点。

跨模型比较优先使用相对自身 baseline 的 normalized effect 与 mechanism-level observable，而不是直接比较 absolute throughput。

如果某一模型或硬件组合无法通过 runtime/state capability gate，则对应结果标记为 `partial`、`unsupported` 或 `invalid`，不通过改变语义不同的配置强行补齐矩阵。
