# Hierarchical Cache Value Evaluation

本目录用于评估现代 hybrid LLM 在 GPU HBM 受限时，将可复用 cache/state 扩展到 CPU 是否仍然具有实际系统价值。

这里关注的核心问题不是单纯定位 I/O bottleneck，而是判断 hierarchical context caching 在现代模型上是否仍然值得使用，以及它的收益边界由什么因素决定。

## Scope

本部分包含四个实验：

1. **Baseline Benefit**：在固定 workload 与代表性 cache pressure 下，对比 GPU-only 与 GPU + CPU hierarchical cache，并分别考察 cold-cache 与 warm-cache。
2. **GPU Cache Pressure Scaling**：只改变 GPU reusable-cache capacity pressure，确定 hierarchy 的收益从什么容量区域开始出现。
3. **Prefix Reuse Scaling**：在固定 cache pressure 下只改变 prefix revisit/reuse opportunity，确定什么程度的 reuse 才值得把状态保留在 CPU。
4. **Cross-Model Validation**：不重复 Experiments 2–3 的完整 sweep，只在第二个模型上复验少量预先选定的代表性配置，检查主要规律是否跨模型成立。

Experiments 1–3 默认在一个通过完整 hierarchy validation gate 的 primary model 上完成。默认候选是 Qwen3.5-9B on A100 40GB。若当前 runtime 无法验证 Qwen3.5 attention KV 与 Gated DeltaNet state 的完整 CPU restore，则不得把 partial offload 当作完整 hierarchy 结果，primary model 应切换为已验证模型。

Experiment 4 负责 A100 上的小规模 cross-model validation。项目级的 Model and Hardware Generalization 组会复用这里的 A100 结果并增加代表性的 L40 验证，不再重复完整 sweep。

## Directory structure

```text
hierarchical-cache-value/
├── README.md
├── docs/
│   ├── 00-common-conventions.md
│   ├── 01-baseline-benefit.md
│   ├── 02-gpu-cache-pressure-scaling.md
│   ├── 03-prefix-reuse-scaling.md
│   ├── 04-cross-model-validation.md
│   └── 05-current-status.md
├── code/
│   └── README.md
└── results/
    └── README.md
```

- `docs/00-common-conventions.md`：统一 validity gate、workload invariants、cache pressure、reuse 与指标口径。
- `docs/01-04*.md`：各实验独立目标、流程、结果结构与解释边界。
- `docs/05-current-status.md`：实现、capability gate 与 smoke 的已完成证据，以及正式重复测量的后续边界。
- `code/`：workload 构造、实验运行、runtime validation、指标采集和结果处理代码。
- `results/`：raw measurements、processed data、统计结果、图表与结果摘要。

## Experiment logic

```text
Experiment 1
固定代表性条件下，hierarchy 有没有基础收益？
        ↓
Experiment 2
收益在什么 GPU reusable-cache pressure 下出现？
        ↓
Experiment 3
在已有容量压力时，需要多少 prefix reuse 才值得保留 CPU state？
        ↓
Experiment 4
这些方向性结论在第二个现代模型上是否仍然成立？
```

Experiment 2 与 Experiment 3 分别隔离 capacity pressure 和 reuse opportunity。Experiment 3 不改变 hotspot concentration、reuse distance 或 request ordering，避免把 prefix reuse 与后续 cache-locality/scheduler 实验混在一起。

## Core metrics

本部分重点观察：

- GPU cache hit / eviction；
- CPU cache hit；
- recomputation；
- CPU-GPU traffic；
- non-overlapped restore stall；
- TTFT；
- throughput；
- active-request preemption。

Hit rate 不能单独作为收益证据。正式结论必须把 reusable-state eviction、CPU-tier hit、avoided recomputation、transfer/stall 和端到端性能放在同一证据链中。

## Runtime baseline

当前实现使用独立的用户目录环境：

```text
/share01/hpc/humxlab_intern/yanglihan/dl-stack/envs/sglang-hicache-cu129-torch211
```

固定版本为 SGLang `0.5.6.post3.dev8468+g4ad990ba7`（commit `4ad990ba7d75bb9f948f5f6bd8d79a66b5d3fd63`）与 PyTorch `2.11.0+cu129`。第一组正式实验已经验证 `direct` I/O backend、`page_first_direct` host layout、`write_through` policy、page size 64 与 `hicache-ratio=3`；本实验组固定复用这组配置，并启用公开 `/metrics` 分层命中证据。它们不作为 Experiments 1–3 的 sweep 变量。`kernel` I/O 留给后续 GPU-assisted I/O 实验组，避免改变层级价值实验的机制条件。

GPU-only 与 hierarchical 配对运行必须使用相同模型、GPU cache budget、page/block/state policy、scheduler、workload 与 serving load；配对差异仅是是否启用经过验证的 CPU hierarchy。环境已通过 A100 BF16、JIT、FlashInfer 和 KV-state Exp1–3 smoke，只表示运行环境可用，不替代下面的 full-hierarchy capability validation。

禁止使用旧 `envs/sglang`、partial `envs/sglang-cu129`、Docker、系统 CUDA/driver 修改或登录节点安装。详细环境与工具链要求见仓库根目录 [`docs/ENVIRONMENT_REQUIREMENTS.md`](../../docs/ENVIRONMENT_REQUIREMENTS.md)。

## Execution gate

所有实验遵循 [`docs/00-common-conventions.md`](docs/00-common-conventions.md) 和仓库根目录 [`docs/TECHNICAL_BASELINE.md`](../../docs/TECHNICAL_BASELINE.md)。

尤其需要满足：

- CPU-resident hit 必须由 runtime observable behavior 验证；
- hybrid model 必须确认所有必要 state groups 都能正确 restore；
- GPU-only 与 hierarchical 使用相同 GPU cache budget；
- cache-pressure sweep 不能通过 active-request preemption 人为制造差异；
- CPU tier capacity 不能在 Experiments 1–3 中成为未控制的第二个容量变量；
- partial hierarchy 结果与 full hierarchy 结果分开报告。

实验不预设 hierarchical cache 一定取得正收益。如果 restore traffic 或 stall 抵消 recomputation reduction，这本身就是有效结论。
