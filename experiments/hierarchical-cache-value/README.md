# Hierarchical Cache Value Evaluation

本目录用于评估现代 hybrid LLM 在 GPU HBM 受限时，将可复用 cache/state 扩展到 CPU 是否仍然具有实际系统价值。

这里关注的核心问题不是单纯定位某个 I/O bottleneck，而是直接判断 hierarchical context caching 这一系统设计在现代模型与当前 GPU 平台上是否仍然值得保留。

## Scope

本部分围绕四个方面展开：

1. **Baseline Benefit**：在固定 workload 与 cache pressure 下，对比 GPU-only 与 GPU + CPU hierarchical cache，并分别考察 cold-cache 与 warm-cache 场景。
2. **GPU Cache Pressure**：逐步提高 GPU cache 压力，判断 hierarchy 的收益从何时开始出现，以及收益如何随 HBM 压力变化。
3. **Prefix Reuse Scaling**：改变 prefix reuse 程度，判断 CPU restore 与 recomputation 的相对收益边界。
4. **Cross-Model Validation**：在不同 hybrid attention/state 设计的模型上复验代表性配置，判断结论是否具有跨模型稳定性。

当前已完成 Experiment 1 与 Experiment 2 的详细实验设计。后续实验继续在同一目录下补充，不重复建立新的顶层实验目录。

## Directory structure

```text
hierarchical-cache-value/
├── README.md
├── docs/
│   ├── 01-baseline-benefit.md
│   └── 02-gpu-cache-pressure-scaling.md
├── code/
│   └── README.md
└── results/
    └── README.md
```

- `docs/`：实验设计、变量定义、判定逻辑与解释边界。
- `code/`：workload 构造、实验运行、cache/state 观测、指标采集与结果处理代码。
- `results/`：raw measurements、processed data、统计结果、图表与结果摘要。

## Core metrics

本部分重点观察以下指标：

- GPU cache hit；
- CPU cache hit；
- recomputation；
- CPU-GPU traffic；
- TTFT；
- throughput。

任何 hierarchy 收益都需要同时结合 cache reuse、recomputation reduction 与 CPU-GPU transfer cost 解释，不能只根据 hit rate 或单个 latency 数字得出结论。

## Execution principle

GPU-only 与 hierarchical cache 的比较必须保持相同模型、请求集合、请求顺序、输出条件和 GPU cache budget。Warm-cache 与 cold-cache 必须具有明确且可复现的初始状态。

实验不预设 hierarchical cache 一定取得正收益。如果 CPU restore 带来的数据移动成本抵消 recomputation reduction，这本身就是本部分需要报告的结论。

当前模型架构与 serving-runtime 假设沿用仓库根目录 [`docs/TECHNICAL_BASELINE.md`](../../docs/TECHNICAL_BASELINE.md)。
