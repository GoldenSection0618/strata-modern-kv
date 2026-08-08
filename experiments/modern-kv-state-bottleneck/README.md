# Modern KV / State Bottleneck Profiling

本目录用于重新评估 Strata 所关注的 context-cache/state loading 问题在现代 hybrid LLM 上是否仍然存在，以及它在什么 workload 条件下会成为主要系统瓶颈。

这里的 `KV/state` 是统一术语，不表示两个模型具有相同的缓存结构。Qwen3.5 同时包含 Gated DeltaNet recurrent state 与 attention KV，Gemma 4 同时包含 sliding-window/local KV 与 global-attention KV。能够分项测量时必须先报告各 state type，再报告 aggregate footprint。

## Scope

本部分包含四个实验：

1. **Context Length Scaling**：固定 reuse ratio 与低负载，研究 context 增长如何改变 cache/state footprint、计算和 CPU-GPU loading 成本。
2. **Shared-Prefix Scaling**：固定总 context，研究更多 prefix reuse 节省的 recomputation 是否被 state restore 成本抵消。
3. **Request-Rate Scaling**：固定 context 与 reuse ratio，研究 serving load 是否把单请求 state cost 放大为系统级 saturation。
4. **Cross-Model Bottleneck Comparison**：复用前三组结果并进行少量 matched validation，判断结论在 Qwen3.5 与 Gemma 4 间是否稳定。

四个实验分别控制 context、reuse、load 和 cross-model synthesis，避免在前三组中重复做多维 sweep。

## Common conventions

所有实验遵循 [docs/00-measurement-conventions.md](docs/00-measurement-conventions.md)。其中统一定义：

- `1K = 1024 tokens`；
- recompute、GPU-resident hit、CPU-resident hit 三种 residency condition；
- attention KV 与 recurrent state 的分项统计原则；
- transfer duration 与 non-overlapped I/O stall 的区别；
- TTFT、queueing、service time 与 saturation 的统一口径。

当前模型架构与 serving-runtime 假设见仓库根目录的 [`docs/TECHNICAL_BASELINE.md`](../../docs/TECHNICAL_BASELINE.md)。

## Directory structure

```text
modern-kv-state-bottleneck/
├── README.md
├── docs/
│   ├── 00-measurement-conventions.md
│   ├── 01-context-length-scaling.md
│   ├── 02-shared-prefix-scaling.md
│   ├── 03-request-rate-scaling.md
│   └── 04-cross-model-bottleneck-comparison.md
├── code/
│   └── README.md
└── results/
    └── README.md
```

- `docs/`：实验设计、统一指标定义与解释边界。
- `code/`：workload 构造、实验运行、profiling、指标采集和结果处理代码。
- `results/`：原始结果、处理后数据、统计结果与可复现图表产物。

## Execution gate

正式收集结果前，必须先验证当前 pinned runtime 对两个模型的 prefix-cache/state restore 行为。尤其是 hybrid/recurrent-state cache path，不能仅凭请求成功运行就假定 CPU-resident reuse 已正确覆盖所有 state groups。
