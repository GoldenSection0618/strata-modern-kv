# Page Granularity and GPU-Assisted I/O

本目录用于评估现代 LLM serving 中 page granularity 对 cache reuse 与 CPU→GPU I/O efficiency 的影响，并验证 GPU-assisted I/O 是否仍能缓解小粒度 cache 带来的 fragmented transfer 问题。

本部分围绕一条完整因果链展开：

```text
page size
    ↓
cache reuse efficiency
    ↓
I/O fragmentation and bandwidth efficiency
    ↓
GPU-assisted I/O compensation
    ↓
GPU computation interference and end-to-end benefit
```

## Scope

这一部分计划包含四个实验：

1. **Page Size vs. Cache Reuse**：只改变 page size，验证小 page 是否提高有效 cache reuse，并确定收益开始趋于饱和的粒度区间。
2. **Page Size vs. I/O Efficiency**：在相同 page-size sweep 下测量 actual transfer fragmentation、sustained host→GPU bandwidth、bandwidth utilization 与 serving-level I/O stall。
3. **GPU-Assisted I/O Compensation**：在 Experiments 1–2 已确定的代表性 page-size operating points 上比较 baseline I/O 与 GPU-assisted I/O，验证其能否恢复 transfer efficiency、降低 non-overlapped I/O stall，并转化为实际 serving benefit。
4. **GPU Compute Cost**：评估 GPU-assisted I/O 对 prefill、decode 和端到端 serving 的计算干扰与净收益。

当前已完成 Experiments 1–3 的实验设计文档。Experiment 4 后续按同一目录继续补充。

## Directory structure

```text
page-granularity-gpu-assisted-io/
├── README.md
├── docs/
│   ├── 01-page-size-cache-reuse.md
│   ├── 02-page-size-io-efficiency.md
│   └── 03-gpu-assisted-io-compensation.md
├── code/
│   └── README.md
└── results/
    └── README.md
```

- `docs/`：实验设计、变量定义、执行记录与分析边界。
- `code/`：workload 构造、page-size sweep、指标采集、实验运行与结果处理代码。
- `results/`：raw measurements、processed data、统计结果、图表与结果摘要。

## Experiment logic

```text
Experiment 1
小 page 是否真的提高有效 cache reuse？
        ↓
Experiment 2
这种粒度是否造成 actual transfer fragmentation、带宽下降与 serving I/O stall？
        ↓
Experiment 3
GPU-assisted I/O 能否恢复传输效率，并把恢复转化为 serving benefit？
        ↓
Experiment 4
恢复 I/O 效率所付出的 GPU computation cost 是否值得？
```

Experiment 2 分为 Controlled I/O experiment 与 Serving-level validation。前者固定逻辑总传输数据量，只改变 page granularity，以隔离 fragmentation 对 bandwidth efficiency 的直接影响；后者从 Experiment 1 选择代表性 operating points，验证该机制是否真正进入 serving critical path。

Experiment 3 不重新进行完整 page-size sweep，而是复用 Experiments 1–2 已确定的 large-page baseline、trade-off page 与 small-page fragmented region。它同样分为 Controlled I/O compensation 与 Serving-level validation，先验证 backend 对 transfer efficiency 的直接补偿，再验证 bandwidth recovery 是否能够降低 non-overlapped I/O stall 并改善 TTFT / throughput。

Experiments 1–3 使用同一 page-size axis 和可对应的 representative operating points。联合分析需要同时观察 effective reuse、actual transfer granularity、bandwidth utilization、non-overlapped I/O stall 与 serving performance，从而判断 small page + GPU-assisted I/O 是否能够形成同时兼顾 reuse 与 I/O efficiency 的有效 operating region。

本部分不预设更小 page 或 GPU-assisted I/O 一定更优。正式结论需要同时考虑 reuse benefit、I/O efficiency、GPU interference 与 end-to-end serving performance。
