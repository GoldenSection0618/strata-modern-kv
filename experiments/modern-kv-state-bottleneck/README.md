# Modern KV / State Bottleneck Profiling

本目录用于复现与扩展 Strata 中与现代模型 KV cache / state bottleneck 相关的实验，重点回答：现代 hybrid attention / state-space architecture 下，Strata 所研究的状态管理与 CPU-GPU I/O 瓶颈是否仍然存在，以及在什么 workload 条件下仍然重要。

## Scope

这一部分主要包含以下实验：

1. Context length scaling
2. Shared-prefix length scaling
3. Concurrency / request-rate scaling
4. Qwen3.5 与 Gemma 4 的跨模型对照

## Directory structure

```text
modern-kv-state-bottleneck/
├── README.md
├── docs/
│   └── 01-context-length-scaling.md
├── code/
└── results/
```

- `docs/`：实验设计与实验记录。
- `code/`：实验脚本、profiling 工具与辅助代码。
- `results/`：原始结果、汇总数据和绘图结果。
