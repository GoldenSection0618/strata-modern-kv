# Page Granularity and GPU-Assisted I/O

本目录评估现代 LLM serving 中 cache page granularity 对 prefix reuse 与 CPU→GPU I/O efficiency 的影响，并验证 GPU-assisted I/O 是否能够修复小粒度 page 带来的 transfer inefficiency，同时保持可接受的 GPU compute cost。

本组围绕以下因果链展开：

```text
page size
    ↓
effective prefix reuse
    ↓
actual transfer fragmentation / I/O efficiency
    ↓
GPU-assisted I/O compensation
    ↓
GPU computation interference
    ↓
end-to-end net benefit
```

## Runtime scope

主路径使用 **SGLang HiCache**，因为当前 HiCache 同时提供：

- `--page-size`；
- `--hicache-io-backend direct`；
- `--hicache-io-backend kernel`；
- explicit host-memory layout 与 write policy controls。

`direct` 作为 standard CUDA-copy baseline，`kernel` 作为 GPU-assisted I/O path。

所有 serving-level 结果都以 [`docs/00-common-conventions.md`](docs/00-common-conventions.md) 的 runtime capability gate 为前提。目标 hybrid model 如果不能验证完整 cache/state restore，则只能保留 mechanism-level / partial evidence，不能报告为完整 modern-hybrid serving result。

## Scope

本部分包含四个实验：

1. **Page Size vs. Cache Reuse**：在同一 attention backend 支持的 page-size range 内，只改变 page size，测量 page-boundary loss 与 effective reuse。
2. **Page Size vs. I/O Efficiency**：固定 logical transfer workload 验证 page/transfer fragmentation 对 bandwidth 的直接影响，再通过 serving-level validation 检查是否形成 non-overlapped I/O stall。
3. **GPU-Assisted I/O Compensation**：在 Experiments 1–2 确定的代表性 operating points 上固定 page size、layout、write policy 和 workload，仅比较 `direct` 与 `kernel` I/O。
4. **GPU Compute Cost and Net Benefit**：测量 kernel I/O 对 prefill/decode 的 computation interference，并通过实际 end-to-end serving measurement 判断净收益。

## Directory structure

```text
page-granularity-gpu-assisted-io/
├── README.md
├── docs/
│   ├── 00-common-conventions.md
│   ├── 01-page-size-cache-reuse.md
│   ├── 02-page-size-io-efficiency.md
│   ├── 03-gpu-assisted-io-compensation.md
│   └── 04-gpu-compute-cost-net-benefit.md
├── code/
│   └── README.md
└── results/
    └── README.md
```

- `docs/00-common-conventions.md`：统一 runtime、page/granularity terminology、backend definition、validity gate 与 metric semantics。
- `docs/01-04*.md`：四个实验的独立目标、变量、执行流程和解释边界。
- `code/`：workload、runtime validation、page sweep、I/O instrumentation、GPU interference profiling 与结果处理代码。
- `results/`：raw measurements、processed data、统计结果、图表和结果摘要。

## Experiment logic

```text
Experiment 1
小 page 是否提高 effective reuse？
        ↓
Experiment 2
这种 page granularity 是否真实产生更多小 transfer，并降低 I/O efficiency？
        ↓
Experiment 3
kernel I/O 能否在相同 page/workload 下恢复 transfer efficiency 并降低 stall？
        ↓
Experiment 4
恢复 I/O 所占用的 GPU resource 是否抵消了收益？
```

Experiment 1 与 Experiment 2 只有在使用 SGLang 这类统一 `page_size` 语义的 runtime 时才能共享同一 page-size axis。如果替换为把 prefix matching 与 physical storage 分离的 runtime，两个实验必须分别报告 reuse-matching granularity 与 physical/transfer granularity。

Experiment 2 不能根据 configured page size 推断 fragmentation。实际 transfer bytes、operation granularity 和 batching/coalescing behavior 必须被观测。

Experiment 3 不重新进行完整 page-size sweep，而是复用前两组确定的 representative points。Backend 比较中必须显式固定 `hicache_mem_layout`、`hicache_write_policy`、attention backend 和 cache state。

Experiment 4 不把 GPU utilization 上升直接解释为 interference。Compute cost 必须由 prefill/decode performance 和 profiler evidence 支持，最终 net benefit 必须来自实际 end-to-end measurement。

## Final output

本实验组最终需要确定三个层次的边界：

- **reuse-efficient region**：继续缩小 page 已很少增加有效 reuse；
- **I/O-compensation region**：kernel I/O 能够显著恢复 small-page transfer efficiency；
- **net-benefit region**：I/O stall reduction 在扣除 GPU computation interference 后仍带来正的 end-to-end benefit。

实验不预设更小 page 或 GPU-assisted I/O 一定更优。Unsupported、partial 和 negative results 都属于正式证据。
