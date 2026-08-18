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

两条执行路径（不可互换的 evidence source）：

- **vLLM（legacy / reference）**：`code/runners/vllm_runner.py` + `code/run_exp{1,2,3}.py`；已有 recompute 结果。
- **SGLang / HiCache（Explicit path）**：`code/sglang_hicache/`（见 [docs/06-sglang-execution-path.md](docs/06-sglang-execution-path.md)），通过公开 HTTP/Prometheus 边界驱动 SGLang server。本地实验包名为 `sglang_hicache`（不是 `sglang`），避免遮蔽上游 SGLang 包。

## Execution status (verified 2026-08-18)

### vLLM path (legacy)

Runtime: vLLM 0.26.0, Qwen/Qwen3.5-9B, A100 40GB (`i56m512A100`).

- **Validation gate**: recompute 模式通过（cache-hit 检查按设计跳过）；gpu_hit/cpu_hit 的 cache-hit 检查因 stats 采集问题必败（见下）。
- **Recompute baseline**: exp1 全部 4 点（4K/8K/16K/32K）、exp2 全部 5 点（0%/25%/50%/75%/87.5%）已完成，各 10 reps，summary.json 含 median/P90/min/max TTFT。
- **gpu_hit / cpu_hit**: 无有效数据。`VLLMStatsCollector` 在 vLLM 0.26 V1 引擎下找不到 `KVCacheManager`（内部对象在 EngineCore 子进程），prefix 统计恒为 0 → 验证门 `queries=0, hits=0` 必败 → 测量中止（退出码仍为 0）。
- **Environment requirements**（A100 / vLLM 0.26 必须）：
  - `VLLM_USE_FLASHINFER_SAMPLER=0`（flashinfer sampler JIT 编译失败）
  - `VLLM_WORKER_MULTIPROC_METHOD=spawn`（否则 fork 竞态崩溃）
  - Qwen 模型 `max_num_seqs=16`（Mamba cache block 预算限制）

### SGLang path (current primary path)

- canonical 环境 `~/yanglihan/dl-stack/envs/sglang-hicache-cu129-torch211` 已安装并在 A100 上验证；最终有效 HiCache 参数固定为 `HICACHE_RATIO=3`、`HICACHE_IO_BACKEND=direct`、`HICACHE_MEM_LAYOUT=page_first_direct`。
- Qwen Exp3 正式 `cpu_hit` primary（job `1293163`）及其严格门禁（`1293164`）已通过；同一 frozen-rate sweep 的 `recompute` / `gpu_hit` controls（`1293758` / `1293759`）及门禁（`1293760` / `1293761`）也已通过。
- Qwen Exp1（`1293813`–`1293824`）与 Exp2（`1293825`–`1293837`）已全部完成且逐项验证通过，分别为 **12/12** 和 **13/13** 有效点；Qwen Exp3 primary 与两个 controls 的严格门禁也全部通过。
- 每次运行写入独立的 `run-<tag>` 输出目录；只有 `validation.json` 的 `all_passed=true`（Exp3 还要求 dominance gate 通过）才能作为有效测量报告。
- 三种 residency condition 的语义与验证证据见 [docs/06-sglang-execution-path.md](docs/06-sglang-execution-path.md)：`cpu_hit` 需要同一 before/after 窗口内的 per-request host 证据 + 正向 `load_back_tokens_total`/`host_hit` delta；Exp3 使用确定性前缀池 + `hit_dominance_threshold` 支配判定（不占支配即标 unsupported）。
- 每次运行写入独立的 `run-<tag>` 输出目录（UTC 时间戳 + SLURM job id，`RUN_TAG` 可覆盖），重复运行不覆盖 raw 文件。
- Gemma 32K 使用 2×A100、TP=2、`MEM_FRACTION=0.75`。其 Exp1 当前为 10/12、Exp2 为 5/13 有效；无效 cache-hit 点均因 `prefix_consistency` 失败，不能用于汇总。Gemma Exp3 单请求 smoke 通过，但 TP=2 的七个并发 load point 缺乏可判定的 per-tier 聚合证据，均标为 unsupported。
- 当前逐项台账、作业清单与重跑边界见 `docs/05-current-status.md`；以其中列出的 validation 与结果文件为准。
