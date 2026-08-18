# Code

本目录用于存放"现代模型上的 KV / 状态瓶颈画像"相关实现。

代码应覆盖以下职责：

- workload 构造与 exact token-length 控制；
- cache-residency 初始化与验证；
- Experiment 1-3 的运行入口；
- cache/state footprint、CPU-GPU transfer、queueing、TTFT 与 throughput 的采集；
- runtime validation checks；
- raw results 到 processed results 的确定性处理；
- Experiment 4 的 cross-model 汇总与绘图。

实现必须遵循 `../docs/00-measurement-conventions.md` 的统一口径。

## Implementation rules

- 不把理论 state size 当作实际 runtime footprint 的替代值。
- 不把 raw transfer duration 与 overlapped computation 直接相加形成 TTFT decomposition。
- CPU-resident hit 必须通过 runtime counter/trace 验证后才能标记为有效 cache hit。
- model revision、runtime commit/version、precision、cache policy 和 workload 参数必须写入每次 run 的 metadata。
- 分析和绘图脚本只读取 raw/processed data，不把最终数字手工写死在代码中。

## Current structure

```
code/
├── configs/
│   ├── __init__.py
│   ├── exp1_config.py          # Experiment 1 参数 dataclass
│   ├── exp2_config.py          # Experiment 2 参数 dataclass
│   └── exp3_config.py          # Experiment 3 参数 dataclass（含 calibration / load sweep）
├── workload/
│   ├── __init__.py
│   └── token_workload.py       # Exact-token-length 工作负载构造
├── runners/
│   ├── __init__.py
│   └── vllm_runner.py           # vLLM 引擎封装 + residency 控制
├── profiling/
│   ├── __init__.py
│   ├── gpu_monitor.py           # pynvml GPU 内存采样
│   ├── vllm_stats.py            # KVCacheManager / PrefixCacheStats 采集
│   ├── timing.py                # TTFT 测量与分解
│   ├── calibration.py           # Exp3：sustainable capacity 校准
│   └── load_driver.py           # Exp3：异步负载驱动（Poisson arrival）
├── analysis/
│   ├── __init__.py
│   ├── exp1_analysis.py         # raw → processed CSV
│   ├── exp2_analysis.py         # raw → processed CSV
│   └── exp4_synthesis.py        # 跨模型综合（复用 exp1/2/3 summary）
│   └── exp4_figures.py          # 跨模型 4 组图（需要 matplotlib）
├── validate.py                  # Validation gate（recompute 跳过 cache-hit 检查）
├── run_exp1.py / run_exp1.sbatch / submit_exp1.sh
├── run_exp2.py / run_exp2.sbatch / submit_exp2.sh
├── run_exp3.py / run_exp3.sbatch / submit_exp3.sh
├── sglang/                      # SGLang / HiCache 执行路径（Exp1-3，见下节）
├── envs/sglang/                 # canonical SGLang 环境方案与已执行的 bootstrap
└── tests/                       # 纯 Python 单元测试
```

## Runtime requirements (A100 / vLLM 0.26.0)

所有 sbatch 脚本必须包含（已内置）：

- `export VLLM_USE_FLASHINFER_SAMPLER=0` — flashinfer sampler JIT 编译在 CUDA/CUB 300302 上失败；
- `export VLLM_WORKER_MULTIPROC_METHOD=spawn` — 否则 vLLM 在 fork/spawn 检测竞态下崩溃（`Cannot re-initialize CUDA in forked subprocess`）；
- Qwen 模型自动设置 `max_num_seqs=16`（Mamba cache block 预算限制，见 `runners/vllm_runner.py`）。

已知问题：`VLLMStatsCollector` 在 vLLM 0.26 V1 引擎下无法定位 `KVCacheManager`（内部对象在 EngineCore 子进程），gpu_hit/cpu_hit 验证门恒失败。修 stats 采集或放宽验证门后才能跑 hit 模式。

## SGLang / HiCache path（Explicit path for Exp1-3）

`code/sglang_hicache/` 通过公开 SGLang server 的 HTTP + Prometheus 边界驱动 Exp1-3，不复制/不 import SGLang 内部实现。本地实验包命名为 `sglang_hicache`（不是 `sglang`），保证 `code/` 下启动的 `python -m sglang.launch_server` 解析到上游 SGLang 包。设计细节见 `../docs/06-sglang-execution-path.md`。

```text
code/sglang_hicache/
├── run_exp1.py / run_exp2.py / run_exp3.py   # entry points
├── server_lifecycle.py                       # 子进程生命周期（bounded readiness + graceful stop）
├── http_client.py                            # stdlib HTTP 客户端（/generate、/flush_cache、/metrics 等）
├── metrics.py                                # Prometheus 解析 + 类型化 CacheStats + before/after delta
├── residency.py                              # recompute/gpu_hit/cpu_hit 准备 + 证据判定（纯函数）
├── prefix_pool.py                            # Exp3 确定性前缀池 + tier 支配判定
├── validation.py                             # validation gate（与 vLLM 路径同 schema）
├── summary.py / load_driver.py               # 百分位/负载摘要 + 并发 HTTP 负载驱动（Exp3）
├── workload.py / config.py / io.py / session.py
└── sbatch/                                   # ylh- sbatch + submit 脚本（full 与 smoke 分离）
code/envs/sglang/                             # canonical 环境方案与非破坏性 bootstrap
code/tests/                                   # 纯 Python 单元测试（无 CUDA/网络/权重）
```

### SGLang 用法

```bash
# 单元测试（纯 stdlib，无 CUDA/网络）
bash code/tests/run_tests.sh

# 单点运行（默认使用 canonical prefix；也可用 PYTHON_BIN 覆盖）
PYTHON_BIN=$DL_ROOT/envs/sglang-hicache-cu129-torch211/bin/python python3 code/sglang_hicache/run_exp1.py \
    --model qwen --model-path $DL_ROOT/cache/huggingface/models--Qwen--Qwen3.5-9B \
    --context-length 4096 --residency recompute \
    --output-dir results/sglang/exp1/qwen/4096 --log-dir logs

# SLURM 提交（A100；smoke 参数与 full sweep 分离；每次运行独立 run-<tag> 目录）
SMOKE=1 bash code/sglang_hicache/sbatch/submit_exp1_sglang.sh qwen smoke
bash code/sglang_hicache/sbatch/submit_exp1_sglang.sh qwen
bash code/sglang_hicache/sbatch/submit_exp2_sglang.sh qwen
bash code/sglang_hicache/sbatch/submit_exp3_sglang.sh qwen primary
FROZEN_RATES=1.2,2.4,3.4,4.1,4.9,5.6,6.3 bash code/sglang_hicache/sbatch/submit_exp3_sglang.sh qwen control

# Exp3 双 control gate 通过后，将完整 Exp1 + Exp2 串成单测量链
ROOT_DEPENDENCY=afterok:<recompute-gate-job>:<gpu-hit-gate-job> \
NODELIST=smtg5001 bash code/sglang_hicache/submit_exp1_exp2_serial.sh
```

正式 Qwen 配置固定为 `HICACHE_RATIO=3`、`HICACHE_IO_BACKEND=direct`、
`HICACHE_MEM_LAYOUT=page_first_direct`。串行提交器一次提交 Exp1 的 12 项和
Exp2 的 13 项，但通过逐项 `afterok` 保证任意时刻只有一个测量作业运行；提交
清单写入 `results/sglang/exp12-pipelines/<pipeline-tag>/jobs.txt`。该机制用于避免
同节点并行实验争用 CPU、DRAM 和 HiCache 路径，不代表跳过每项 validation gate。

SGLang 结果写入 `results/sglang/exp{1,2,3}/`，每次运行落在独立的 `run-<tag>` 目录（UTC 时间戳 + SLURM job id，可用 `RUN_TAG` 覆盖）内，raw/summary/validation/metadata 分离与 vLLM 路径一致，分析脚本（`exp{1,2}_analysis.py`、`exp4_synthesis.py`）通过递归查找直接指向对应目录。Exp3 使用确定性前缀池（`prefix_pool_size` 个互异前缀族 + `hit_dominance_threshold` 支配判定，均固定在 metadata 与 sbatch）。

## Usage

### Single run

```bash
source ~/yanglihan/env.sh
module load miniforge3/24.11.2_1

python3 code/run_exp1.py \
    --model qwen \
    --model-path $DL_ROOT/cache/huggingface/models--Qwen--Qwen3.5-9B \
    --context-length 4096 \
    --residency recompute \
    --output-dir results/exp1/qwen/4096/recompute/
```

### SLURM submission

```bash
# Single job
MODEL=qwen CTX=4096 MODE=recompute sbatch -J ylh-exp1-qwen-4k-recmp code/run_exp1.sbatch

# All jobs for a model
bash code/submit_exp1.sh qwen
```

### Analysis

```bash
python3 code/analysis/exp1_analysis.py \
    --input-dir results/exp1/qwen/ \
    --output-dir results/exp1/processed/
```

### Experiment 4 (cross-model synthesis)

```bash
# 跨模型综合：读 exp1/exp2/exp3 的 summary.json → processed CSV
python3 code/analysis/exp4_synthesis.py \
    --exp1-dir results/exp1/ \
    --exp2-dir results/exp2/ \
    --exp3-dir results/exp3/ \
    --output-dir results/exp4/processed/

# 生成 4 组跨模型图（需要 matplotlib，服务器 qwen env 未安装）
python3 code/analysis/exp4_figures.py \
    --input-dir results/exp4/processed/ \
    --output-dir results/exp4/figures/
```

exp4 只复用 exp1/2/3 的结果，不产生新测量。`exp4_synthesis.py` 自动归一化模型标识（qwen/Qwen3.5-9B → qwen），exp1 的 prefix_ratio 按设计固定 0.5；缺失的 mode（如 gpu_hit/cpu_hit 尚无数据）在 CSV 中留空并在 `synthesis_report.json` 记录可用性，不中断。
