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
│   └── exp1_config.py          # Experiment 1 参数 dataclass
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
│   └── timing.py                # TTFT 测量与分解
├── analysis/
│   ├── __init__.py
│   └── exp1_analysis.py         # raw → processed CSV
├── validate.py                  # Validation gate
├── run_exp1.py                  # CLI 入口
├── run_exp1.sbatch              # SLURM 模板
└── submit_exp1.sh               # 批量提交脚本
```

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
