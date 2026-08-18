# Experiment Suite Current Status

> 核验快照：2026-08-18（Asia/Shanghai）。结论仅基于服务器上的
> `validation.json`、`summary.json`、raw 数据和 Slurm 退出状态；已完成的
> Slurm 作业并不自动等同于有效测量。
>
> 设计见 `00`–`04`，SGLang / HiCache 的配置和证据规则见
> `06-sglang-execution-path.md`。

## 0. 当前结论

SGLang / HiCache 是 Exp1–3 的主证据路径，固定运行环境为
`envs/sglang-hicache-cu129-torch211`，安装提交为
`4ad990ba7d75bb9f948f5f6bd8d79a66b5d3fd63`，Qwen 的正式 HiCache 配置为
`ratio=3 / direct / page_first_direct`。

- **Qwen3.5-9B：Exp1、Exp2、Exp3 均已有完整有效的 SGLang 数据。**
  Exp1 为 12/12，Exp2 为 13/13，Exp3 primary 与两个 control 均通过各自
  validation / strict gate。
- **Gemma 4 12B：32K 需 2×A100 TP=2，Exp1 仅 10/12、Exp2 仅 5/13 有效，
  Exp3 仅完成单请求 smoke。** 其余结果保留为失败或 unsupported 证据，不能
  用于论文曲线或跨模型结论。
- **vLLM 是 legacy/reference 路径。** 其中只有 Qwen recompute 的历史结果
  可用；cache-hit 验证仍受 V1 stats 采集限制，不能与 SGLang 结果混用。

当前没有本实验正在运行或排队的 Slurm 作业。

## 1. Qwen3.5-9B：可报告的 SGLang 数据

| 实验 | 覆盖 | 有效性 | 证据 / 产物 |
|---|---|---|---|
| Exp1：context scaling | 4K / 8K / 16K / 32K × recompute / gpu_hit / cpu_hit | **12/12 PASS** | formal jobs `1293813`–`1293824`；每项 `validation.json: all_passed=true` |
| Exp2：shared-prefix scaling | 32K、0 / 25 / 50 / 75 / 87.5%（0% 仅 recompute） | **13/13 PASS** | formal jobs `1293825`–`1293837`；每项 `validation.json: all_passed=true` |
| Exp3：request-rate scaling | 32K、50% prefix；primary cpu_hit + recompute / gpu_hit controls | **通过** | primary `1293163` + gate `1293164`；controls `1293758` / `1293759` + gates `1293760` / `1293761` |

Exp1/2 的提交清单为
`results/sglang/exp12-pipelines/20260812T160717Z-exp1-exp2-qwen/jobs.txt`。
作业在 A100 `smtg5001` 上以 `afterok` 串行执行，避免同节点 CPU、DRAM 与
HiCache 路径互相干扰。

Qwen Exp3 的 primary `cpu_hit` 以及两个 frozen-rate control 都通过严格门禁；
primary 的七个归一化负载点也都通过 residency-dominance 判定。因此三组 Qwen
SGLang 数据可进入后续的处理和作图阶段。

一个可复核的 32K / 50% prefix 对比是：recompute median TTFT `2635.663 ms`、
gpu_hit `1428.541 ms`、cpu_hit `1546.407 ms`。完整数字必须继续从对应 run
目录内的 `summary.json` 与 raw 数据生成，不应手工复制到图表脚本。

## 2. Gemma 4 12B：已完成、无效与阻塞

### 2.1 固定可运行配置

Gemma 的单张 A100 在本工作负载上最多约容纳 29,248 tokens，不能覆盖 32K
设计点。有效的启动配置是 **2×A100、TP=2、`MEM_FRACTION=0.75`**；`0.85`
曾在 NCCL all-reduce 分配额外 256 MB 时失败。该配置已通过 32K 单请求
cpu_hit validation（job `1294801`）。

`/v1/tokenize` 曾因 Gemma checkpoint 的 `tokenizer_config.json` 中 sentinel
`model_max_length` 超出 ORJSON 64-bit 范围而返回 500。runner 现在优先使用
公开 endpoint，出现该特定失败时才延迟加载同一 checkpoint 的本地
`AutoTokenizer` 生成 exact token ids；该兼容路径已经通过纯 Python 单元测试。

### 2.2 Exp1 / Exp2 的正式批次

| 实验 | 已完成 Slurm 作业 | 有效点 | 不可报告点 |
|---|---|---:|---|
| Exp1 | `1294881`–`1294896` | **10/12** | 32K `gpu_hit`、32K `cpu_hit` |
| Exp2 | `1294897`–`1294909` | **5/13** | 所有非零 prefix ratio 的 gpu_hit / cpu_hit（8 点） |

对应提交清单：
`results/sglang/exp12-pipelines/20260812T192943Z-exp1-exp2-gemma/jobs.txt`。
所有作业均正常退出，但不通过 validation 的点仍是无效数据：

- 失败点的 tier evidence 本身存在（GPU hit / host hit 均可验证）；
- 失败统一来自 `prefix_consistency`：reference output token 为 `805`，
  cache-hit 请求为 `236779`；
- 因此这些结果没有被重标、没有被零填充，也没有用于任何汇总。

当前只确认到输出一致性失败这一观测，**尚未把它归因于采样参数、缓存语义或
TP 行为中的任何一种**。修复前必须先缩小原因并在一个 32K cache-hit validation
点上重新验证，不能直接重跑整批实验。

### 2.3 Exp3 状态

Gemma TP=2 的单请求 Exp3 smoke（`1294801`）通过完整 validation，证明 32K
启动、tokenization fallback 与孤立 host-hit 路径均可用。但是它的七个并发
load-point 都因 TP 聚合后没有可判定的 device/host tier ratio 而被标为
`unsupported`（`residency_dominance_ok=false`）。这些 TTFT/throughput 数值
不是 Gemma Exp3 正式结果。

Gemma 的下一步有两个独立前置条件：先解决 32K cache-hit 的
`prefix_consistency`，再使 TP=2 的并发窗口正确汇总 per-tier evidence；两者
完成并各自通过最小验证后，才重提受影响的 Exp1/2 hit 点与 Exp3 正式 sweep。

## 3. legacy vLLM 状态

vLLM 0.26.0 路径只保留为历史 reference：Qwen Exp1 的四个 recompute 点和
Exp2 的五个 recompute 点有原始结果。其 gpu_hit/cpu_hit 无有效结果，因为 V1
engine 的 `KVCacheManager` 位于 EngineCore 子进程，旧 stats collector 无法读取
prefix-cache 计数，validation 因而失败。SGLang 与 vLLM 的证据来源不可互换，
不得将两条路径的数据拼成同一条 cache-hit 曲线。

## 4. 后续顺序

1. 对 Gemma 的 32K cache-hit 输出一致性做最小定向诊断和 validation 重试；
2. 修复后仅重跑 Exp1 的 2 个无效点和 Exp2 的 8 个无效 hit 点；
3. 为 Gemma TP=2 完成 Exp3 tier-metric 聚合，再运行一次 smoke，最后才提交正式 load sweep；
4. 当两模型对应数据均通过 validation 后，运行 Exp4 synthesis 和绘图。

## 5. 关键路径

- 实验根目录：`/share01/hpc/humxlab_intern/yanglihan/dl-stack/projects/strata-modern-kv/experiments/modern-kv-state-bottleneck/`
- SGLang 结果：`results/sglang/exp1|exp2|exp3/`
- SGLang 实现：`code/sglang_hicache/`
- canonical 环境方案：`code/envs/sglang/SGLANG_ENV_PLAN.md`
- Slurm 日志：`/share01/hpc/humxlab_intern/yanglihan/logs/ylh-*.out/.err`
