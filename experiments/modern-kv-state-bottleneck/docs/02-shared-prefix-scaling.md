# Experiment 2: Shared-Prefix Scaling

## 1. 实验目标

本实验用于研究在总 context length 固定的情况下，shared-prefix reuse 增加能够节省多少 recomputation，以及更大的 reusable cache/state 是否会引入足以抵消该收益的 CPU-GPU restore cost。

实验主要回答三个问题：

1. shared prefix 增长能够减少多少计算；
2. reusable state 增长如何改变 CPU-GPU transfer 与 non-overlapped I/O stall；
3. prefix reuse 的 end-to-end 收益是否持续扩大、进入平台期，或在高 reuse 区域被 restore cost 抵消。

## 2. 实验对象

实验分别使用 Qwen3.5-9B 与 Gemma 4 12B。

两种模型保持各自原生 attention/cache/state 机制。正式运行前通过 runtime validation gate，确认 CPU-resident hit 对各 cache/state group 的 restore 行为有效。

主实验统一在 A100 40GB 上完成。

## 3. Context length 设置

本实验固定总 context length，不再做完整 context sweep。

主实验使用 32,768 tokens。辅助验证点使用 16,384 tokens。

16K 仅用于检查主趋势是否稳定，不重新展开一套与 32K 等规模的完整实验。

## 4. Shared-prefix ratio 设置

32K 主实验采用可自然落在整数 token 边界上的比例：

| Shared-prefix ratio | Shared prefix | Unique suffix |
|---|---:|---:|
| 0% | 0 | 32,768 |
| 25% | 8,192 | 24,576 |
| 50% | 16,384 | 16,384 |
| 75% | 24,576 | 8,192 |
| 87.5% | 28,672 | 4,096 |

原设计中的 90% 不再作为默认点。87.5% 能保持高 reuse 压力，同时避免 32K 下出现非整数或额外 rounding 规则。

如果实际 runtime 的 cache/checkpoint granularity 对合法边界有更严格要求，则所有 prefix length 向同一合法粒度对齐，并把精确值写入实验配置。

100% reuse 不作为主实验点，因为完全没有 unique suffix 的 workload 与常规请求差异较大，对主要趋势的增益有限。

## 5. Workload 构造

每个配置中的请求共享完全相同的 prefix，并具有 request-specific suffix。

同一模型内尽量从同一批基础 token sequences 构造不同 reuse ratio，只改变“共享边界”，避免文本内容分布与 reuse ratio 同时变化。

跨模型比较匹配 exact token length、reuse ratio 与 workload structure，不要求 raw text 完全一致。

## 6. Cache-residency 条件

主要条件为 **CPU-resident hit**。

正式请求开始前，共享 prefix 对应的可复用 cache/state 已存在于 CPU/offload tier。请求执行时恢复该 state，并计算 unique suffix。

同时设置两个控制条件。

### Recompute baseline

不恢复共享 prefix，对完整 context 执行正常计算。

### GPU-resident hit control

在 runtime 能够可靠控制 residency 时，使相同 prefix state 已驻留 GPU。该条件估计 prefix reuse 在没有 CPU-GPU restore cost 时的下界 latency。

0% reuse 点天然对应没有 reusable prefix 的 baseline，不人为构造无意义的 CPU-resident hit。

## 7. 并发与 output 条件

实验保持固定低负载，使 queueing 与 scheduler contention 不成为主要变量。

所有请求使用相同且较短的 output length。

高并发 shared-prefix reuse 留到 Experiment 3 和后续 scheduler 实验处理。

## 8. 核心测量指标

所有 timing 定义遵循 `00-measurement-conventions.md`。

### 8.1 Reusable state footprint

记录不同 shared-prefix ratio 下实际可复用的 cache/state footprint。

能够分项时分别报告 attention KV、local/sliding-window KV 与 recurrent state。

形成：

> Shared-prefix ratio → reusable state footprint by state type

### 8.2 Computation saving

比较 recompute baseline 与 CPU-resident/GPU-resident reuse 下的 computation path。

形成：

> Shared-prefix ratio → avoided recomputation / residual computation

该结果量化 prefix reuse 理论上释放的计算量。

### 8.3 CPU-GPU transfer

CPU-resident hit 记录：

- transferred bytes per request；
- transfer activity/duration；
- achieved bandwidth。

Transfer duration 允许与 computation 重叠，因此不直接作为 TTFT 的可加项。

### 8.4 Non-overlapped I/O stall

记录真正阻塞 service path 的 restore stall，并计算：

```text
service stall ratio = I/O stall / service time
```

该指标用于判断 reuse 增长后，state restore 是否逐渐主导执行路径。

### 8.5 TTFT 与 reuse benefit

记录各 residency condition 的 client-observed TTFT。

定义 CPU hierarchical reuse 的净收益：

```text
Net Reuse Benefit = TTFT_recompute - TTFT_CPU-hit
```

定义 speedup：

```text
Reuse Speedup = TTFT_recompute / TTFT_CPU-hit
```

如果 GPU-resident hit 可用，还计算 CPU restore penalty：

```text
CPU Restore Penalty = TTFT_CPU-hit - TTFT_GPU-hit
```

这样可以把“reuse 节省的计算”和“CPU restore 吃掉的收益”分开。

## 9. 实验执行方式

每个配置先执行 warm-up，再进行多次独立重复测量。

正式结果至少报告 median、P90 与波动范围。

不同 reuse ratio 采用交替或随机顺序执行。每个配置开始前恢复规定的 cache residency，防止前一个实验点改变当前缓存状态。

Qwen3.5 与 Gemma 4 使用相同的 exact context token 数、prefix ratio、output length 和低负载条件。

## 10. 最终结果组织

实验形成四组核心结果：

1. **Shared-prefix ratio → reusable state footprint by state type**；
2. **Shared-prefix ratio → avoided/residual computation**；
3. **Shared-prefix ratio → transfer volume/bandwidth 与 non-overlapped I/O stall**；
4. **Shared-prefix ratio → TTFT / Net Reuse Benefit，比较 recompute 与 CPU-resident hit，并在可用时加入 GPU-resident hit**。

16K 辅助点主要用于验证曲线方向，不要求重复全部细粒度图表。

## 11. 结果判断逻辑

### 情况 A：Reuse 收益持续扩大

shared prefix 增长后 computation 明显下降。虽然 CPU restore demand 增加，但 CPU-resident TTFT 仍持续改善。

该结果说明 hierarchical context caching 在现代模型上仍有明确价值，同时 state loading 构成真实但尚未主导的成本。

### 情况 B：Reuse 收益进入平台期

shared prefix 从低比例增加时 TTFT 明显下降，但高 reuse 区域的净收益快速减弱，同时 service stall ratio 上升。

该结果说明 prefix reuse 仍然有效，但更大的 reusable state 正受到 I/O bottleneck 限制。

### 情况 C：高 reuse 下 CPU restore 抵消计算收益

如果高 reuse 区域出现：

```text
TTFT_CPU-hit >= TTFT_recompute
```

则说明在当前模型、runtime 与硬件条件下，CPU-resident hierarchical reuse 已无法兑现其计算节省，state movement 成为主要问题。

### 情况 D：不同 state type 呈现不同 scaling

如果某模型的 reusable state 因 sliding window 或 recurrent-state checkpoint policy 呈现 bounded、stepwise 或其他非线性行为，应直接报告该行为，不强行套用 ordinary KV cache 的线性预期。

### 情况 E：模型之间趋势明显不同

跨模型差异用于建立 model-dependent behavior，不直接证明 attention architecture 是唯一原因。

## 12. 与 Experiment 1 的分工

Experiment 1 固定 reuse ratio 并改变 context length，研究 long-context scaling。

Experiment 2 固定 context length 并改变 reuse ratio，研究 prefix reuse 的计算收益与 state-restore 代价。

统一指标和 cache-residency 术语见 [00-measurement-conventions.md](00-measurement-conventions.md)。

## 13. 执行后端

本实验可在两条执行路径上运行（evidence 不可互换）：legacy vLLM 路径与显式 SGLang / HiCache 路径（installed commit `4ad990ba7d75bb9f948f5f6bd8d79a66b5d3fd63`，仅通过公开 HTTP/Prometheus 边界驱动）。SGLang 路径的 residency 语义、验证证据与提交方式见 [06-sglang-execution-path.md](06-sglang-execution-path.md)；状态见 [05-current-status.md](05-current-status.md)。每条报告的曲线必须记录 runtime、commit/version 与 validation evidence。
