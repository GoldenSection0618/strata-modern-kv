# Experiment 1: Context Length Scaling

## 1. 实验目标

本实验用于评估现代 hybrid 模型在长上下文场景下的状态管理压力，并判断 Strata 所关注的 KV / state loading bottleneck 是否仍然存在。

实验主要回答以下问题：

1. context length 增长时，模型的 cache/state 显存需求如何变化。
2. context length 增长时，TTFT 的增长主要来自 prefill computation，还是来自 CPU-GPU state transfer 与 I/O stall。
3. Qwen3.5 与 Gemma 4 是否表现出不同的 scaling behavior，以及这种差异是否会改变 Strata 的适用价值。

## 2. 实验对象

实验分别使用 Qwen3.5-9B 与 Gemma 4 12B。

两种模型均保留各自原生的 attention、cache 与 state 机制，不人为统一成相同的 KV cache 形式。

主实验首先在同一块 GPU 上完成，避免硬件差异干扰模型比较。A100 40GB 作为主实验平台，L40 48GB 留到后续硬件泛化实验中进行重复验证。

## 3. Context length 设置

实验设置 4K、8K、16K、32K 四个主要 context length。

如果两个模型在 32K 条件下仍具有充分显存余量并能够稳定运行，则增加 64K 作为额外压力测试点。

64K 不作为必须完成的主实验点，不为了加入该点而改变其他实验条件。

## 4. Shared-prefix 条件

主实验固定 shared-prefix ratio 为总 context length 的 50%。

对应关系如下：

| Context length | Shared prefix | Unique suffix |
|---|---:|---:|
| 4K | 2K | 2K |
| 8K | 4K | 4K |
| 16K | 8K | 8K |
| 32K | 16K | 16K |

实验固定 reuse ratio，而不是固定绝对 prefix length。这样可以避免 context length 增长时 reuse proportion 同时变化，从而保证主要自变量仍然是 context length。

Shared-prefix ratio 的独立变化留到后续 Shared-prefix Scaling 实验中完成。

## 5. Cache 状态

主实验采用 warm hierarchical cache 条件。

在正式请求开始前，对应 shared prefix 的 cache/state 已经生成并保存在 CPU 侧。请求执行时需要将相关状态重新加载到 GPU。

这一条件用于直接观察 Strata 所研究的 state reuse 与 CPU-GPU loading 问题。

实验同时设置 cold-cache control。在该条件下不复用已有 cache/state，而是执行完整 prefill。

Cold-cache control 只作为辅助对照，不展开为另一套大规模实验。该对照用于判断 warm cache 节省的 recomputation 是否被 state loading 成本抵消。

## 6. 并发与请求条件

实验保持低且固定的并发，使请求之间不产生明显 queueing。

Request rate 在本实验中保持不变，不引入 scheduler contention。

Concurrency 与 request-rate scaling 留到后续独立实验中完成。

因此，本实验的核心自变量保持为 context length。

## 7. Output 条件

所有请求保持相同且较短的 output length。

实验重点放在 prefill、state loading 与 TTFT，不让 decode length 成为额外变量。

## 8. 核心测量指标

### 8.1 Cache / state memory

记录 GPU cache/state memory，并分析其随 context length 增长的变化趋势。

最终形成：

> Context length → cache/state memory

重点比较两种模型的增长速度，而不仅比较绝对显存占用。

### 8.2 Prefill latency

记录完整 prefill latency，并分析其随 context length 增长的 scaling trend。

最终形成：

> Context length → prefill latency

该指标用于描述长上下文带来的基础计算压力。

### 8.3 CPU-GPU state transfer

记录每个请求产生的 CPU-GPU state transfer volume 与 transfer time。

最终形成：

> Context length → state transfer cost

该指标用于判断 hierarchical cache 是否随着 context 增长产生越来越大的数据移动压力。

### 8.4 I/O stall

记录请求执行过程中由 state loading 引起的等待时间，并计算 I/O stall 在 TTFT 中的占比。

定义：

\[
\text{I/O Stall Ratio} = \frac{\text{I/O stall time}}{\text{TTFT}}
\]

该指标用于区分 I/O 成本只是随 context 增长，还是正在逐渐成为主要 latency component。

### 8.5 TTFT

记录最终 TTFT，并将其与 prefill computation、state transfer 和 I/O stall 对应分析。

实验最终需要解释 TTFT 为什么随 context length 增长，而不是只给出一条 latency 曲线。

## 9. 实验执行方式

每一个配置先进行 warm-up，再进行多次独立重复测量。

正式结果至少报告 median、P90 与波动范围。

Qwen3.5 与 Gemma 4 使用相同的 context length、reuse ratio、请求结构和并发条件。

实验配置采用交替或随机顺序执行，降低机器长期运行状态变化对结果造成的系统性影响。

## 10. 结果组织

本实验最终形成四组核心结果：

1. Context length → cache/state memory。
2. Context length → prefill latency。
3. Context length → CPU-GPU transfer / I/O stall。
4. Context length → TTFT，并展示主要 latency composition。

每组结果同时比较 Qwen3.5 与 Gemma 4，重点分析 scaling trend。

Cold-cache control 作为辅助结果，用于比较 recomputation cost 与 cache-loading cost。

## 11. 结果判断逻辑

### 情况 A：State loading 随 context 增长逐渐成为主要开销

如果 CPU-GPU transfer 与 I/O stall 在 TTFT 中的占比明显提高，则说明现代 hybrid architecture 虽然可能降低了状态规模，但 Strata 所针对的 state-loading bottleneck 仍然存在，并在 long-context workload 下加剧。

### 情况 B：主要增长来自 prefill computation

如果 state transfer 的绝对成本增加，但其在 TTFT 中的占比保持稳定或下降，而 prefill computation 成为主要增长项，则说明 hierarchical state management 仍存在成本，但已经不是 long-context serving 的主要 scaling bottleneck。

### 情况 C：两个模型表现显著不同

如果 Qwen3.5 与 Gemma 4 的 state-loading pressure 随 context length 呈现明显不同的增长趋势，则说明 Strata 问题在现代模型上具有显著 model-dependent 特征。

该结果只用于建立跨模型现象，不直接把差异归因于某一种 attention architecture。具体原因留到后续模型泛化实验中进一步分析。

## 12. 实验边界

本实验只系统改变 context length。

Shared-prefix ratio、concurrency、request rate、page size 与 scheduler strategy 均不在本实验中进行 sweep。

这一限制用于保证实验一能够提供清晰的 context-scaling 因果画像，并为后续实验提供统一基线。
