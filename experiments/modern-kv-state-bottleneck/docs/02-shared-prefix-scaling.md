# Experiment 2: Shared-Prefix Scaling

## 1. 实验目标

本实验用于研究在总 context length 不变的情况下，shared prefix 比例增加是否仍然能够有效降低 TTFT，以及越来越大的可复用 cache/state 是否会带来明显的 CPU-GPU loading 压力。

实验主要回答以下问题：

1. shared prefix 增长能够节省多少重复 prefill computation。
2. shared prefix 增长是否同步增加 cache/state transfer 和 I/O stall。
3. 在现代 hybrid 模型上，prefix reuse 的收益是否存在拐点，即继续增加 reusable prefix 后，state loading 成本开始抵消 recomputation savings。

这一实验直接判断 Strata 所针对的 shared-prefix reuse 场景在现代模型上是否仍然成立。

## 2. 实验对象

实验分别使用 Qwen3.5-9B 与 Gemma 4 12B。

两种模型保持原生 attention、cache 和 state 机制，不人为统一成相同的 KV cache 形式。

主实验统一在 A100 40GB 上完成，与实验一保持相同的软件环境和硬件条件。

## 3. Context length 设置

本实验固定总 context length，不再对 context length 做完整 sweep。

主实验设置 total context length 为 32K。

同时增加 16K context 作为辅助验证点，用于判断 32K 下观察到的 shared-prefix trend 是否具有基本稳定性。

16K 不重新展开一套完整的大规模实验，主要实验自变量始终保持为 shared-prefix ratio。

## 4. Shared-prefix ratio 设置

主实验设置以下 shared-prefix ratio：

| Shared-prefix ratio | Shared prefix，32K | Unique suffix，32K |
|---|---:|---:|
| 0% | 0K | 32K |
| 25% | 8K | 24K |
| 50% | 16K | 16K |
| 75% | 24K | 8K |
| 90% | 28.8K | 3.2K |

总 context length 在所有配置下保持不变。

0% 作为无 prefix reuse baseline。

100% reuse 不作为主要实验点，因为完全没有 unique context 的 workload 与常规 serving 请求差异较大，并且对判断主要趋势的增益有限。

## 5. Workload 构造

每组请求共享完全相同的 prefix，并具有不同的 unique suffix。

例如，在 50% reuse、32K context 条件下，每个请求由 16K shared prefix 与 16K request-specific suffix 组成。

同一组中的请求复用相同 prefix，不同实验配置使用相同规模和相近分布的请求集合。

这一设计保证 shared-prefix ratio 的变化来自实验控制，而不是文本内容或请求结构的系统性变化。

## 6. Cache 条件

主实验使用 warm hierarchical cache。

在正式计时前，shared prefix 对应的 cache/state 已经生成并保存在 CPU cache 中。

正式请求到来后，shared prefix 不重新执行完整计算，对应 cache/state 从 CPU 侧加载，unique suffix 正常执行 prefill。

这一设置直接模拟 Strata 所关注的 hierarchical prefix-cache reuse 场景。

## 7. Baseline 设置

实验设置两个主要 baseline。

### Baseline A：No reuse

shared-prefix ratio 为 0%，完整 context 均执行正常 prefill。

该配置用于提供没有 prefix reuse 时的基础 TTFT。

### Baseline B：Cold recomputation

对于具有 shared prefix 的相同请求，不使用已有 cache/state，而是重新计算完整 context。

该配置用于判断使用 hierarchical cache 进行 state loading，相比直接 recomputation 实际节省了多少时间。

定义：

\[
\text{Reuse Benefit}
=
\text{TTFT}_{cold}
-
\text{TTFT}_{warm}
\]

## 8. 并发条件

实验使用固定的低并发环境。

请求之间避免产生明显 queueing 和 scheduler contention。

request rate 和 concurrency 在所有 shared-prefix ratio 下保持一致。

高并发 shared-prefix reuse 场景留到后续 concurrency 和 scheduler 实验中研究。

## 9. Output 条件

所有请求采用相同且较短的 output length。

实验主要研究 prefill 和 TTFT，因此避免生成长度成为主要变量。

## 10. 核心测量指标

### A. Reusable state size

记录每种 shared-prefix ratio 对应的 reusable cache/state size、CPU 侧 cache 占用以及每个请求需要加载的 state volume。

得到：

> Shared-prefix ratio → reusable state size

该结果用于确认不同模型实际产生多少可复用状态。

### B. CPU-GPU transfer

记录每个请求加载的 state volume 和 CPU-GPU transfer time。

得到：

> Shared-prefix ratio → transfer cost

重点判断 shared prefix 增长是否产生持续增加的 I/O pressure。

### C. Prefill latency

分别观察完整 recomputation 的 prefill latency，以及使用 cache reuse 后 unique suffix 的 prefill latency。

随着 shared prefix 增长，需要重新计算的 token 数量持续下降。

这一指标用于量化 prefix reuse 所节省的 computation。

### D. I/O stall

记录 state loading 引起的等待时间，并计算：

\[
\text{I/O Stall Ratio}
=
\frac{\text{I/O stall time}}{\text{TTFT}}
\]

该指标用于判断 shared prefix 增大时，I/O 是否逐渐成为请求的主要 latency component。

### E. TTFT

记录不同 shared-prefix ratio 下的 TTFT，并同时比较 warm cache 与 cold recomputation。

得到：

> Shared-prefix ratio → TTFT

这一结果直接反映 prefix reuse 的实际 end-to-end 收益。

### F. Net reuse benefit

进一步计算：

\[
\text{Net Reuse Benefit}
=
\text{TTFT}_{cold}
-
\text{TTFT}_{warm}
\]

以及：

\[
\text{Speedup}
=
\frac{\text{TTFT}_{cold}}{\text{TTFT}_{warm}}
\]

该指标用于判断 hierarchical cache 是否真正优于 recomputation。

## 11. 实验执行方式

每个配置首先执行固定数量的 warm-up 请求，然后进行多次独立重复测量。

正式结果至少报告 median、P90 和波动范围。

不同 shared-prefix ratio 的测试顺序交替或随机执行，减少机器状态随时间变化带来的系统性偏差。

Qwen3.5 和 Gemma 4 使用相同的 context length、prefix ratio、request count、concurrency 和 output length。

模型之间主要比较 scaling trend，而不是简单比较绝对 latency。

## 12. 最终结果组织

实验建议形成以下四组主要结果：

1. Shared-prefix ratio → reusable cache/state size。
2. Shared-prefix ratio → CPU-GPU transfer time / I/O stall。
3. Shared-prefix ratio → prefill computation cost。
4. Shared-prefix ratio → TTFT，同时展示 warm cache 与 cold recomputation。

必要时增加：

> Shared-prefix ratio → Net reuse benefit

用于直接展示 reuse 收益是否随着 prefix 增长持续扩大。

## 13. 结果判断逻辑

### 情况 A：Reuse 收益持续扩大

shared prefix 增长后，prefill computation 显著减少。虽然 transfer cost 增长，但 TTFT 仍持续下降。

该结果说明 hierarchical context caching 在现代模型上仍具有明显价值，同时 state loading 构成需要继续优化的真实成本。

### 情况 B：Reuse 收益出现明显平台期

shared prefix 从低比例增加时 TTFT 明显下降，但达到一定比例后收益迅速减弱，同时 I/O stall 占比持续提高。

该结果说明 prefix reuse 本身仍然有效，但 large reusable state 已经受到 I/O bottleneck 限制。

### 情况 C：高 reuse 反而恶化 TTFT

shared prefix 增长后，state loading cost 的增长超过 recomputation savings，并可能出现：

\[
\text{TTFT}_{warm}
\geq
\text{TTFT}_{cold}
\]

该结果说明在当前模型与硬件条件下，简单 hierarchical caching 对大 shared prefix 已经失去收益，state movement 本身成为主要瓶颈。

### 情况 D：模型之间趋势明显不同

Qwen3.5 与 Gemma 4 的 reuse benefit、state growth 或 I/O stall scaling 明显不同。

该结果说明 shared-prefix caching 的收益和瓶颈具有明显的 model-dependent 特征。

这一实验不能仅凭模型间差异把现象直接归因于某一种 attention architecture，具体原因留到后续模型泛化实验进一步分析。

## 14. 与实验一的分工

实验一固定 reuse ratio 并改变 context length，用于研究 long-context scaling。

实验二固定 context length 并改变 reuse ratio，用于研究 prefix reuse 的收益与 state-loading 代价。

两组实验共同描述 context 长度与 context reuse 两个核心变量如何影响现代模型上的 KV/state bottleneck。