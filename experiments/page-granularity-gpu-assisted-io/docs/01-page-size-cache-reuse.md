# Experiment 1: Page Size vs. Cache Reuse

## 1. Objective

本实验定量评估 page granularity 对 cache reuse 的影响，验证在现代模型与 serving workload 下，更小的 page 是否能够减少无效缓存与无效加载，从而提高实际可复用 token 的比例。

本实验不以证明“page 越小越好”为目标，而是确定 page size 减小时 reuse benefit 是否稳定存在、收益主要出现在哪些 workload 中，以及收益从什么粒度开始趋于饱和。

实验最终需要给出一个 reuse-efficient page-size region，作为后续 I/O efficiency 实验的直接输入。

## 2. Research questions

本实验回答以下问题：

1. page size 减小时，cache hit rate 是否提高；
2. page size 减小时，实际避免重新 prefill 的有效复用 token 是否增加；
3. 小 page 的收益是否主要来自减少 page 内部的无效 token；
4. 不同 prefix overlap、context length 与 cache pressure 下，该趋势是否保持一致；
5. page size 缩小到什么范围后，reuse benefit 开始明显趋于饱和。

## 3. Independent variable

实验只系统改变 page size。

Page size 采用从小到大的多档设置，覆盖当前 serving runtime 实际能够支持的粒度范围。各档之间保持规则变化，使结果能够形成完整趋势曲线，而不是只比较两个端点。

本实验统一使用同一种普通 I/O backend，不启用 GPU-assisted I/O。这样可以避免后续 I/O 优化改变 cache 行为或性能表现，从而保持 Experiment 1 对 page granularity 本身的隔离。

## 4. Controlled variables

同一组 page-size sweep 中保持以下条件不变：

- model 与 model revision；
- hardware；
- serving runtime 与 runtime configuration；
- precision；
- cache capacity；
- cache replacement / eviction policy；
- scheduler configuration；
- I/O backend；
- 请求集合与请求顺序；
- context length；
- output length；
- prefix reuse pattern；
- concurrency 或 request arrival condition；
- random seed。

不同 page size 必须使用同一份请求 trace。这样才能把 cache reuse 的变化主要归因于 page granularity。

## 5. Workload design

实验使用三类 workload，分别覆盖可控复用、混合复用与无复用场景。

### 5.1 Controlled prefix-reuse workload

该 workload 构造具有明确共享 prefix 的请求组，并系统设置不同程度的 prefix overlap。

实验至少覆盖低、中、高三档 prefix overlap。不同 overlap 档位分别独立执行完整 page-size sweep。

该 workload 用于建立 page size 与有效复用之间最直接的关系，并减少 request ordering、热点分布和其他复杂 workload 特征的干扰。

### 5.2 Mixed-reuse workload

该 workload 包含多个不同 context。同一 context 会被重复访问，但不同请求具有不同程度的 prefix overlap。

该 workload 用于模拟更接近实际 serving 的复用模式，验证 controlled workload 中观察到的 page-size trend 是否在不规则复用条件下仍然存在。

Mixed-reuse workload 的请求集合和顺序在不同 page size 之间保持完全一致。

### 5.3 No-reuse control workload

该 workload 中不同请求之间不存在可利用的共享 prefix。

该 workload 作为负对照组，用于确认实验中观察到的 reuse 改善确实来自 page granularity 与共享状态的匹配关系，而不是 cache accounting、调度差异或其他系统噪声。

No-reuse workload 不承担寻找最佳 page size 的任务，其主要作用是验证测量链路与结论归因。

## 6. Context-length dimension

每类主要 workload 至少覆盖短、中、长三个 context 区间。

不同 context-length 档位保持相同的 prefix reuse pattern，使实验能够区分 page-size effect 是普遍存在，还是主要随着 context 增长而变得显著。

中长 context 是本实验的重点，因为 page fragmentation 和无效 cache occupancy 在更大状态规模下更可能形成实际差异。

## 7. Cache-pressure dimension

实验至少设置两种 cache pressure：

1. cache capacity 相对充足；
2. cache capacity 明显受限。

容量相对充足的配置用于观察 page granularity 本身造成的复用损失。

容量受限的配置用于观察较大的 page 是否因为无效占用增加而进一步挤出真正有价值的 reusable state。

本实验不进行大规模 cache-capacity sweep。Cache pressure 只作为必要的 robustness dimension，避免把 Experiment 1 变成 eviction-policy 或 hierarchical-cache capacity 实验。

## 8. Execution procedure

每个独立实验组由固定 model、hardware、workload、context length、prefix reuse condition 和 cache pressure 共同定义。

在一个实验组内部依次执行所有 page-size 配置。

每个配置先执行统一 warm-up，再运行正式 workload。正式测量使用完全相同的请求 trace 与随机种子。

每个配置进行多次重复运行。所有重复运行保留独立 raw result，不覆盖前一次结果。

如果某次运行出现 runtime error、cache behavior 异常、请求 trace 不一致或其他违反实验不变量的情况，该运行标记为 invalid，并保留 invalid reason，而不是直接从结果目录删除。

## 9. Metrics

### 9.1 Cache hit rate

Cache hit rate 用于描述请求所需 reusable state 中有多少能够直接命中已有 cache。

该指标不能单独作为 page-size benefit 的证据，因为较大 page 即使命中，也可能包含当前请求并不需要的大量数据。

### 9.2 Effective reused tokens

Effective reused tokens 统计实际通过 cache reuse 避免重新 prefill 的 token 数量。

这是本实验的核心指标。它直接反映 page granularity 最终保留下来的有效计算复用，而不是只记录 page-level bookkeeping hit。

### 9.3 Cache utilization efficiency

Cache utilization efficiency 衡量已缓存或已加载的数据中真正属于有效 reuse 的比例。

该指标用于识别较大 page 中由于边界不匹配产生的内部无效部分，并解释 page-level hit 与 token-level reuse 之间可能出现的差异。

### 9.4 Supporting counters

实验同时保留必要的 cache occupancy、eviction 和 reusable-state accounting 信息，用于检查结果是否受到容量压力或异常 cache behavior 的影响。

这些 supporting counters 用于解释核心指标，不替代核心指标本身。

## 10. Analysis plan

第一步绘制 page size 与 effective reused tokens 的关系，观察 page size 减小时有效复用是否稳定增加，以及曲线在什么范围开始趋于平缓。

第二步联合分析 page size、cache hit rate 与 cache utilization efficiency，判断 reuse improvement 主要来自更多有效命中，还是来自减少 page 内部无效数据。

第三步分别比较不同 prefix overlap、context length 与 cache pressure 下的 page-size curve，判断结论的适用范围。

第四步检查 no-reuse control workload。在不存在共享 prefix 的情况下，不应出现与共享复用相似的系统性 improvement。如果出现明显 improvement，需要优先检查测量口径、cache accounting 或其他未控制变量。

第五步根据多组曲线确定 reuse-efficient page-size region。该区域应满足继续减小 page size 已不能带来显著额外 reuse benefit，而不是简单选择实验中最小的 page size。

## 11. Expected result structure

正式结果至少形成以下几类输出：

1. page size vs. effective reused tokens；
2. page size vs. cache hit rate；
3. page size vs. cache utilization efficiency；
4. 不同 prefix overlap 下的对照；
5. 不同 context length 下的对照；
6. 不同 cache pressure 下的对照；
7. no-reuse control；
8. reuse-efficient page-size region 的总结表。

所有图表必须能够追溯到 processed data，再追溯到具体 raw runs 与完整配置。

## 12. Interpretation boundary

如果小 page 显著提高 effective reused tokens，可以得出 page granularity 仍然影响现代 serving 中 cache reuse efficiency 的结论。

如果 page-level hit rate 提高但 effective reused tokens 没有同步提高，则不能把 hit-rate improvement 解释为实际 reuse benefit。

如果收益只出现在高 prefix overlap、长 context 或高 cache pressure 下，应明确把结论限制在这些 workload 条件内。

如果 page size 缩小到某一区间后收益趋于饱和，该区间比“最小 page size”更适合作为后续实验输入。

本实验不判断小 page 是否具有更好的整体 serving performance，也不判断 GPU-assisted I/O 是否值得使用。这两个问题分别由后续 I/O efficiency 与 GPU-assisted I/O 实验回答。

## 13. Final conclusion target

实验一最终应回答：

> 在现代模型 serving 中，page size 减小时，有效 cache reuse 是否提高，提高多少，在什么 workload 下提高，以及收益从什么粒度开始趋于饱和。

Experiment 1 输出的 reuse-efficient page-size region 将直接进入 Experiment 2，用于继续验证这些更有利于 reuse 的 page granularity 是否同时造成更严重的 fragmented I/O 与 host→GPU bandwidth efficiency 下降。
