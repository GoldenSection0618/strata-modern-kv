# Cache Locality and Scheduler Behavior

本目录用于评估 Strata 的 cache-aware scheduling 机制在现代 hybrid LLM serving workload 中是否仍然具有实际价值，并确定这些机制的有效 workload 区域。

本部分重点研究 request ordering、cache distance、request arrival pressure、same-context overlap 与 cache-loading stall 之间的关系。实验不预设 Strata 的 scheduler 在现代 runtime 上一定有效，而是先定位 pathology，再做机制消融，最后形成 operating region。

## Scope

本部分包含四个实验：

1. **Locality × Arrival Rate Baseline Profiling**：只使用 baseline scheduler，在固定 cache/I/O configuration 下联合控制 cache distance 与 request arrival rate，建立 delay hit、redundant prefill、host-loading stall、queueing、TTFT 和 throughput 的基础画像。
2. **Scheduler Component Ablation**：在 Experiment 1 选出的 representative workloads 上，依次验证 delay-hit mitigation、balanced batching 与 bubble filling / stall hiding 的增量贡献，并进行少量 attribution check。
3. **Same-Context Concurrency Stress**：固定长期平均 offered load，只改变同一 context 在 cache miss resolve window 内的并发聚集程度，验证 hot-context delay hit 是否仍然存在。CPU-resident restore 作为可选扩展控制，不作为 delay hit 的定义前提。
4. **Scheduler Operating Region**：复用前三组结果，只补充少量 boundary points，形成 workload-to-mechanism operating map。

四个实验分别回答：问题在哪里、哪个机制解决哪个问题、hot-context 压力下是否仍然成立、最终什么 workload 下值得启用什么机制。

## Original-paper reference mapping

本组实验保留 Strata 的 causal questions，但不机械复制原论文 workload 或绝对参数。

- **Figure 7 / §4.3**：定义三阶段 scheduler。先处理 delay hit，再进行 balanced batch formation，最后对仍然 loading-bound 的 batch 做 bubble filling。
- **Figure 9**：展示 Strata 的整体 I/O 与 scheduling breakdown，用于说明 scheduler 需要和固定的数据路径一起解释。
- **Figure 11**：比较 Min Cache Distance、Shuffle、Max Cache Distance 下各优化的 attribution。原论文中 minimum cache distance 表示相同 context 请求连续出现，即最高 locality；此时 delay-hit mitigation 收益最明显。Maximum cache distance 降低 delay-hit 概率，但增加 CPU DRAM cache hit / host-loading 压力，因此 I/O、balanced batching 与 stall hiding 更相关。
- **Figure 12**：单独研究 delay hit 对 cache resolve time 与 request arrival rate 的敏感性，并使用 effectively unlimited cache 排除 eviction effect。

这些方向只作为历史 reference hypothesis。现代模型/runtime 的实验结果可以不同，不能把原论文结果写成预设结论。

## Terminology

本组优先使用 **cache distance / reuse distance**，避免只写“high/low locality”造成方向歧义。

- `min-distance`：相同 context 请求尽可能连续，cache distance 最小，locality 最高。
- `shuffle`：固定 seed 的随机访问顺序。
- `max-distance`：同一 context 的 revisit 尽可能分散，cache distance 最大，locality 最低。
- `cache resolve time`：从一次未就绪 context 的首次 miss 被接受开始，到该 context 对后续请求可被正确复用为止的实际时间。
- `same-context fan-in`：同一 cache resolve window 内到达并引用相同 reusable context 的请求数量。

Cache distance 不对应单一、单调的 scheduler pathology。短 distance 更容易形成 delay hit；长 distance 更可能增加 host-tier restore / loading pressure。两类效应必须分开测量。

## Directory structure

```text
cache-locality-scheduler-behavior/
├── README.md
├── docs/
│   ├── 00-common-conventions.md
│   ├── 01-locality-arrival-rate-baseline.md
│   ├── 02-scheduler-component-ablation.md
│   ├── 03-same-context-concurrency-stress.md
│   └── 04-scheduler-operating-region.md
├── code/
│   └── README.md
└── results/
    └── README.md
```

- `docs/`：统一 workload/measurement 约定与各实验设计。
- `code/`：trace 构造、runtime capability validation、scheduler mechanism implementation/toggles、指标采集、结果处理与绘图代码。
- `results/`：raw measurements、processed data、统计结果与可追溯图表产物。

Experiments 1–4 均已有详细设计。

## Core metrics

本部分重点观察：

- realized cache reuse / reuse realization；
- delay-hit count / affected volume；
- redundant prefill / recomputation；
- cache resolve time 与 same-context overlap；
- host restore volume 与 non-overlapped I/O stall；
- batch load / compute ratio；
- bundle-hit behavior；
- GPU idle / filled-bubble time；
- queueing delay；
- TTFT distribution；
- TPOT 或等价 decode-latency metric；
- achieved throughput。

Scheduler-level pathology、mechanism metric 与用户可见 serving performance 必须形成同一证据链。单独的 cache hit rate 或单独的 throughput 变化都不足以支持机制归因。

## Runtime capability gate

正式 scheduler ablation 前必须确认当前 pinned runtime 能否实现与 Strata 三阶段机制语义等价的配置。

项目不能因为 upstream runtime 提供了一个名字相近的 scheduler option，就假定它等价于 Strata 的 delay-hit deferral、balanced batching 或 bubble filling。每个 mechanism 必须能够独立启用/禁用或通过可验证 instrumentation 建立等价语义。

如果某个阶段无法被独立实现或观测，则该组件标记为 `unsupported`，不能用另一个 scheduler policy 静默替代后继续做 attribution claim。

涉及 CPU-resident cache loading 的实验还必须通过对应模型的 hierarchy/state restore validation。若只能验证部分 state group，则 loading-related scheduler 结果必须标记为 `partial`。Cold-miss delay-hit 实验可以在不依赖 CPU hierarchy 的条件下独立进行。

## Execution discipline

所有 paired scheduler comparison 使用同一模型、硬件、request trace、cache capacity、cache hierarchy 和固定 I/O backend。

I/O backend 在 scheduler ablation 开始前冻结。若使用来自前一实验组验证出的高效 I/O backend，则所有 scheduler configuration 都使用该 backend；若只能使用 standard-copy path，则结论明确限定在该数据路径下。

所有 workload trace 具有稳定 identifier，并能够由版本控制配置与 seed 重建。正式结果保留 raw run、processed result 与 figure/table 的可追溯关系。
