# Experiment 3: GPU-Assisted I/O Compensation

## 1. Objective

本实验定量评估 GPU-assisted I/O 是否能够在小 page 条件下恢复 CPU→GPU cache/state transfer efficiency，并验证这种带宽恢复是否能够转化为实际 serving 性能收益。

本实验重点研究如下因果链：

```text
I/O backend
    ↓
transfer efficiency
    ↓
cache restore latency / non-overlapped I/O stall
    ↓
serving performance
```

Page size 本身不再作为新的独立研究对象，而是复用 Experiments 1–2 已经确定的代表性 page-size region。实验不预设 GPU-assisted I/O 一定有效，而是验证它是否能够修复由 fragmented transfer 导致的实际 I/O penalty。

## 2. Research questions

本实验回答以下问题：

1. GPU-assisted I/O 是否能够提高小 page 条件下的 sustained host→GPU bandwidth；
2. GPU-assisted I/O 是否能够恢复 fragmented transfer 场景中的 bandwidth utilization；
3. GPU-assisted I/O 的收益是否随 page size 和 fragmentation 程度变化；
4. bandwidth improvement 是否能够降低 cache/state restore latency 与 non-overlapped I/O stall；
5. I/O 层面的改善是否能够进一步转化为 prefill、decode、TTFT 或 overall throughput 的实际收益；
6. GPU-assisted I/O 是否主要在 fragmentation 明显的 operating region 中具有价值。

## 3. Experimental variables

本实验控制两个主要变量：

- page size；
- I/O backend。

I/O backend 至少包含 baseline I/O 与 GPU-assisted I/O 两种配置。

Page size 不再执行完整范围 sweep，而是从 Experiments 1–2 中选择少量代表性 operating points。这样可以避免重复前两组实验，并把实验三的主要因果变量集中在 I/O backend 上。

同一比较组中保持以下条件一致：

- model 与 model revision；
- hardware；
- serving runtime 与 runtime configuration；
- precision；
- cache capacity；
- cache replacement / eviction policy；
- scheduler configuration；
- request sequence；
- workload；
- transfer direction；
- host memory configuration；
- repetition protocol。

## 4. Page-size operating points

本实验从前两组结果中选择三个代表性 page-size operating points。

### 4.1 Large-page baseline

该点选择 Experiment 2 中 bandwidth utilization 较高、actual transfer fragmentation 较弱的 page size。

该点用于建立负对照，判断 baseline I/O 已经较高效时，GPU-assisted I/O 是否仍然具有明显收益。

### 4.2 Trade-off page size

该点选择 Experiment 1 中已经获得明显 reuse benefit，同时 Experiment 2 开始出现 I/O degradation 的 page size。

该点是实验三的主要 operating point，用于验证 GPU-assisted I/O 是否能够在保留小 page reuse benefit 的同时消除主要 I/O penalty。

### 4.3 Small-page fragmented region

该点选择 Experiment 2 中 actual transfer fragmentation 和 bandwidth degradation 均较明显的 page size。

该点用于评估 GPU-assisted I/O 的最大潜在补偿能力，并检查其在严重 fragmentation 下是否存在收益上限。

## 5. Experimental structure

实验分为两个层次：

1. Controlled I/O compensation；
2. Serving-level validation。

Controlled I/O experiment 用于验证 GPU-assisted I/O 是否直接改善相同 logical transfer workload 的传输效率。Serving-level validation 用于确认这种改善是否能够进入实际 serving critical path，并产生端到端收益。

## 6. Experiment 3A: Controlled I/O compensation

### 6.1 Purpose

Controlled experiment 在固定 logical transfer workload 下直接比较 baseline I/O 与 GPU-assisted I/O。

同一个实验配置分别通过两种 backend 执行，除 I/O backend 外其余条件保持一致。因此，该实验主要用于测量 GPU-assisted I/O 对 transfer efficiency 的直接补偿效果。

### 6.2 Transfer workload

本实验复用 Experiment 2 已验证的 controlled transfer workload，不重新执行完整 transfer-volume sweep。

实验至少保留能够代表实际 cache restoration 的 medium transfer volume 和 large transfer volume，并覆盖 contiguous transfer 与 fragmented page selection 两种 access pattern。

该设计用于同时观察低 fragmentation 与高 fragmentation 下两种 backend 的差异。

### 6.3 Comparison matrix

主要 controlled comparison matrix 为：

```text
Large page
    baseline I/O
    GPU-assisted I/O

Trade-off page
    baseline I/O
    GPU-assisted I/O

Small page
    baseline I/O
    GPU-assisted I/O
```

每个 page size 下，两种 backend 使用完全相同的 logical transfer request、数据范围和 repetition protocol。

## 7. Controlled I/O metrics

### 7.1 Sustained host→GPU bandwidth

该指标用于直接比较不同 backend 在相同 logical transfer workload 下的持续传输效率。

分析重点是 GPU-assisted I/O 相对于 baseline I/O 的 bandwidth recovery，以及这种 recovery 是否随着 page size 减小和 fragmentation 增强而扩大。

### 7.2 Bandwidth utilization

该指标将实际 sustained bandwidth 与同一硬件上的 reference bandwidth 进行归一化，用于判断 GPU-assisted I/O 能否缩小 small-page configuration 与 large-page baseline 之间的 efficiency gap。

### 7.3 Transfer completion time

该指标测量完成相同 logical data movement 所需要的时间，用于将 bandwidth improvement 转换为更直接的 cache/state restoration latency 证据。

### 7.4 Effective transfer granularity

实验记录 backend 实际执行的 transfer operation 数量与 transfer-size distribution。

该指标用于确认 GPU-assisted backend 是否真正改变 fragmented I/O 的执行方式，而不是仅根据 backend 名称或配置推断其已经生效。

## 8. Experiment 3B: Serving-level validation

### 8.1 Purpose

Serving-level experiment 将 baseline I/O 与 GPU-assisted I/O 放入真实 cache reuse workload 中比较，验证 controlled I/O 中观察到的 bandwidth recovery 是否能够降低实际 serving stall 并改善用户可见性能。

实验复用 Experiment 2 的 representative workloads 和 operating points，不额外引入新的 workload dimension。

### 8.2 Workload selection

实验保留三个代表性 serving scenarios。

#### A. Low-fragmentation workload

该场景对应较大 page 或较低 actual transfer fragmentation。

该组作为负对照，用于确认 GPU-assisted I/O 的收益是否确实集中在存在 fragmented I/O 的场景，而不是所有 workload 中都出现同等改善。

#### B. Moderate-fragmentation workload

该场景对应 Experiments 1–2 中确定的 trade-off region，同时存在明显 reuse benefit 与明显 I/O degradation。

该组是本实验最重要的 serving configuration，用于判断 GPU-assisted I/O 是否能够将 reuse-efficient page size 转化为实际可用的 operating region。

#### C. High-fragmentation workload

该场景对应大量小 page transfer 与明显 bandwidth degradation。

该组用于观察 GPU-assisted I/O 的最大潜在收益，以及改善后的系统是否会转而受到其他瓶颈限制。

## 9. Serving-level execution

对于每一个固定 workload 和 page-size operating point，分别运行 baseline I/O 与 GPU-assisted I/O。

两组运行使用相同的 request sequence、initial cache state、cache capacity、scheduler policy 和 workload parameters。

每个配置完成统一 warm-up 后执行正式测量，并进行多次重复运行。

没有实际触发目标 CPU→GPU cache/state restore 的 run 不进入 GPU-assisted I/O compensation 分析。

## 10. Serving-level metrics

每个正式 run 至少记录以下指标：

- effective cache reuse；
- actual transferred bytes；
- transfer count 与 transfer-size distribution；
- sustained host→GPU bandwidth；
- bandwidth utilization；
- transfer / restore latency；
- non-overlapped I/O stall；
- prefill throughput；
- decode throughput；
- TTFT；
- request completion time；
- overall throughput。

Cache reuse 相关指标主要用于确认 backend 切换没有改变实验 workload 的复用条件。实验三的核心分析集中在 transfer efficiency、stall 与 serving performance 之间的关联。

## 11. Analysis

实验首先比较不同 page-size operating point 下两种 backend 的 sustained bandwidth 与 bandwidth utilization。

随后比较相同配置下的 transfer completion time 与 non-overlapped I/O stall，判断 bandwidth recovery 是否真正减少进入 serving critical path 的 I/O 时间。

最后比较 TTFT、prefill throughput、decode throughput、request completion time 与 overall throughput，判断 I/O 层面的改善是否能够转化为实际系统收益。

正式分析需要验证如下完整因果链：

```text
GPU-assisted I/O
        ↓
effective transfer efficiency ↑
        ↓
cache restore time ↓
        ↓
non-overlapped I/O stall ↓
        ↓
serving performance ↑
```

如果链条在某一步中断，则结果需要按实际机制解释，而不能仅根据 bandwidth improvement 宣称端到端优化有效。

## 12. Expected result structure

正式结果至少形成以下四组图表。

### Figure A: Page size × I/O backend → Sustained bandwidth

该图直接展示不同 granularity 下 GPU-assisted I/O 的 bandwidth recovery。

### Figure B: Page size × I/O backend → Bandwidth utilization

该图展示 GPU-assisted I/O 是否能够缩小 small-page 与 large-page baseline 之间的 I/O efficiency gap。

### Figure C: Page size × I/O backend → Non-overlapped I/O stall

该图用于确认 controlled I/O 中的传输效率提升是否真正减少 serving critical-path stall。

### Figure D: Page size × I/O backend → TTFT / Throughput

该图用于验证 GPU-assisted I/O 的最终 serving benefit。

## 13. Joint interpretation with Experiments 1–2

前三个实验需要形成统一证据链。

```text
Experiment 1
page size ↓
→ effective reuse ↑

Experiment 2
page size ↓
→ actual transfer fragmentation ↑
→ I/O efficiency ↓

Experiment 3
GPU-assisted I/O
→ I/O efficiency recovery
→ I/O stall reduction
→ serving performance recovery
```

联合分析最重要的比较是 small page + GPU-assisted I/O 是否能够同时保留 small page 的 cache reuse benefit，并获得接近 large-page baseline 的 I/O efficiency。

如果这一结果成立，则说明 GPU-assisted I/O 实际缓解了 page granularity 在 cache reuse 与 transfer efficiency 之间的主要冲突。

## 14. Validity checks

正式结果必须满足以下条件：

1. 两种 backend 实际完成的 logical data movement 一致；
2. backend 切换前后的 effective cache reuse 基本一致；
3. backend 切换不改变 cache replacement policy、scheduler policy 或 request ordering；
4. actual transferred bytes、transfer count 与 transfer-size distribution 可观测；
5. GPU-assisted backend 必须确认实际生效，不允许 silently fallback 到 baseline path 后仍标记为 GPU-assisted result；
6. serving-level performance difference 必须能够与 transfer efficiency 和 non-overlapped I/O stall 的变化对应；
7. GPU utilization 上升不能直接解释为正收益，GPU-assisted I/O 的 computation interference 留在 Experiment 4 单独分析。

## 15. Interpretation boundaries

如果 GPU-assisted I/O 显著提高 bandwidth，同时降低 non-overlapped I/O stall 和 TTFT，则可以认为原有 fragmented I/O 位于 critical path，并且 GPU-assisted I/O 具有实际系统价值。

如果 bandwidth 明显提高但 TTFT 与 throughput 基本不变，则应解释为 I/O 已被 computation overlap 或系统瓶颈已经转移，而不能把 microbenchmark speedup 等同于 end-to-end benefit。

如果 bandwidth improvement 本身有限，则需要结合 effective transfer granularity 判断 runtime 是否已经进行了较充分的 aggregation，或者 fragmented I/O 是否并非当前配置的主要瓶颈。

## 16. Final conclusion target

本实验最终需要明确回答：

> GPU-assisted I/O 是否能够有效修复小 page 带来的 fragmented I/O penalty，以及这种修复是否能够转化为实际 serving 性能收益。

实验结果应尽量区分三个 operating regions：

- **No-need region**：baseline I/O 已经足够高效，GPU-assisted I/O 的额外收益有限；
- **Effective compensation region**：GPU-assisted I/O 显著恢复 bandwidth、降低 I/O stall，并带来实际 serving benefit；
- **Limited-benefit region**：GPU-assisted I/O 改善 transfer efficiency 后，整体收益仍受到其他瓶颈限制。

最关键的结论是确定 Experiment 1 中的 reuse-efficient page-size region 是否能够通过 GPU-assisted I/O 转化为同时兼顾 cache reuse 与 I/O efficiency 的实际 operating region。

Experiment 4 在此基础上继续评估 GPU-assisted I/O 占用的 GPU computation resource，以及这种 compensation 的净收益是否值得。