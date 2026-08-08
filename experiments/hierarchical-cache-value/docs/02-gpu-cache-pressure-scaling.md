# Experiment 2: GPU Cache Pressure Scaling

## 1. 实验目标

本实验用于研究 GPU reusable-cache pressure 如何影响 hierarchical cache 的实际收益，并确定 CPU tier 的价值从什么容量压力区域开始出现。

实验固定模型、workload、prefix reuse、request ordering、serving load 和 scheduler policy，只系统改变 GPU 可用于 reusable cache/state 的容量预算。

本实验主要回答三个问题：

1. reusable GPU capacity 下降时，GPU eviction 与 recomputation 如何变化；
2. hierarchical cache 能否通过 CPU tier 接管被驱逐但后续仍会 revisit 的状态；
3. hierarchy 的 TTFT / throughput 收益是否存在清晰的 value-onset region，以及压力继续增加后收益是否受 transfer/stall 限制。

所有配置遵循 [`00-common-conventions.md`](00-common-conventions.md)。

## 2. 实验对象

本实验在 Experiments 1–3 的 validated primary model 上执行完整 sweep，默认平台为 A100 40GB。

Experiment 2 不在两个模型上各自做完整 sweep。第二个模型只在 Experiment 4 进行少量 matched validation，从而避免与项目级 generalization 重复。

## 3. 核心自变量

本实验唯一系统 sweep 的变量是 **GPU reusable-cache budget**。

在正式 sweep 前先执行 calibration，确定固定 serving load 在不发生 active-request preemption 时所需的最低运行容量。主实验只压缩该条件之上的 reusable-cache headroom。

Pressure level 使用实际 GPU hit/eviction 行为验证，不仅根据配置的显存比例命名。

主实验至少覆盖四个区域：

| Pressure level | 实际系统状态 |
|---|---|
| Low | reusable working set 基本能被 GPU 覆盖，eviction 很少 |
| Medium | 开始出现稳定 reusable-state eviction |
| High | GPU 明显无法覆盖 reusable working set，eviction 频繁 |
| Very High | reusable state 高度竞争，但固定 active workload 仍无 preemption / OOM |

Very High point 如果无法在不触发 active-request preemption 的情况下建立，则不强行保留。主曲线宁可停在 High，也不把 scheduler preemption 混入 reusable-cache pressure。

## 4. 对照配置

每一个 pressure point 都严格配对运行：

- **GPU-only**；
- **GPU + CPU hierarchical cache**。

两种架构使用完全相同的 GPU cache budget、CPU-independent workload trace 和 serving load。

Hierarchical 配置的 CPU tier 容量保持足够，使 CPU eviction 不成为第二个自变量。

## 5. Workload 设计

实验使用固定 shared-prefix workload。

请求由多个 prefix groups 构成。同一 group 内存在稳定 revisit，不同 group 之间保持独立。

Prefix length、prefix revisit fraction、reuse-distance pattern、request ordering、context/output distribution 和总请求数在所有 pressure point 中完全一致。

Working set 需要足够大，使降低 reusable-cache budget 后能够从低 eviction 平滑进入高 eviction，但不能大到所有 pressure point 都直接处于 thrashing。

Experiment 2 不改变 prefix reuse，也不改变 cache locality。

## 6. 初始状态

主实验使用 **warm-cache steady-state**。

每轮先执行固定 cache-population trace，并验证实际 GPU/CPU cache occupancy。随后进入正式测量阶段。

Cold-cache 不做完整 pressure sweep，因为 Experiment 1 已经单独研究 initial-state effect。可以保留一个代表性 cold-cache sanity check，但不进入主曲线。

## 7. Validity conditions

每个主实验 run 必须满足：

- full-hierarchy restore path 已通过验证；
- active-request preemption count 为 0；
- effective concurrency 与 offered load 不因预算变化而改变；
- CPU tier 未发生未控制的 capacity eviction；
- 两种 architecture 使用相同 workload trace 与 GPU budget；
- pressure level 的实际 GPU eviction/hit 与预期区域一致。

如果降低 GPU budget 导致 scheduler preemption、OOM 或 effective concurrency 改变，该点从主 hierarchy-capacity curve 中剔除并单独记录原因。

## 8. 实验执行过程

从 Low pressure 开始逐步降低 reusable GPU capacity，直到达到最高仍有效的 pressure point。

每个 point 分别运行 GPU-only 与 hierarchical。配对配置的执行顺序交替或随机化，而不是始终固定先跑某一种 architecture。

每轮进行 warm-up/cache population，然后执行足够长的正式 trace，使 hit、eviction、restore、recomputation 和 throughput 进入稳定状态。

不同 pressure point 之间重新初始化 cache。

## 9. 核心测量指标

### 9.1 GPU cache hit / eviction

记录 GPU hit volume 和 eviction volume，用于证明 pressure sweep 实际改变了 reusable-cache capacity pressure。

### 9.2 CPU cache hit

Hierarchical 配置记录 CPU hit volume，并尽可能按 state group 分项。

Eviction 增加但 CPU hit 不增加，意味着被驱逐状态缺乏后续 revisit，不能简单解释为 offload 无效。

### 9.3 Recomputation

记录 GPU miss 导致的 recomputation，并计算 hierarchical 相对 GPU-only 避免的 recomputation。

### 9.4 CPU-GPU traffic / stall

记录 restore traffic、transfer activity 和能够测量时的 non-overlapped restore stall。

### 9.5 Serving performance

记录：

- median / tail TTFT；
- steady-state throughput；
- achieved request rate；
- active-request preemption count。

## 10. 派生指标

### Recomputation reduction

```text
recomputation reduction
=
recompute_GPU-only - recompute_hierarchical
```

### Relative TTFT improvement

```text
relative TTFT improvement
=
(TTFT_GPU-only - TTFT_hierarchical)
/
TTFT_GPU-only
```

### Throughput gain

```text
throughput gain
=
throughput_hierarchical / throughput_GPU-only - 1
```

所有派生指标必须保留对应绝对值和 raw measurements。

## 11. 结果组织

实验至少形成四组主要结果：

1. **GPU budget / pressure → GPU hit and eviction**，验证 pressure curve；
2. **GPU pressure → CPU-tier hit contribution**，判断 hierarchy 是否接管被驱逐 working set；
3. **GPU pressure → recomputation reduction + restore traffic/stall**，解释 trade-off；
4. **GPU pressure → TTFT / throughput benefit**，得到 hierarchy value curve。

## 12. 结果判断逻辑

### 区域 A：Low pressure

GPU eviction 很少，GPU-only 已覆盖主要 reusable working set。Hierarchical CPU hit 很少，端到端收益接近零。这说明 CPU tier 在该容量区域没有明显必要性。

### 区域 B：Value onset

GPU eviction 稳定增加，GPU-only recomputation 开始上升，hierarchical 出现稳定 CPU hit。如果 restore cost 小于被避免的 recomputation，TTFT 或 throughput 开始改善。

这一位置定义 hierarchy 的 **capacity value-onset region**。

### 区域 C：High pressure

CPU hit 和 restore traffic 继续增加。Hierarchy 收益可能继续扩大，也可能因为 CPU-GPU traffic / non-overlapped stall 进入平台甚至下降。

两种结果都有效，但必须用 recomputation reduction 与 restore stall 解释。

### 情况 D：Eviction 增加但 CPU hit 仍低

说明被驱逐状态本身缺乏后续 reuse。该结果指向 workload reuse，而不是 data-movement cost。Experiment 3 会独立研究这一变量。

## 13. 与其他实验的关系

Experiment 1 回答固定代表性条件下 hierarchy 是否有基础收益。

Experiment 2 隔离 **capacity pressure**。

Experiment 3 在 Experiment 2 选出的代表性 pressure region 下隔离 **prefix reuse opportunity**。

Experiment 4 只复验代表性 pressure/reuse 结论，不重复本实验完整 sweep。

## 14. 实验边界

本实验只系统改变 GPU reusable-cache budget。

Prefix reuse、cache locality、request ordering、request rate、concurrency target、context/output distribution 和 scheduler strategy 均保持固定。
