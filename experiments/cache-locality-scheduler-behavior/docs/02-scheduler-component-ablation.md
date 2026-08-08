# Experiment 2: Scheduler Component Ablation

## 1. 实验目标

本实验分离 Strata 三类 cache-aware scheduling mechanism 在现代 hybrid LLM serving workload 中的实际贡献。

实验只在 Experiment 1 已经冻结的 representative workloads 上运行，不重新扫描完整 cache-distance × arrival-rate space。

本实验主要回答：

1. delay-hit mitigation 是否通过等待尚未 resolve 的同-context miss，减少 redundant prefill / recomputation；
2. balanced batching 是否通过改善 batch load / compute composition，降低 exposed host-loading stall；
3. bubble filling / stall hiding 是否能够利用 balanced batching 后仍然存在的 I/O bubble，而不明显损害 decode-side QoS；
4. 三个机制的收益是互补、重叠，还是已经被现代 runtime 的其他机制吸收。

所有配置遵循 [`00-common-conventions.md`](00-common-conventions.md)。

## 2. Runtime capability gate

正式消融前必须确认当前 pinned runtime 能够实现与 Strata 三阶段机制语义等价的实验配置。

需要分别验证：

- delay-hit configuration 确实能够识别“matching context 尚未 ready”的请求并 defer，而不是仅改变普通 FIFO priority；
- balanced-batch configuration 确实依据 cache loading 与 compute requirement 影响 batch formation，并能够观测 loading-bound decision；
- stall-hiding configuration 确实在 residual loading stall 中插入可重叠的有效工作，而不是简单提高全局 concurrency。

当前 upstream runtime 是否存在名称相同的 flag 不是充分条件。

如果某机制不能被独立切换或可靠 instrument，则该机制标记为 `unsupported`。Progressive ablation 只运行语义可验证的配置，不用相似 policy 静默替代。

## 3. 固定 data path

本实验只改变 scheduler mechanism。

以下条件在所有 paired comparison 中固定：

- model / revision；
- hardware；
- cache hierarchy 与 capacity；
- cache/page policy；
- I/O backend；
- host-memory layout；
- request trace；
- offered load；
- token/batch limits。

I/O backend 在运行任何 scheduler ablation 前冻结。

如果前一实验组已经得到一个通过 validation 的高效 I/O backend，可以将其作为本实验固定 data path，使 balanced batching / stall hiding 面对的是优化 I/O 后仍然暴露的 loading stall。如果只能验证 standard-copy path，则所有 scheduler configuration 都使用该 path，并把结论限定在该数据路径下。

## 4. Representative workloads

Experiment 1 在 optimized scheduler 运行前冻结 W0–W3。

### W0: Control

Baseline pathology 很弱且系统稳定。

W0 用于检查 scheduler overhead、queueing regression 和不必要的 request deferral。

### W1: Delay-hit-sensitive

Baseline 中 delay hit / redundant prefill 明显。

优先来自短 cache distance、较高但稳定的 arrival pressure，以及较高 same-context overlap 的 workload。

W1 主要用于验证 delay-hit mitigation。

### W2: Loading-balance-sensitive

Baseline 中 CPU-resident restore / host loading 明显，batch 间 load/compute composition 具有差异，且 delay hit 不是 dominant pathology。

W2 主要用于验证 balanced batching。

如果完整 CPU hierarchy 无法验证，则 W2 标记为 unavailable，而不是通过人工增加 transfer 制造替代结果。

### W3: Residual-stall-sensitive

在固定 data path 与 baseline/balanced scheduling 下仍存在可观测 non-overlapped I/O stall 和 GPU idle interval。

W3 主要用于验证 bubble filling / stall hiding。

如果没有找到 residual stall workload，则 stall-hiding operating region 可以直接收缩，不人为制造极端 workload。

## 5. 主消融配置

主实验使用 progressive sequence：

| Configuration | Delay-hit mitigation | Balanced batching | Stall hiding |
|---|---:|---:|---:|
| S0 Baseline | × | × | × |
| S1 + Delay-hit | ✓ | × | × |
| S2 + Balanced batch | ✓ | ✓ | × |
| S3 Full scheduler | ✓ | ✓ | ✓ |

S0 → S1 测量 delay-hit mitigation 的增量。

S1 → S2 测量 balanced batching 在 delay-hit candidates 已处理后的增量。

S2 → S3 测量 residual loading stall 上的 bubble-filling 增量。

S3 已经是完整 scheduler，不再额外设置一个重复的“complete scheduler”阶段。

在 W0–W3 都可用时，主矩阵为 4 workloads × 4 configurations = 16 conditions。某 workload role 不存在或某机制 unsupported 时，矩阵按 capability status 缩减，并显式记录原因。

## 6. Targeted attribution check

Progressive ablation 能够反映实际阶段顺序，但不能自动证明某个阶段具有完全独立的因果贡献。

因此只在实现允许语义独立切换时增加 targeted leave-one-out：

- Full − Delay-hit mitigation；
- Full − Balanced batching；
- Full − Stall hiding。

每种 leave-one-out 只运行在对应 mechanism 最敏感的一个或两个 representative workloads 上。

如果实现中的阶段强耦合，关闭中间机制会改变后续机制语义，则不执行该 leave-one-out。此时 progressive ablation + mechanism instrumentation 是主要 attribution evidence。

## 7. Workload invariants

同一 W point 在不同 scheduler configuration 下保持：

- request set；
- exact arrival timestamps；
- request ordering；
- context/prefix distribution；
- input/output length distribution；
- theoretical reuse opportunity；
- reuse-distance structure；
- cache initial state / residency mode。

Scheduler 不得通过降低 offered load、丢弃请求或改变 token limit 获得表面 speedup。

实际 admission rate、effective concurrency、preemption 和 backlog 都写入 run metadata。

## 8. Delay-hit mitigation measurement

重点记录：

- cache resolve time；
- same-context overlap / fan-in；
- delay-hit event / affected volume；
- deferred request count；
- deferral waiting time；
- redundant prefill / recomputation；
- realized reuse；
- queueing delay；
- TTFT；
- throughput。

核心证据链：

```text
delay-hit mitigation
        ↓
requests wait for matching unresolved context
        ↓
fewer premature duplicate computations
        ↓
higher realized reuse
        ↓
TTFT / throughput change
```

如果 S1 相对 S0 只改变 throughput，但 delay-hit / redundant-work 指标没有对应变化，则不能把收益归因于预期的 delay-hit mechanism。

## 9. Balanced batching measurement

Balanced batching 的目标不是降低 logical restore bytes，而是改善 loading 与 computation 的组合。

重点记录：

- per-request / per-batch load amount；
- compute amount；
- aggregated load / compute ratio；
- loading-bound batch fraction；
- bundle-hit count / token volume；
- host restore volume；
- non-overlapped I/O stall；
- queueing / deprioritization time；
- GPU utilization；
- TTFT / throughput。

不复制 Strata 论文中的默认 load/compute threshold。该 threshold 是 hardware/model dependent，当前实现必须独立 calibration 或明确记录实际 fixed value 与选择依据。

核心证据链：

```text
balanced batch formation
        ↓
better load / compute composition + bundle hits
        ↓
fewer severely loading-bound batches
        ↓
lower exposed I/O stall
        ↓
TTFT / throughput change
```

Loading bytes 基本不变而 exposed stall 下降，仍然符合机制预期。

## 10. Bubble filling / stall hiding measurement

Bubble filling 只针对 S2 后仍然存在的 residual loading stall。

重点记录：

- residual I/O stall；
- GPU idle interval；
- fill opportunity；
- filled-bubble time；
- inserted work type；
- inserted useful computation；
- GPU utilization；
- TTFT / throughput；
- TPOT / decode latency；
- scheduler-induced prefill deferral。

如果实现使用 decode batch 填充 bubble，则必须检查 decode QoS。如果使用另一 prefill batch，则必须记录该选择，不能统一写成“decode insertion”。

核心证据链：

```text
residual loading-bound interval
        ↓
bubble filling
        ↓
useful overlapping computation
        ↓
lower exposed GPU idle
        ↓
end-to-end change
```

## 11. Safety / fairness metrics

所有 configuration 统一记录：

- P50/P90/P99 TTFT；
- queueing-delay distribution；
- maximum waiting time；
- TPOT 或等价 decode latency；
- starvation event；
- active-request preemption；
- offered / achieved request rate；
- throughput。

Scheduler 不能通过长期推迟某类请求换取更高 aggregate throughput 而不报告 tail-latency cost。

## 12. 实验执行流程

### Phase A: semantic validation

验证 S0–S3 的实际机制差异和 instrumentation。

### Phase B: main progressive ablation

在所有可用 W0–W3 上执行 S0–S3，多次重复并随机化运行顺序。

### Phase C: targeted attribution

仅在独立开关语义成立时运行 leave-one-out。

每个 run 保存 scheduler configuration、mechanism support status、workload identifier、repetition index 与 validity status。

## 13. 分析逻辑

### Delay hit

主要比较 W1 的 S0 vs S1，并用 W0 作为 control。

如果 W1 中 delay hit / redundant work 明显下降且端到端改善，而 W0 基本无变化，则支持 workload-dependent delay-hit mechanism。

### Balanced batching

主要比较 W2 的 S1 vs S2。

如果 logical restore volume 接近，但 load/compute ratio、loading-bound fraction 和 exposed I/O stall 改善，则支持 scheduling overlap 解释。

### Stall hiding

主要比较 W3 的 S2 vs S3。

如果 residual stall / GPU idle 被有效工作覆盖，同时 decode QoS 无明显 regression，则支持 bubble-filling mechanism。

### Full scheduler

比较 S0–S3 的总体趋势，并结合 targeted attribution 判断机制互补性。

完整 scheduler 不要求三个组件都在现代 workload 上取得明显收益。某一阶段接近 neutral 本身就是 operating-region 结论。

## 14. Control regression

W0 不可删除。

如果 W0 中不存在明显 delay hit、loading imbalance 或 residual stall，则 optimized scheduler 理论上不应得到大幅收益。

重点检查：

- scheduler overhead；
- unnecessary deferral；
- queueing tail；
- TTFT regression；
- throughput regression。

出现稳定 regression 时，该 workload 明确进入 Experiment 4 的 scheduler-not-recommended region。

## 15. 结果输出

至少形成：

1. W0–W3 的 progressive S0→S3 throughput / TTFT ablation；
2. delay-hit mitigation → delay hit / redundant work / reuse realization；
3. balanced batching → load/compute ratio / bundle hit / exposed stall；
4. stall hiding → residual stall / GPU idle / filled-bubble time；
5. P50/P90/P99 TTFT + TPOT safety summary；
6. targeted attribution table when supported；
7. mechanism support / partial / unsupported table。

## 16. 结果判断逻辑

### A. 三阶段分工清晰

三个阶段分别改善对应 pathology，并在 full scheduler 中形成互补收益。该结果支持 Strata-style control-plane design 在当前系统仍然成立。

### B. 只有部分机制有效

例如 delay-hit mitigation 有明显价值，而 balanced batching 或 stall hiding 接近 neutral。该结果说明现代 runtime 已改变 bottleneck composition。

### C. Mechanism metric 改善但端到端收益有限

该 pathology 存在，但不是 dominant serving bottleneck。不能只凭内部 metric 声称系统收益。

### D. Full scheduler 出现负面 interaction

单组件有效但 full configuration 增益下降或 tail latency 恶化。使用 targeted attribution / instrumentation 定位冲突，不隐藏结果。

### E. 所有机制都很弱

在有效 representative workloads 中内部 pathology 和端到端收益都接近 baseline。该结果说明 Strata 的 scheduler optimization space 在当前 stack 上明显收缩。

## 17. 实验边界

本实验只做 scheduler component attribution。

完整 cache-distance × load surface 已由 Experiment 1 建立。Same-context concurrency 的专门压力测试由 Experiment 3 完成。最终 operating region 由 Experiment 4 综合形成。
