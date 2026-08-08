# Experiment 4: Scheduler Operating Region

## 1. 实验目标

本实验确定 Strata-style scheduler optimization 在现代 hybrid LLM serving 系统中的实际有效范围。

Experiment 4 是 synthesis + boundary-validation experiment。它复用 Experiments 1–3 的 baseline profiling、component attribution 与 same-context stress 结果，只补充少量无法从已有数据确定的边界点。

本实验最终回答：

1. 哪些 workload 中 baseline scheduler 已经足够；
2. delay-hit mitigation、balanced batching、stall hiding 分别在哪些条件下具有稳定价值；
3. full scheduler 在哪里产生互补收益，在哪里接近 neutral 或出现 regression；
4. hot-context fan-in 是否需要独立的 scheduler rule；
5. scheduler-sensitive region 与 capacity-limited region 的边界在哪里。

## 2. 基础坐标系

一般 workload 以 Experiment 1 的二维空间为基础：

```text
cache distance / measured reuse timing
        ×
normalized system load
```

这里优先使用 cache distance，而不是只写“high/low locality”，因为同一个 locality 标签无法区分 delay-hit pressure 与 host-loading pressure。

Experiment 3 的 same-context fan-in 作为特殊 workload dimension 单独形成补充规则，不扩展成完整三维 sweep。

## 3. 数据来源

### Experiment 1

提供：

- actual reuse-distance distribution；
- delay hit / redundant work surface；
- host restore / I/O-stall surface when hierarchy is valid；
- queueing / TTFT / throughput；
- W0–W3 representative workload selection。

### Experiment 2

提供：

- delay-hit mitigation attribution；
- balanced-batch load/compute + bundle-hit attribution；
- stall-hiding GPU-idle / filled-bubble attribution；
- full scheduler gain；
- control workload regression；
- unsupported / partial capability status。

### Experiment 3

提供：

- same-context fan-in；
- cache resolve time；
- cold-miss delay hit；
- gpu-ready control；
- optional cpu-restore extension；
- S0/S1/S3 tail-latency behavior。

## 4. Historical reference hypothesis

Strata Figure 11 / 12 提供历史 reference，但不作为本项目预设结论。

原论文中：

- minimum cache distance 表示相同 context 请求连续出现，locality 最高；该条件更容易形成 delay hit，delay-hit mitigation 收益最大；
- maximum cache distance 降低 delay-hit likelihood，但更多 reuse 落到 CPU DRAM，host-loading pressure 更明显；
- shuffle / maximum distance 中 balanced batching 与 stall hiding 对 loading-bound behavior 继续提供收益；
- 更长 cache resolve time 与更高 request arrival rate 会增加 delay-hit misses。

现代 hybrid model/runtime 的 operating map 可以不同。Experiment 4 的职责是验证哪些方向仍然成立，而不是复写上述结论。

## 5. Operating-region 判定标准

一个 mechanism 只有同时满足以下条件，才被标记为 `effective`：

1. 对应 baseline pathology 明确存在；
2. mechanism metric 按预期方向改善；
3. 改善传导到 TTFT、throughput、tail latency 或其他明确 serving objective；
4. 没有不可接受的 queueing、decode QoS、fairness 或 starvation regression；
5. 重复运行结果稳定；
6. runtime mechanism semantics 与 cache/state capability status 已验证。

不能根据单个 throughput speedup 直接划 operating region。

## 6. Mechanism-specific regions

### R0: Scheduler-unnecessary region

Baseline 中 delay hit、host-loading imbalance、residual I/O stall 都较弱，且系统有充足余量。

该区域应优先检查 optimized scheduler 是否引入 overhead 或 unnecessary deferral。

### R1: Delay-hit region

特征为：

- same-context requests 在 unresolved miss / resolve window 内重叠；
- delay-hit affected volume 或 redundant prefill 明显；
- S1 能够通过合理 deferral 提高 reuse realization。

该区域通常与较短 cache distance、高 same-context overlap、更长 resolve time或更高但稳定的 arrival pressure相关，但最终由 measurement 决定。

### R2: Loading-balance region

特征为：

- CPU-resident restore / host loading 明显；
- batch load/compute composition 不均衡；
- exposed I/O stall 明显；
- balanced batching 能在 logical restore volume近似不变时降低 exposed stall。

该区域可能更多出现在较长 cache distance 或更高 stable load 下，但不预设固定边界。

### R3: Residual-stall region

经过 delay-hit handling 与 balanced batching 后，仍存在可观测 loading bubble / GPU idle，且存在可插入的有用工作。

只有此时 stall hiding 才具有明确作用对象。

### R4: Multi-pathology / full-scheduler region

同一 workload 同时存在多种可测 pathology，并且 full scheduler 比最有效单组件进一步改善端到端性能。

R4 不等价于“low locality + high load”。它由实际 pathology coexistence 定义。

### R5: Capacity-limited region

系统形成持续 backlog，achieved throughput 无法跟随 offered load，queueing 主要由容量不足支配。

该区域单独标记。Scheduler 在此处的 robustness gain 可以报告，但不能用于扩大正常 stable-serving operating region。

## 7. Provisional map 构建

首先不运行新实验。

汇总 Experiments 1–3 的所有 valid / partial / unsupported runs，每个 workload point 保存：

- cache-distance condition 与 actual reuse distance；
- offered / achieved load；
- cache resolve time / same-context fan-in when relevant；
- hierarchy capability status；
- dominant baseline pathology；
- S1/S2/S3 mechanism metrics；
- TTFT / throughput / TPOT；
- fairness / starvation status；
- validity status。

根据这些 measurement 建立 provisional map，并把每个 point 标记为 clear-effective、clear-neutral、regressive、capacity-limited 或 uncertain-boundary。

## 8. Boundary-point selection

只对 uncertain boundary 补充实验。

Selection rule 在任何新增 scheduler result 出现前冻结。

优先选择三类边界：

### B1: Neutral → Effective

确定某 mechanism 从无明显价值转变为稳定收益的起始区域。

### B2: Single mechanism → Full scheduler

确定什么时候单个 dominant mechanism 已经足够，什么时候多阶段组合产生额外价值。

### B3: Scheduler-sensitive → Capacity-limited

确定 High stable 与 sustained overload 之间的边界，避免把纯 saturation 错误归因于 scheduler。

建议总共补充约 4–6 个 boundary points。已有数据足够时不重复测量。

## 9. Boundary refinement

每个 selected boundary point 只沿一个变量做小幅 perturbation：

- slightly shorter / longer cache distance；或
- slightly lower / higher stable arrival pressure。

同一个 refinement 不同时改变两个维度。

如果某个 benefit 只存在于孤立单点而相邻条件立即消失，则不把它描述为稳定 operating region。

## 10. Delay-hit operating analysis

在固定 load 下，从 Min distance → Shuffle → Max distance 比较 delay-hit-related metrics。

重点验证：

- delay-hit mitigation 是否在短 cache distance / 高 same-context overlap 区域更有效；
- cache distance 增大后 delay-hit benefit 是否下降；
- arrival pressure 与 cache resolve time 是否放大短-distance delay hit；
- gpu-ready control 是否证明该额外 pathology 确实来自 unresolved context，而不是普通并发。

不再使用“locality 越差，delay-hit mitigation 越有效”的单调假设。

## 11. Host-loading / balanced-batch analysis

在 full hierarchy capability 有效时，比较 cache distance 增大后：

- CPU-resident hit / restore volume；
- load/compute ratio；
- loading-bound batch fraction；
- bundle-hit behavior；
- exposed I/O stall；
- balanced batching benefit。

如果 max-distance 中 delay hit 很弱但 loading-related scheduler benefit 增强，这与 Strata 的原始机制链一致。

如果现代 hybrid state 使 host-loading pressure 本身很弱，则对应 operating region应明确收缩。

## 12. Stall-hiding analysis

Stall hiding 的 region 只由 residual stall 定义。

需要证明：

```text
S2 still has exposed loading bubble
        ↓
S3 fills a measurable portion with useful work
        ↓
GPU idle decreases
        ↓
end-to-end metric improves without QoS regression
```

如果现代 I/O path 已经几乎消除 residual stall，则 stall hiding 接近 neutral 是合理结论。

## 13. Load-direction analysis

在固定 cache-distance condition 下比较 Low → Medium → High → Overload。

最终至少区分：

```text
resource-rich
→ scheduler-sensitive
→ capacity-limited
```

不同 mechanism 的 transition point 可以不同。

Delay hit 可能随 arrival pressure 增强；balanced batching / stall hiding 可能在 host-loading 进入 critical path 后才出现价值；进入 sustained overload 后，queueing 需要单独解释。

## 14. Same-context concurrency rule

Experiment 3 不扩展为完整 cache-distance × load × fan-in cube。

最终形成一个单独规则层，例如：

```text
normal overlap
→ use general cache-distance × load map

high cold-miss fan-in
→ evaluate delay-hit mitigation first

high fan-in + independent host-loading pressure
→ evaluate full scheduler
```

如果 C0–C3 在 baseline 下均稳定，则不建立独立 hot-context region。

CPU-restore control 只用于说明 unresolved loading path 是否存在类似 coordination issue，不与 cold-miss 主规则混合。

## 15. Benefit classification

每个 mechanism × workload point 分类为：

### Effective

Mechanism metric 与端到端 objective 均稳定改善，没有明显 safety regression。

### Mechanism-only

内部 pathology 改善，但端到端收益有限。说明机制有效但不是 dominant bottleneck。

### Neutral

Mechanism metric 与端到端结果都接近 baseline。

### Regressive

Queueing、tail latency、decode QoS、fairness 或 throughput 稳定恶化。

### Unsupported / Partial

当前 runtime 无法建立所需 mechanism semantics 或完整 cache/state capability。该状态不参与 full-mechanism performance map，但保留为 capability evidence。

## 16. Regression region

必须明确标出 optimized scheduler 不值得启用的条件，而不是只报告 speedup 最大值。

典型候选包括：

- resource-rich 且 pathology 很弱；
- context 已 gpu-ready，delay-hit deferral 没有作用对象；
- host loading 很弱，balanced batching 只增加 queue scanning / waiting；
- residual stall 很低，bubble filling 只引入调度复杂度；
- deferral cost 大于 avoided redundant work。

最终是否属于 regression region 由 measurement 决定。

## 17. Cross-model / hardware handoff

Experiment 4 只在本组 primary model / primary hardware 上形成完整 operating map。

后续 Model and Hardware Generalization 组从这里冻结少量代表点：

- neutral/control；
- clear delay-hit benefit；
- clear loading/balance benefit；
- full-scheduler benefit；
- boundary point；
- hot-context point when applicable。

后续 generalization 不重新跑整个 map，而是检查 region-level direction 是否跨模型和硬件稳定。

## 18. 实验执行流程

1. 汇总 Experiments 1–3 的有效 runs 与 capability status。
2. 建立 provisional mechanism-specific map。
3. 冻结 4–6 个以内 uncertain boundary points。
4. 只运行必要的 Baseline / candidate mechanism / Full scheduler。
5. 对 boundary 做单变量小扰动 validation。
6. 形成最终 operating map 与 decision matrix。

## 19. 结果输出

### Figure A: Mechanism-specific maps

分别展示：

- delay-hit mitigation region；
- balanced-batching region；
- stall-hiding region。

横轴使用 cache distance / observed reuse timing，纵轴使用 normalized stable load。

### Figure B: Full-scheduler operating map

标记：

- scheduler unnecessary；
- delay-hit dominated；
- loading-balance dominated；
- residual-stall dominated；
- multi-pathology / full scheduler beneficial；
- capacity limited；
- unsupported / partial。

### Figure C: Boundary validation

展示 selected boundary points 附近 Baseline、candidate mechanism、Full scheduler 的 mechanism metric 与端到端 metric。

### Figure D: Hot-context rule

展示 C0–C3 下 S0/S1/S3 的 delay hit、reuse realization 与 P99 TTFT。

### Final decision matrix

最终表格使用实际结果填写：

| Workload signature | Dominant pathology | Recommended configuration | Evidence |
|---|---|---|---|
| resource-rich / weak pathology | none | baseline | ... |
| unresolved same-context overlap | delay hit | ... | ... |
| host-loading imbalance | loading-bound batch | ... | ... |
| residual loading bubble | GPU idle | ... | ... |
| multiple pathologies | mixed | ... | ... |
| sustained overload | capacity limit | no normal-region claim | ... |

## 20. 结果判断逻辑

### A. 存在清晰 mechanism-specific operating regions

说明 Strata scheduler 在现代系统中仍然有效，但其价值具有明确 workload dependency。

### B. 只有 delay-hit mechanism 保留明显区域

说明现代 data path 已弱化 loading-related scheduler bottleneck，但 same-context coordination仍有价值。

### C. 只有 loading/bubble mechanisms 保留明显区域

说明当前 runtime 已较好解决 delay hit，但 host restore 仍可能成为关键路径。

### D. Operating region 很小

内部 pathology 偶尔存在，但大多数 stable workload 无端到端收益。说明现代 serving stack 已显著压缩 Strata scheduler optimization space。

### E. 主要收益只在 sustained overload 出现

不能据此声称正常 serving workload 下 scheduler 仍然重要，只能作为 overload robustness 结果。

## 21. 实验边界

Experiment 4 不重新执行完整 cache-distance sweep、完整 component ablation、完整 fan-in sweep 或 model × hardware sweep。

本实验只补充决定 operating-region boundary 所必需的少量 workload，并把 Experiments 1–3 收束为最终 scheduler applicability 结论。
