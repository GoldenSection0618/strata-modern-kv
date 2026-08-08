# End-to-End Serving Common Conventions

本文件定义 End-to-End Serving 三个实验共享的实验口径。各分实验只改变其明确声明的 workload dimension，其余系统条件保持一致。

## 1. 实验定位

本组实验用于验证完整 serving system 的最终收益，不替代前面针对 cache hierarchy、I/O path 与 scheduler mechanism 的 microbenchmark 和 ablation。

端到端结果用于回答系统是否整体更快、更稳定。机制归因必须与前述实验结果共同解释。

## 2. 系统配置

正式比较统一使用以下五种配置：

1. **Baseline**：不启用本项目新增的 hierarchical cache、I/O 与 scheduler 优化。
2. **Hierarchical Cache**：只启用经过验证的 hierarchical cache/state path，使用 reference/baseline I/O 与 scheduler path。
3. **Hierarchical Cache + I/O Optimization**：在相同 hierarchy 上启用前置实验验证过的 I/O optimization，scheduler 保持 reference/baseline path。
4. **Hierarchical Cache + Scheduler Optimization**：在相同 hierarchy 上启用前置实验验证过的 scheduler optimization，I/O 保持 reference/baseline path。
5. **Full Configuration**：同时启用最终选定并经过验证的 hierarchy、I/O 与 scheduler mechanisms。

配置 3 与配置 4 是 parallel attribution branches，而不是严格的逐层 feature chain。Full Configuration 用于测试机制同时启用后的最终收益和 interaction。

如果某项机制在当前 pinned runtime 上无法实现与前置实验相同的语义，则该配置标记为 `unsupported` 或 `partial`，不得使用名字相近但语义不同的 runtime option 静默替代。

## 3. Paired comparison 原则

同一 workload point 的不同系统配置必须保持以下条件一致：

- model identifier 与 revision；
- hardware allocation；
- precision 与 cache dtype；
- serving runtime version / commit；
- logical request trace；
- input/output token distribution；
- request arrival schedule；
- GPU reusable-cache budget；
- batch / concurrency limits；
- sampling / generation settings；
- measurement rule 与 measurement window。

CPU tier 只存在于启用 hierarchy 的配置中，因此不能要求 Baseline 与 hierarchy 使用相同 CPU cache capacity。所有启用 hierarchy 的配置必须使用相同 CPU-tier budget、host-memory policy 和 offload policy，除非这些变量本身就是当前实验对象。

系统配置是 paired comparison 中的主要自变量。任何为了让某个配置成功运行而修改 GPU cache budget、concurrency limit、workload trace 或其他关键条件的 point，都不得进入严格 paired comparison。

## 4. Workload trace

所有 workload 使用可重复生成的 deterministic trace，并保存稳定 identifier、seed 与配置 hash。

Trace 至少记录：

- request identifier；
- request class when applicable；
- context / prefix identifier；
- input token length；
- shared-prefix length when applicable；
- output token target；
- realized output token length；
- request arrival timestamp；
- reuse / locality metadata when applicable。

不同系统配置必须消费同一逻辑 request trace。不得因为某一配置运行更慢而改变后续请求内容、到达顺序或目标 output work。

如果 sampling 会导致不同配置产生不同 realized output length，则必须记录实际生成长度，并在 throughput / completion-time 分析中保留 token-work 差异。优先使用能够稳定控制 output work 的生成设置，避免把输出随机性当成系统性能差异。

## 5. Load scaling

Load scaling 是三个分实验的共同实验维度，不单独构成第四个实验。

每类 workload 覆盖多个 offered-load 条件，从低负载逐步提高到接近或进入系统饱和区域。

主要比较覆盖：

- 低负载，用于识别 optimization fixed overhead；
- 中等负载，用于观察资源竞争开始后的收益；
- 接近饱和的高负载，用于比较 serving capacity 与 tail latency。

Load grid 在正式 paired comparison 前通过 calibration 冻结，然后所有 system configurations 使用完全相同的 offered-load points 和 arrival traces。不得根据某个配置的结果单独选择更有利的 load points。

当 queue 持续增长、achieved throughput 不再随 offered load 提升，且 latency 明显恶化时，将该配置视为进入 saturation region。不同配置可以在不同 load point 饱和，但 saturation rule 必须统一并预先确定。

## 6. Cache initial state

涉及 context reuse 的实验同时区分 cold-start 与 steady-state。

Cold-start 从规定的空 reusable-cache 状态开始，用于保留首次 context processing 成本。

Steady-state 不通过观察某个配置“看起来已经稳定”后动态开始计时。正式 steady-state measurement 使用固定、可重建的 cache-population / preconditioning trace，并在预定逻辑边界开始 measurement。预处理 trace 在所有 paired configurations 中保持一致，并记录测量开始时的实际 cache residency / occupancy。

Cold-start 与 steady-state 可以属于独立 runs。两者必须分开报告，不能把不同配置在不同时间进入 reuse 状态后的数据混成单一平均值。

Short-context regression 不要求完整的 cold/warm sweep，但必须从统一的 clean initial state 开始，并避免 warm-up 无意建立大量 reusable prefix state。

## 7. Core metrics

所有实验统一记录：

- request throughput；
- token throughput；
- P50 / P90 / P99 TTFT；
- request completion time；
- GPU utilization。

其中 throughput 与 tail TTFT 是主要系统指标。不能只依据平均 latency 或单一 throughput 数字判断整体系统更优。

TPOT 或等价 decode-latency metric 在需要解释 output-length heterogeneity、decode interference 或 batch behavior 时作为辅助指标记录。

## 8. Auxiliary metrics

为建立与前置实验的证据链，按 runtime capability 尽可能保留：

- GPU / CPU cache hit；
- reusable-state eviction；
- recomputation；
- CPU-GPU data movement；
- non-overlapped I/O stall；
- queueing time；
- scheduler stall / idle behavior；
- batch composition / batch shape when relevant。

这些指标用于解释端到端结果，不替代用户可见 serving metrics。

## 9. Mixed-workload comparison semantics

Mixed workload 中，request composition 和 output-length distribution 会直接改变单位时间的 token work 与 compute demand。因此只固定 request rate 并不能形成严格的单变量因果比较。

相关 robustness check 区分两种语义：

### Operational sensitivity

保持相同 request-arrival schedule，仅改变 composition 或 output-length distribution。

该结果回答真实业务流量结构变化后系统会怎样表现。它允许总 offered token/compute work 随 workload 变化，因此不能被解释为 composition 或 output-length heterogeneity 的纯因果效应。

### Matched-work attribution control

在需要判断某个 workload dimension 本身是否改变系统行为时，补充 matched-load control。匹配方式在正式实验前冻结，可基于 offered input/output token volume、baseline load fraction 或其他可验证的 work proxy。

Matched-work control 的目标不是制造完全相同的内部 GPU cost，而是避免“只是因为总工作量更大”成为唯一解释。

所有 mixed-workload 结果同时记录 offered request rate、offered token/work summary 和 achieved request/token throughput。

## 10. Regression 与 equivalence 判定

Short-context regression 和其他“无明显退化”结论必须使用预先冻结的 decision rule。

Regression/equivalence margin、重复次数、aggregation method 和 uncertainty reporting 在查看 Full Configuration 的正式结果前确定并写入 versioned config 或 analysis metadata。

“没有统计显著差异”不能自动等价于“性能等价”。如果实验精度不足以排除具有实际意义的 regression，则结论应写成 inconclusive，而不是 no regression。

## 11. Warm-up 与正式测量

每个配置在正式统计前完成模型加载、首次 kernel / allocator initialization 与必要 runtime warm-up。

Warm-up 请求不进入正式统计，除非分实验明确研究 cold-start 阶段。

正式 run 使用明确的 measurement window，并记录实际 completed requests、achieved request/token rate 与系统状态。

## 12. Repetition 与执行顺序

每个正式 workload point 进行多次独立重复测量。

不同系统配置的执行顺序交替或随机化，降低机器温度、后台噪声、长期内存状态等因素造成的系统偏差。

异常 run 不直接删除。Raw result 保留，并记录 validity status 与 invalid reason。

## 13. Validity status

正式结果至少区分：

- `valid`：满足当前实验全部执行与 measurement 条件；
- `partial`：某些目标 mechanism 或 instrumentation 只能部分实现；
- `unsupported`：当前 runtime 无法建立目标系统配置；
- `invalid`：发生 OOM、trace mismatch、unexpected fallback、measurement failure 或其他破坏 paired comparison 的情况。

只有符合当前结论要求的 runs 进入主 aggregation。

## 14. 结果解释边界

Full Configuration 的 throughput 提高但 P99 TTFT 明显恶化时，应报告 throughput-latency trade-off，而不能简单写成系统整体更优。

低负载下与 Baseline 接近、高负载下明显受益时，应解释为优化主要提高资源竞争条件下的 serving capacity，而不是声称所有请求都更快。

如果 cache reuse 明显存在但端到端收益有限，应结合 recomputation、data movement、queueing 与 GPU utilization 判断收益被哪个后续瓶颈抵消。

如果现代 runtime 的 Baseline 已经包含与 Strata 相近的能力，必须明确记录 baseline semantics，避免把 upstream optimization 当作本项目增量收益。

Mixed workload 的 aggregate gain 不能掩盖 request-class regression。任何 long-context 或 short-context class 的稳定 tail-latency degradation 都必须单独报告。
