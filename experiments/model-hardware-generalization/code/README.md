# Code

本目录用于存放 “Model and Hardware Generalization” 实验实现。

所有实现遵循 [`../docs/00-common-conventions.md`](../docs/00-common-conventions.md) 与仓库根目录 [`../../../docs/TECHNICAL_BASELINE.md`](../../../docs/TECHNICAL_BASELINE.md)。

## Responsibilities

代码需要覆盖以下职责：

- representative-point / representative-workload configuration materialization；
- Qwen3.5-9B 与 Gemma 4 12B 的 matched workload generation；
- tokenizer-aware token/work summary；
- relative state-pressure calibration；
- per-model A100 Baseline load calibration 与 frozen arrival-schedule generation；
- A100 / L40 same-workload orchestration；
- 必要时的 matched-pressure control generation；
- baseline bottleneck profiling；
- hierarchy / I/O / scheduler mechanism capability validation；
- configured cache budget 与 resolved per-state-group GPU/CPU allocation 采集；
- frozen `common_full` feature-set validation；
- Baseline / `common_full` end-to-end execution；
- optional `best_validated_configuration` supplementary execution；
- targeted attribution execution；
- serving-state、cache、I/O、scheduler 与 serving metrics 采集；
- mixed-workload request-class tagging 与 class-level metrics；
- raw result metadata 与 validity status 管理；
- normalized-effect calculation；
- cross-model / cross-hardware / final model × hardware robustness matrix generation；
- figures and tables generation。

## Experiment 1 requirements

Experiment 1 的实现必须能够：

1. 在 A100 40GB 上运行 Qwen3.5-9B 与 Gemma 4 12B；
2. 使用同一 logical workload definition 为两个 tokenizer materialize request traces；
3. 保存 actual input/output token counts 与 offered-work summary；
4. 构造 State Pressure、Reuse and I/O、Locality and Scheduling 三类 representative workloads；
5. 在 optimized results 产生前冻结 relative pressure/load regions；
6. 对每个模型建立 baseline bottleneck profile；
7. 只启用与当前 mechanism attribution 有关的 validated system configurations；
8. 记录 full / partial hierarchy capability；
9. 对 hybrid state 分别记录可观测的 attention KV 与 recurrent/linear-attention GPU/CPU allocation、occupancy、eviction、restore 与 traffic；
10. 记录 cache/state residency、recomputation、CPU-GPU traffic、stall、queueing、TTFT 与 throughput；
11. 对同一 representative point 执行多次独立重复测量；
12. 保存 absolute measurements、normalized effect 与 uncertainty；
13. 输出 `stable`、`weakened`、`model_sensitive`、`boundary_case`、`inconclusive` 等 conclusion 所需的证据字段。

## Experiment 2 requirements

Experiment 2 的实现必须能够：

1. 对已经冻结的 representative point 在 A100 40GB 与 L40 48GB 上执行 paired hardware comparison；
2. 将 `same_workload` 作为主要 comparison type，并保持 logical trace、arrival schedule、system semantics 与 measurement rule 一致；
3. 在两个平台落入明显不同的 capacity / saturation region 时，根据预先冻结的规则生成少量 `matched_pressure` controls；
4. 将 `same_workload` 与 `matched_pressure` 使用不同 identifiers 保存和处理；
5. 对 cache/hierarchy、I/O、scheduler representative point 只运行当前 mechanism 所需的最小 system-configuration pair；
6. 保存 GPU form factor、usable memory、CPU-GPU topology、CPU、NUMA placement、host-memory policy、driver、CUDA/runtime 与实际 transfer path；
7. 保存每个平台实际解析后的 per-state-group GPU/CPU cache allocation，避免把相同配置参数误当成相同实际 capacity；
8. 在 host platform 不一致时把 comparison scope 标记为 `platform_level`，而不是默认 `gpu_silicon_only`；
9. 只有在 checkpoint revision、runtime semantics、trace、cache/state budget、measurement rule 与 validity contract 全部一致时复用已有 A100 result；
10. 分别计算 A100 与 L40 上相对自身 baseline 的 normalized effect；
11. 记录 bottleneck location、pressure proxy、mechanism observable 与 boundary shift；
12. 保留 hardware-specific unsupported / partial capability，不通过改变 mechanism semantics 强行补齐矩阵；
13. 输出 `stable`、`hardware_sensitive`、`boundary_case`、`inconclusive` 等 hardware robustness conclusion 所需字段。

## Experiment 3 requirements

Experiment 3 的实现必须能够：

1. 覆盖 Qwen3.5-9B / Gemma 4 12B × A100 40GB / L40 48GB 四种 model × hardware 组合；
2. materialize Long-context reuse、Short-context control 与 Mixed workload 三类冻结的 representative workloads；
3. 对每个模型使用 A100 Baseline 建立 Low / Medium / High 三个 primary load points，并在 optimized result 产生前冻结 exact arrival schedules；
4. 对同一个模型在 A100 与 L40 上使用相同 frozen primary arrival schedules；
5. 在 optimized result 产生前冻结 `common_full` mechanism set 和 feature-set identifier；
6. 验证 `common_full` 的每个组成 mechanism 在四种 model × platform combination 上均通过 capability gate 或经过验证的语义等价 gate；
7. 某一组合无法支持 `common_full` 时将该 primary matrix cell 标记为 `unsupported`，不得动态删减 feature set；
8. 对 primary matrix 只执行 Baseline 与 `common_full`；
9. Mixed workload 为每条请求保存 request class，并分别统计 overall 与 long-context / short-context class-level performance；
10. 同时保存 request throughput、token throughput、P50/P90/P99 TTFT、request completion time 与 GPU utilization；
11. 保存 reuse realization、recomputation、CPU-GPU traffic、I/O stall、queueing、scheduler stall/idle 与必要 batch behavior，用于解释 end-to-end result；
12. 保存 `common_full` 下实际启用的 state groups 与 resolved GPU/CPU allocations；
13. 对每个 point 同时生成 absolute measurement 与 `common_full`-vs-Baseline normalized effect；
14. 只在预定义触发条件满足时执行 targeted attribution runs，例如 abnormal gain、regression、throughput-latency trade-off、cross-class interference 或 mechanism prediction mismatch；
15. Targeted attribution 只运行定位问题所需的最小中间 configuration set，不扩展成完整五配置笛卡尔积；
16. same-workload primary result 与必要的 matched-pressure explanatory control 分开保存；
17. 可选的 `best_validated_configuration` 使用独立 comparison type，不进入 `common_full` primary aggregation；
18. 输出最终 model × hardware robustness matrix，并支持 `stable_generalization`、`model_sensitive`、`hardware_sensitive`、`boundary_shifted`、`throughput_latency_tradeoff`、`cross_class_tradeoff`、`unsupported`、`inconclusive` 等分类。

## Runtime validation

正式 runner 不得只验证模型能够启动。

至少需要验证：

- exact target checkpoint revision 与 runtime implementation；
- 当前 model × platform 组合的 native extension / CUDA path 可执行；
- 目标 serving-state group 可观测；
- full hierarchy 所需 state group 均能正确 restore；
- configured cache budget 与 runtime resolved per-state-group GPU/CPU allocation 均被记录；
- Qwen3.5 的 attention KV 与 recurrent/Gated-DeltaNet state hierarchy 分别通过 restore / eviction / numerical-consistency checks；
- I/O backend 的实际 path 与配置一致；
- scheduler mechanism semantics 与前置实验冻结的定义一致；
- Experiment 3 `common_full` feature-set identifier 在四种组合中一致；
- `common_full` 不包含当前组合未验证的 mechanism；
- paired runs 未发生未记录的 fallback；
- trace、cache/state budget、arrival/load rule 与 measurement boundary 与配置一致；
- Experiment 2 的 comparison type 与 pressure-control rule 被正确记录；
- Experiment 3 的 per-model primary load calibration、frozen arrival schedule、workload class、request-class ratio 与 targeted-attribution trigger 被正确记录；
- instrumentation failure 不会静默生成缺失指标。

Validation result 必须写入 run metadata。

## Trace metadata

每条正式 trace 至少保存：

- logical trace identifier；
- materialized trace identifier；
- model/tokenizer identifier；
- seed / config hash；
- request count；
- request identifier；
- request class when applicable；
- context / prefix identifier；
- input token count；
- reusable/shared token count；
- output target；
- realized output length after execution；
- arrival timestamp；
- reuse / revisit / locality metadata；
- workload family；
- representative-point / workload identifier；
- pressure/load region；
- primary load-calibration identifier when applicable；
- offered request/token/work summary；
- comparison type when applicable。

## Run metadata

每次 run 至少保存：

- experiment ID；
- representative-point / workload identifier；
- comparison type: cross-model / same_workload / matched_pressure / end_to_end_primary / targeted_attribution / best_validated_supplementary；
- model identifier 与 revision；
- serving runtime version / commit / build source；
- hardware platform 与 GPU form factor；
- CPU-GPU topology、CPU、NUMA placement 与 host-memory policy；
- driver、CUDA/runtime 与 PyTorch build；
- precision 与 cache/state dtype；
- system configuration；
- `common_full` feature-set identifier when applicable；
- configured GPU reusable-state budget；
- configured CPU-tier budget when applicable；
- resolved GPU allocation by state group when observable；
- resolved CPU allocation by state group when applicable and observable；
- trace identifier；
- workload family 与 request-class composition when applicable；
- pressure/load region 与 calibration identifier；
- frozen arrival-schedule identifier when applicable；
- offered request/token/work summary；
- achieved request/token throughput；
- cache/state initial condition；
- repetition index；
- runtime capability status；
- validity status 与 invalid reason；
- targeted-attribution trigger when applicable。

## Processing rules

- Raw measurements 不被 processing scripts 修改或覆盖。
- Invalid / partial / unsupported runs 不删除。
- 主 aggregation 只包含满足当前 experiment validity requirements 的 runs。
- Absolute measurements 与 normalized effects 同时保留。
- Configured budget 与 resolved state-group allocation 不合并为同一个字段。
- `same_workload` 与 `matched_pressure` 不混合 aggregation。
- Experiment 3 primary matrix 只包含相同 `common_full` feature-set identifier 的 cells。
- `best_validated_supplementary` 不进入 `common_full` robustness aggregation。
- Cross-model comparison 必须保留 actual token/work summary。
- Experiment 3 必须同时生成 overall 与 request-class-level mixed-workload aggregation。
- Saturation / relative-pressure region 使用冻结的 calibration rule，不根据 optimized result 后验移动。
- Targeted attribution 只由 versioned trigger rule 触发，并与 primary matrix 使用不同 identifiers。
- Robustness category 必须由 processed measurements 和 mechanism observables 生成，不由 plotting code 手工录入。
- Figure/table 只从 versioned processed data 生成。

## Suggested structure

后续实现可以按以下职责拆分：

```text
code/
├── configs/
├── workloads/
├── calibration/
├── validation/
├── runners/
├── profiling/
├── analysis/
└── README.md
```

实际目录以实现规模为准，不为了形式预建无内容目录。
