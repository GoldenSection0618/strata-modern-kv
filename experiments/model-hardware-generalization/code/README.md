# Code

本目录用于存放 “Model and Hardware Generalization” 实验实现。

所有实现遵循 [`../docs/00-common-conventions.md`](../docs/00-common-conventions.md) 与仓库根目录 [`../../../docs/TECHNICAL_BASELINE.md`](../../../docs/TECHNICAL_BASELINE.md)。

## Responsibilities

代码需要覆盖以下职责：

- representative-point / representative-workload configuration materialization；
- Qwen3.5-9B 与 Gemma 4 12B 的 matched workload generation；
- tokenizer-aware token/work summary；
- relative state-pressure calibration；
- relative serving-load / saturation calibration；
- A100 / L40 same-workload orchestration；
- 必要时的 matched-pressure control generation；
- baseline bottleneck profiling；
- hierarchy / I/O / scheduler mechanism capability validation；
- Baseline / Full Configuration end-to-end execution；
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
9. 记录 cache/state residency、eviction、restore、recomputation、CPU-GPU traffic、stall、queueing、TTFT 与 throughput；
10. 对同一 representative point 执行多次独立重复测量；
11. 保存 absolute measurements、normalized effect 与 uncertainty；
12. 输出 `stable`、`weakened`、`model_sensitive`、`boundary_case`、`inconclusive` 等 conclusion 所需的证据字段。

## Experiment 2 requirements

Experiment 2 的实现必须能够：

1. 对已经冻结的 representative point 在 A100 40GB 与 L40 48GB 上执行 paired hardware comparison；
2. 将 `same_workload` 作为主要 comparison type，并保持 logical trace、arrival schedule、system semantics 与 measurement rule 一致；
3. 在两个平台落入明显不同的 capacity / saturation region 时，根据预先冻结的规则生成少量 `matched_pressure` controls；
4. 将 `same_workload` 与 `matched_pressure` 使用不同 identifiers 保存和处理；
5. 对 cache/hierarchy、I/O、scheduler representative point 只运行当前 mechanism 所需的最小 system-configuration pair；
6. 保存 GPU form factor、usable memory、CPU-GPU topology、CPU、NUMA placement、host-memory policy、driver、CUDA/runtime 与实际 transfer path；
7. 在 host platform 不一致时把 comparison scope 标记为 `platform_level`，而不是默认 `gpu_silicon_only`；
8. 只有在 checkpoint revision、runtime semantics、trace、cache/state budget、measurement rule 与 validity contract 全部一致时复用已有 A100 result；
9. 分别计算 A100 与 L40 上相对自身 baseline 的 normalized effect；
10. 记录 bottleneck location、pressure proxy、mechanism observable 与 boundary shift；
11. 保留 hardware-specific unsupported / partial capability，不通过改变 mechanism semantics 强行补齐矩阵；
12. 输出 `stable`、`hardware_sensitive`、`boundary_case`、`inconclusive` 等 hardware robustness conclusion 所需字段。

## Experiment 3 requirements

Experiment 3 的实现必须能够：

1. 覆盖 Qwen3.5-9B / Gemma 4 12B × A100 40GB / L40 48GB 四种 model × hardware 组合；
2. materialize Long-context reuse、Short-context control 与 Mixed workload 三类冻结的 representative workloads；
3. 为每类 workload 建立 Low / Medium / High operating regions，并保存 calibration source；
4. 对 primary matrix 只执行 Baseline 与 Full Configuration；
5. 验证 Full Configuration 的每个组成 mechanism 在当前 model × platform 上均通过 capability gate；
6. Mixed workload 为每条请求保存 request class，并分别统计 overall 与 long-context / short-context class-level performance；
7. 同时保存 request throughput、token throughput、P50/P90/P99 TTFT、request completion time 与 GPU utilization；
8. 保存 reuse realization、recomputation、CPU-GPU traffic、I/O stall、queueing、scheduler stall/idle 与必要 batch behavior，用于解释 end-to-end result；
9. 对每个 point 同时生成 absolute measurement 与 Full-vs-Baseline normalized effect；
10. 只在预定义触发条件满足时执行 targeted attribution runs，例如 abnormal gain、regression、throughput-latency trade-off、cross-class interference 或 mechanism prediction mismatch；
11. Targeted attribution 只运行定位问题所需的最小中间 configuration set，不扩展成完整五配置笛卡尔积；
12. same-workload main result 与必要的 matched-pressure explanatory control 分开保存；
13. 输出最终 model × hardware robustness matrix，并支持 `stable_generalization`、`model_sensitive`、`hardware_sensitive`、`boundary_shifted`、`throughput_latency_tradeoff`、`cross_class_tradeoff`、`unsupported`、`inconclusive` 等分类。

## Runtime validation

正式 runner 不得只验证模型能够启动。

至少需要验证：

- exact target checkpoint revision 与 runtime implementation；
- 当前 model × platform 组合的 native extension / CUDA path 可执行；
- 目标 serving-state group 可观测；
- full hierarchy 所需 state group 均能正确 restore；
- I/O backend 的实际 path 与配置一致；
- scheduler mechanism semantics 与前置实验冻结的定义一致；
- Full Configuration 不包含当前组合未验证的 mechanism；
- paired runs 未发生未记录的 fallback；
- trace、cache/state budget、arrival/load rule 与 measurement boundary 与配置一致；
- Experiment 2 的 comparison type 与 pressure-control rule 被正确记录；
- Experiment 3 的 workload class、load region、request-class ratio 与 targeted-attribution trigger 被正确记录；
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
- offered request/token/work summary；
- comparison type when applicable。

## Run metadata

每次 run 至少保存：

- experiment ID；
- representative-point / workload identifier；
- comparison type: cross-model / same_workload / matched_pressure / end_to_end_primary / targeted_attribution；
- model identifier 与 revision；
- serving runtime version / commit / build source；
- hardware platform 与 GPU form factor；
- CPU-GPU topology、CPU、NUMA placement 与 host-memory policy；
- driver、CUDA/runtime 与 PyTorch build；
- precision 与 cache/state dtype；
- system configuration；
- GPU reusable-state budget；
- CPU-tier budget when applicable；
- trace identifier；
- workload family 与 request-class composition when applicable；
- pressure/load region 与 calibration identifier；
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
- `same_workload` 与 `matched_pressure` 不混合 aggregation。
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
