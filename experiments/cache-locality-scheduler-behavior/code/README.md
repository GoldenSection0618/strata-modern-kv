# Code

本目录用于存放 “Cache Locality and Scheduler Behavior” 的实验实现。

所有实现遵循 [`../docs/00-common-conventions.md`](../docs/00-common-conventions.md) 和仓库根目录 [`../../../docs/TECHNICAL_BASELINE.md`](../../../docs/TECHNICAL_BASELINE.md)。

## Responsibilities

代码需要覆盖以下职责：

- 固定 request set 与 context/shared-prefix group 构造；
- `min-distance`、`shuffle`、`max-distance` trace 生成与 actual reuse-distance validation；
- request arrival-rate calibration 与 deterministic replay；
- Experiment 3 的 same-context burst / fan-in trace 生成，同时保持长期平均 offered load 不变；
- `cold-miss`、`gpu-ready` 和可选 `cpu-restore` resolve mode 的建立与验证；
- cache resolve time 与 unresolved same-context overlap instrumentation；
- baseline、delay-hit mitigation、balanced batching、stall hiding / full scheduler 的 mechanism-equivalent switch 或独立运行入口；
- scheduler semantic capability validation，不能只根据 option name 判断机制等价；
- delay hit、redundant work、host restore、queueing、TTFT、TPOT 与 throughput 采集；
- per-batch load / compute ratio、loading-bound decision 与 bundle-hit instrumentation；
- residual I/O stall、GPU idle、filled-bubble time 与 inserted-work type instrumentation；
- Experiment 2 representative workload selection 与 targeted attribution；
- Experiment 4 provisional map、boundary-point selection 与 operating-region aggregation；
- raw → processed → figure/table 的 deterministic processing。

## Runtime capability validation

正式 scheduler ablation 前，代码必须输出 machine-readable capability status。

至少验证：

1. delay-hit mechanism 能识别 matching context 尚未 ready 的请求，并按预期 defer；
2. balanced-batch mechanism 的决策实际依赖 load / compute information，并能够记录 loading-bound decision；
3. bundle-hit behavior 可被识别或等价观测；
4. stall-hiding mechanism 实际在 residual loading interval 中插入可重叠工作，而不是只提高全局 concurrency；
5. 每个 stage 是否可以独立启用/禁用；
6. 若阶段强耦合，targeted leave-one-out 被自动标记为 unsupported，而不是生成语义错误的结果；
7. 涉及 CPU-resident restore 的实验继承完整 hierarchy/state validation status。

Capability output 写入 run metadata，状态至少区分 `supported`、`partial`、`unsupported`。

## Workload requirements

不同 cache-distance conditions 必须使用相同 logical request set，只改变 request ordering / revisit timing。

每条 trace 必须具有稳定 identifier，并能够由版本控制配置和 seed 重建。

Trace metadata 至少保存：

- request count；
- context/prefix group identifier；
- exact arrival timestamps；
- cache-distance condition；
- actual reuse-distance summary；
- input/output token lengths；
- theoretical reusable volume；
- offered-load parameters；
- seed / configuration hash。

Experiment 3 额外保存：

- burst ID；
- target context；
- configured / observed fan-in；
- instantaneous burst rate；
- resolve mode；
- observed cache resolve time。

## Arrival-rate calibration

Experiment 1 正式 sweep 前先完成当前模型、硬件和固定 serving configuration 的 capacity calibration。

Calibration 输出至少包括 offered request rate、achieved throughput、queueing behavior、backlog status 与 repetition information，并据此冻结 Low、Medium、High 和 Overload。

Experiment 3 复用 High stable load point，不重新用更高全局 arrival rate 制造 fan-in。

## Resolve-mode validation

`cold-miss` 必须确认 target context 在 burst 前尚未 materialize。

`gpu-ready` 必须确认 target state 在请求到达前已可直接复用。

`cpu-restore` 必须确认 target state 真实存在于经过验证的 CPU tier，且 burst 开始时尚未 GPU-ready。

状态不能仅由配置推断，需要 runtime event/counter/instrumentation 证据。

## Measurement requirements

每次 run 至少保存：

- experiment ID；
- model identifier / revision；
- runtime version / commit；
- hardware、driver、CUDA/runtime；
- precision / cache dtype；
- cache hierarchy / capacity / policy；
- hierarchy validation status；
- I/O backend / host layout when applicable；
- scheduler configuration；
- mechanism capability status；
- workload trace identifier；
- cache-distance condition / actual reuse distance；
- offered / achieved request rate；
- load level；
- resolve mode / cache resolve time / fan-in when applicable；
- delay-hit count/volume；
- redundant-prefill or recomputation volume；
- host restore / duplicate restore when applicable；
- load / compute ratio and bundle hits when applicable；
- residual I/O stall / GPU idle / filled-bubble time when applicable；
- queueing delay；
- per-request TTFT；
- TPOT or equivalent decode metric；
- throughput accounting；
- repetition index；
- validity status / invalid reason。

## Processing rules

- Raw measurements 不修改或覆盖。
- Invalid / partial / unsupported runs 不删除，按 status 分开 aggregation。
- Cache-distance comparison 必须验证 request set 与 theoretical reuse opportunity 一致。
- Delay-hit analysis 不假设 cache distance 越大 pathology 越严重。
- Arrival-rate analysis 同时保留 offered 与 achieved load。
- Overload 与 stable-serving region 分开解释。
- Experiment 2 的 W0–W3 selection rule 从 Experiment 1 processed data 生成，并在任何 optimized result 出现前冻结。
- Experiment 4 boundary-point selection 也必须在新增 boundary result 出现前冻结。
- Derived metrics 由 raw/processed data 计算，不把最终数字写死在绘图脚本中。

## Suggested structure

后续实现可以按职责拆分：

```text
code/
├── configs/
├── workloads/
├── validation/
├── runners/
├── profiling/
├── analysis/
└── README.md
```

实际目录以实现规模为准，不为了形式预建无内容目录。
