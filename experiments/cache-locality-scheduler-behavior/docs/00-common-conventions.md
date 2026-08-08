# Common Conventions

本文件定义 “Cache Locality and Scheduler Behavior” 实验组共享的 workload、对照与测量约定。

## 1. 变量隔离原则

本实验组把 request arrival pressure 与 cache locality 视为主要 workload 变量。

同一组对照实验使用完全相同的请求集合、context 内容、input/output length distribution、cache capacity、I/O backend、模型和硬件。不同 locality 条件只改变请求访问顺序或等价的 reuse-distance 结构。

除非具体实验明确声明，scheduler policy、cache hierarchy 和 serving configuration 均保持固定。

## 2. Locality 定义

本实验组使用三类基础 locality condition。

- `min-distance` 使相同或相关 context 尽可能连续出现，用于构造高 locality 条件。
- `shuffle` 对同一请求集合进行固定随机化，用于构造中等、无特殊优化的访问顺序。
- `max-distance` 使同一 context 的 revisit 尽可能分散，用于构造低 locality 条件。

三类条件必须保持相同的理论 reuse opportunity。实验不能通过改变 prefix group 数量、context 内容或请求数量来伪造 locality 差异。

## 3. Arrival-rate 定义

Arrival rate 使用相对于当前模型、硬件和 serving configuration 的系统容量定义，而不是直接把同一绝对 request rate 套用于所有平台。

正式 sweep 前先进行独立 calibration，确定系统从明显未饱和、接近稳定上限到持续排队的工作区间。

实验至少区分 `low`、`medium`、`high` 与 `overload` 四种负载状态。`overload` 主要用于识别 saturation 与 scheduler pathology 的边界，不直接作为 scheduler 正收益的核心证据。

## 4. Baseline 与 optimized scheduler

Experiment 1 只使用 baseline scheduler，以建立未优化时的 workload-response surface。

Experiments 2–4 才引入 delay-hit mitigation、balanced batching、bubble filling / stall hiding 和 full scheduler。

任何 optimized scheduler 的收益都必须相对于同一 workload trace、同一 offered load 和同一 cache/I/O 条件下的 baseline 计算。

## 5. Workload trace

每条 workload trace 保存稳定 identifier、配置和随机 seed。

Trace metadata 至少记录：

- request count；
- context/prefix group identifier；
- request ordering；
- reuse-distance summary；
- input length distribution；
- output length distribution；
- locality condition；
- offered-load condition；
- seed / configuration hash。

## 6. Core metrics

本实验组统一记录以下指标：

- realized cache hit / reuse；
- delay hit；
- redundant prefill；
- queueing delay；
- I/O stall；
- TTFT distribution；
- achieved throughput。

可以获得时同时记录 request-level 与 token/state-volume-weighted cache statistics。

## 7. 重复测量与有效性

每个正式配置执行多次独立重复测量。

不同配置的执行顺序交替或随机化，避免机器长期状态变化系统性偏向某一 workload。

正式统计使用稳定 serving 区间。初始化、模型加载和一次性 runtime warm-up 不进入主性能统计。

出现 OOM、runtime fallback、workload trace mismatch、配置漂移或无法解释的 measurement failure 时，run 保留在 raw results 中并标记为 invalid，不进入主 aggregation。

## 8. 解释边界

本实验组只研究 request ordering、cache locality、arrival pressure 与 scheduler mechanism 的关系。

Cache capacity、page granularity、GPU-assisted I/O 和 hierarchical cache value 由其他实验组独立研究。本实验组不通过同时改变这些变量来制造 scheduler 收益。

跨模型与跨硬件的完整 generalization 由项目级 Model and Hardware Generalization 实验组处理。本组可以保存必要的 representative results 供后续复用。