# Common Conventions

本文件定义 “Cache Locality and Scheduler Behavior” 实验组共享的 workload、runtime capability 与 measurement 约定。

## 1. 变量隔离原则

本实验组的两个基础 workload 变量是 request arrival pressure 与 cache distance / reuse distance。

同一组 paired comparison 使用完全相同的 request set、context 内容、input/output length distribution、cache capacity、I/O backend、模型和硬件。不同 cache-distance 条件只改变 request ordering 或等价的 revisit timing structure。

除非具体实验明确声明，cache hierarchy、I/O backend、batch/token limit 和 serving configuration 均保持固定。Scheduler policy 只在 Experiments 2–4 中按设计改变。

## 2. Cache distance 定义

本实验组使用三类基础 cache-distance condition。

- `min-distance`：相同 context / shared-prefix group 的请求尽可能连续，cache distance 最小，locality 最高。
- `shuffle`：对同一 request set 使用固定 seed 随机化，作为普通无序 workload。
- `max-distance`：同一 context 的 revisit 尽可能均匀分散，cache distance 最大，locality 最低。

三类 workload 必须保持相同的 request set 与 theoretical reuse opportunity。实验不能通过减少 prefix group、集中热点 context、修改 prefix 长度或改变 request count 来伪造 locality 差异。

每条 trace 同时保存 configured condition 与实际 reuse-distance distribution。最终分析以 observed distance 为准，而不是只依据标签。

### Interpretation rule

Cache distance 对不同 pathology 的影响不要求单调同向。

Strata 原论文的 reference behavior 是：minimum cache distance 下相同 context 请求聚集，更容易出现 delay hit；maximum cache distance 降低 delay-hit likelihood，但更多 reusable cache 可能需要从 CPU DRAM 恢复，因此 host-loading / I/O pressure 更明显。

该方向只作为 historical hypothesis。现代 runtime 的结论必须由当前实验测量决定。

## 3. Arrival-rate 定义

Arrival rate 使用相对于当前模型、硬件和固定 serving configuration 的系统容量定义，而不是把相同绝对 request rate 套用于所有平台。

正式 sweep 前先进行 capacity calibration，确定系统从明显未饱和、稳定并发、接近稳定 throughput limit 到持续 backlog 的工作区间。

实验至少区分：

- `low`：明显未饱和；
- `medium`：稳定并发且仍有余量；
- `high`：接近稳定吞吐上限但尚未持续 overload；
- `overload`：offered load 超过稳定处理能力并形成持续 backlog。

`overload` 主要用于识别 saturation boundary，不作为 scheduler 正收益的核心证据。

## 4. Delay-hit 术语

Delay hit 指多个请求引用同一 context，而第一个相关 cache miss 尚未被 resolve，导致后续请求在 cache 真正 ready 前到达的情况。

本项目记录：

- `cache resolve time`：从首次未就绪 context miss 被接受，到对应 context 对后续请求可被正确复用为止的实际时间；
- `same-context fan-in`：同一 resolve window 内引用相同 context 的请求数量；
- `delay-hit affected volume`：受 delay hit 影响的 request/token/state volume；
- `redundant work`：本可等待并复用但实际被重复执行的 prefill / recomputation。

Delay hit 不要求该 context 已经存在于 CPU tier。Cold miss 在计算尚未完成时同样可以产生 delay hit。

## 5. Cache residency / resolve modes

需要时使用三种清晰区分的状态：

- `cold-miss`：目标 reusable context 在本轮开始前尚未 materialize。第一个请求触发计算/建立，后续请求可能在 resolve window 内形成 delay hit。
- `cpu-restore`：目标 context 已存在于经过验证的 CPU tier，但尚未 GPU-ready，需要 restore。该模式用于研究 loading-related coordination，不能替代 cold-miss delay-hit definition。
- `gpu-ready`：目标 state 在请求到达前已可直接复用，用作并发本身的 control。

任何 `cpu-restore` 结果都必须通过对应模型的 full-state hierarchy validation。Partial restore 只能标记为 `partial`，不能并入 full-hierarchy scheduler claim。

## 6. Scheduler configurations

Experiment 1 只使用 baseline scheduler。

Experiments 2–4 根据 capability gate 使用以下机制语义：

1. baseline；
2. delay-hit deferral / mitigation；
3. balanced batch formation；
4. bubble filling / stall hiding；
5. full scheduler，由前三个 mechanism 组成。

在 progressive ablation 中，实际主序列为：

```text
S0 Baseline
→ S1 + Delay-hit mitigation
→ S2 + Balanced batching
→ S3 + Stall hiding = Full scheduler
```

因此 full scheduler 不是 progressive sequence 中额外的第五个独立阶段。

每个 mechanism 的实现必须通过 semantic validation。若当前 runtime 无法独立启用、禁用或观测某机制，则将该机制标记为 `unsupported`，不使用名字相似但语义不同的 policy 静默替代。

## 7. Balanced-batch 与 stall-hiding measurement

Balanced batching 至少保存：

- estimated/observed load amount；
- compute amount；
- batch load / compute ratio；
- loading-bound batch status；
- bundle-hit count / volume；
- non-overlapped I/O stall。

不直接复用 Strata 论文中的 load/compute threshold 数值。该 threshold 是 model/hardware dependent，必须在当前平台上单独 calibration 或按 implementation semantics 固定并记录。

Bubble filling 至少保存：

- residual loading stall；
- GPU idle interval；
- filled-bubble time；
- inserted work type；
- inserted useful computation；
- decode-side latency / TPOT。

## 8. Workload trace

每条 workload trace 保存稳定 identifier、配置和随机 seed。

Trace metadata 至少记录：

- request count；
- context/prefix group identifier；
- request ordering / arrival timestamps；
- configured and observed reuse-distance summary；
- input/output length distribution；
- theoretical reusable volume；
- locality/cache-distance condition；
- offered-load condition；
- seed / configuration hash。

Experiment 3 额外记录 target context、resolve mode、cache resolve time、same-context fan-in 与 overlap window。

## 9. Core metrics

本实验组统一记录：

- realized cache reuse / reuse realization；
- delay hit；
- redundant prefill / recomputation；
- host restore volume 与 duplicate restore when applicable；
- queueing delay；
- non-overlapped I/O stall；
- batch load / compute ratio 与 bundle hit when applicable；
- GPU idle / filled-bubble time when applicable；
- TTFT distribution；
- TPOT 或等价 decode latency；
- achieved throughput。

可以获得时同时记录 request-level 与 token/state-volume-weighted cache statistics。

## 10. 重复测量与有效性

每个正式配置执行多次独立重复测量。

不同配置的执行顺序交替或随机化，避免机器长期状态变化系统性偏向某一 workload。

正式统计使用稳定 serving 区间。初始化、模型加载和一次性 runtime warm-up 不进入主性能统计。

出现 OOM、runtime fallback、workload trace mismatch、配置漂移、hierarchy validation failure、scheduler semantic mismatch 或无法解释的 measurement failure 时，run 保留在 raw results 中并标记为 invalid / partial / unsupported，不进入不适用的主 aggregation。

## 11. 解释边界

本实验组只研究 request ordering、cache distance、arrival pressure、same-context overlap 与 scheduler mechanism 的关系。

Cache capacity、page granularity、GPU-assisted I/O 和 hierarchical-cache value 由其他实验组独立研究。本组只能选择一个已验证且固定的 cache/I/O configuration 作为 scheduler 实验底座，不能在 scheduler comparison 中同时改变 data path。

跨模型与跨硬件的完整 generalization 由项目级 Model and Hardware Generalization 组处理。本组保存 representative points 与绝对 measurement，供后续直接复用。
