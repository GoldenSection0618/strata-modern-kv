# Code

本目录用于存放“现代模型上的 KV / 状态瓶颈画像”相关实现。

代码应覆盖以下职责：

- workload 构造与 exact token-length 控制；
- cache-residency 初始化与验证；
- Experiment 1-3 的运行入口；
- cache/state footprint、CPU-GPU transfer、queueing、TTFT 与 throughput 的采集；
- runtime validation checks；
- raw results 到 processed results 的确定性处理；
- Experiment 4 的 cross-model 汇总与绘图。

实现必须遵循 `../docs/00-measurement-conventions.md` 的统一口径。

## Implementation rules

- 不把理论 state size 当作实际 runtime footprint 的替代值。
- 不把 raw transfer duration 与 overlapped computation 直接相加形成 TTFT decomposition。
- CPU-resident hit 必须通过 runtime counter/trace 验证后才能标记为有效 cache hit。
- model revision、runtime commit/version、precision、cache policy 和 workload 参数必须写入每次 run 的 metadata。
- 分析和绘图脚本只读取 raw/processed data，不把最终数字手工写死在代码中。

建议后续实现按 `configs/`、`runners/`、`profiling/`、`analysis/` 等职责拆分，实际目录以代码规模为准，不为了形式预建空目录。
