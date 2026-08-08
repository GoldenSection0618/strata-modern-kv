# Code

本目录用于存放“Hierarchical Cache Value Evaluation”相关实验实现。

代码应覆盖以下职责：

- shared-prefix workload 构造与请求轨迹生成；
- GPU-only 与 GPU + CPU hierarchical cache 两种运行模式；
- cold-cache 与 warm-cache 初始状态建立和验证；
- GPU cache hit、CPU cache hit、recomputation、CPU-GPU traffic、TTFT 与 throughput 的采集；
- GPU cache pressure 与 prefix reuse 后续 sweep 的实验入口；
- raw results 到 processed results 的确定性处理；
- cross-model validation 的结果汇总与绘图。

## Implementation rules

- GPU-only 与 hierarchical cache 必须使用相同 GPU cache budget。
- Cold-cache 与 warm-cache 的初始状态必须能够显式建立并验证。
- CPU cache hit 不能仅根据配置推断，必须由 runtime 可观测行为确认。
- Recomputation 与 CPU restore 必须能够在结果中区分，不能只记录最终 TTFT。
- CPU-GPU traffic 必须与 cache hit 和 recomputation reduction 对齐分析。
- model revision、runtime version/commit、hardware、precision、cache policy、GPU cache budget、workload 参数和 repetition index 必须写入每次 run 的 metadata。
- 分析和绘图脚本只读取 raw/processed data，不把最终实验数字手工写死在代码中。

建议后续实现按 `configs/`、`workloads/`、`runners/`、`profiling/`、`analysis/` 等职责拆分，实际目录以代码规模为准，不为了形式预建空目录。
