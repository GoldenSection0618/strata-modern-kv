# Code

本目录用于存放 Page Granularity and GPU-Assisted I/O 实验实现。

代码应覆盖以下职责：

- workload 构造与固定请求 trace 管理；
- page-size sweep 配置与运行入口；
- cache hit、effective reused tokens、cache utilization efficiency 等指标采集；
- 后续 host→GPU I/O bandwidth 与 fragmented transfer 测量；
- 普通 I/O 与 GPU-assisted I/O 的统一实验入口；
- GPU computation interference 与 end-to-end serving 指标采集；
- raw results 到 processed results 的确定性处理与绘图。

## Implementation rules

- 同一实验组的不同 page size 必须复用完全相同的请求 trace、随机种子和非 page-size 配置。
- page-level hit 与 token-level effective reuse 必须分别记录，不能用 page hit rate 替代实际复用收益。
- cache capacity、replacement policy、scheduler 和 I/O backend 必须写入每次 run 的 metadata。
- invalid run 必须保留状态与原因，不通过删除异常结果制造整洁数据。
- 分析和绘图代码只读取 raw / processed data，不把最终数字手工写死在脚本中。

后续实现可按 `configs/`、`runners/`、`profiling/`、`analysis/` 等职责拆分，实际目录以代码规模为准，不为了形式预建空目录。
