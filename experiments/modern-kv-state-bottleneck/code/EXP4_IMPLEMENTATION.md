# Experiment 4 Implementation Notes

> 实现状态记录（2026-08-09）。设计依据见 [`../docs/04-cross-model-bottleneck-comparison.md`](../docs/04-cross-model-bottleneck-comparison.md)。
> 本文件只覆盖 Exp4 的实现，不覆盖另一个会话对 exp3/vllm_stats 的未提交改动。

## 1. 定位

Exp4 是 **cross-model synthesis**：不产生新测量，复用 Exp1/2/3 的 `summary.json`，产出跨模型 processed CSV 与图表。回答"KV/state bottleneck 在 Qwen3.5-9B 与 Gemma 4 12B 上是否都存在、严重程度与触发条件是否稳定"。

## 2. 交付文件

| 文件 | 说明 |
|---|---|
| `analysis/exp4_synthesis.py` | 主分析脚本，纯标准库（json/csv/statistics），无第三方依赖 |
| `analysis/exp4_figures.py` | 4 组跨模型图，matplotlib 惰性导入，缺数据时跳过对应图不崩 |

提交：`3357adb`（分支 `feat/exp1-implementation`，未 push）。

## 3. 输出（synthesis）

`--output-dir` 下生成 5 个文件：

| 文件 | 内容 |
|---|---|
| `context_scaling.csv` | model × context_length × residency_mode 的 TTFT 统计（来自 Exp1） |
| `reuse_benefit.csv` | model × prefix_ratio 的 reuse 指标（来自 Exp2） |
| `load_sensitivity.csv` | model × normalized_load 负载曲线（来自 Exp3，未跑时为空表但结构完整） |
| `bottleneck_composition.csv` | 3 个 matched points 的 TTFT 分解 |
| `synthesis_report.json` | 数据可用性 + 结论判定指南（A/B/C/D 四种情形） |

**reuse_benefit.csv 指标**（口径见设计文档 §8）：
- `net_reuse_benefit_ms` = recompute TTFT − cpu_hit TTFT
- `reuse_speedup` = recompute / cpu_hit
- `gpu_reuse_benefit_ms` = recompute − gpu_hit（无 restore 成本的上界）
- `cpu_restore_penalty_ms` = cpu_hit − gpu_hit
- `service_stall_ratio` = (cpu_hit − gpu_hit) / cpu_hit
- `reuse_realization_ratio` = net benefit / gpu_reuse_benefit；**分母 < 1ms 时不报告**（避免不稳定放大）

**matched points**（设计文档 §4）：short-light (8K, 50%) / long-light (32K, 50%) / long-high-reuse (32K, 75%)。TTFT 分解中 compute path 用 gpu_hit TTFT、I/O stall 用 cpu−gpu；无法分离的分量标 `None`，不伪造数值。

## 4. 关键设计决策（容易踩坑的点）

1. **模型标识归一化**：Exp1 summary 的 `model="qwen"`，Exp2 的是 `"Qwen/Qwen3.5-9B"` → 统一映射为 `qwen` / `gemma4`（`normalize_model()`）。
2. **Exp1 的 summary 没有 `prefix_ratio` 字段**（设计固定 50%）→ `discover_summaries(..., experiment="exp1")` 时补默认值 0.5，否则 Exp1 数据会被 `ratio is None` 检查全部丢弃。
3. **目录结构不同**：Exp1 `{model}/{ctx}/{mode}/{mode}/summary.json`（mode 重复两次）；Exp2 `{model}/{ctx}-{pct}pct-{mode}/{mode}/summary.json` → 分开解析，优先 JSON 字段、回退目录名。
4. **Exp3 summary.json 是 list**（每 load point 一条聚合记录），与 Exp1/2 的 dict 不同 → 单独 `discover_exp3()`，从目录名解析 model/ctx/ratio/mode（`32k-50pct-cpu_hit` 风格）。
5. **缺失数据优雅处理**：gpu_hit/cpu_hit 无 summary（验证失败）时 CSV 留空、`data_complete=False`，不中断；可用性记录进 report。
6. **纯标准库**：服务器 qwen env 与本地均无 matplotlib/pandas，synthesis 只用 stdlib。

## 5. 验证状态

- 已用真实数据验证：正确解析 Exp1 4 个 summary（4096/8192/16384/32768 recompute）+ Exp2 5 个（0/25/50/75/87.5% recompute），输出 4 CSV + report。
- `bottleneck_composition.csv` 当前只含 recompute TTFT（hit 数据缺失），`parts_available=False`，属预期。

## 6. 已知限制

- **matplotlib 未安装**（服务器 qwen env 和本机都没有）→ figures 脚本需在有 matplotlib 的环境运行，当前会打印清晰报错退出。
- fig3（load sensitivity）依赖 Exp3 数据，Exp3 未跑前不出图。
- fig4（composition）需要 matched points 的 cpu_hit + gpu_hit 数据，当前缺失。
- 跨模型对比需要 Gemma 4 数据，当前只有 qwen。

## 7. 数据补齐后

无需改脚本：hit 模式、Gemma 4、Exp3 数据就绪后重跑 synthesis 即自动填充全部指标与图。

## 8. Usage

```bash
# synthesis（无第三方依赖）
python3 code/analysis/exp4_synthesis.py \
    --exp1-dir results/exp1/ \
    --exp2-dir results/exp2/ \
    --exp3-dir results/exp3/ \
    --output-dir results/exp4/processed/

# figures（需要 matplotlib）
python3 code/analysis/exp4_figures.py \
    --input-dir results/exp4/processed/ \
    --output-dir results/exp4/figures/
```
