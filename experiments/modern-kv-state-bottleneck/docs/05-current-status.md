# Experiment Suite Current Status

> 状态快照：2026-08-09 17:00 GMT+8（Asia/Shanghai）。基于服务器真实文件核对。
> 设计文档见 `docs/00`–`docs/04`；Exp4 实现细节见 `code/EXP4_IMPLEMENTATION.md`。

## 0. 一句话总结

**只有 recompute 模式的测量数据是真实可用的（exp1 4 点 + exp2 5 点）；gpu_hit / cpu_hit 全部因 stats 采集问题验证门 FAILED、无测量数据；exp3 只有一个失败的冒烟测试；exp4 分析脚本已就绪并跑通，但输入数据残缺。**

---

## 1. 代码与 Git 状态

分支 `feat/exp1-implementation`，未 push。

| 提交 | 说明 |
|---|---|
| `24b96a1` | exp1/exp2 可在 A100 + vLLM 0.26 跑通（flashinfer/spawn/max_num_seqs 修复） |
| `ff60752` | 强制 spawn 避免 CUDA fork 崩溃 |
| `d6e5060` | exp3 代码 + docs |
| `3357adb` | **Exp4**：synthesis + figures |
| `5e449eb` | **Exp4**：实现笔记文档 |

**工作区有 7 个未提交改动（另一个会话修改，非本会话）**：

| 文件 | 改动内容 |
|---|---|
| `profiling/vllm_stats.py` | **重写**：patch `engine_core.get_output()` 捕获 SchedulerStats（替代找不到 KVCacheManager 的旧方案）；prefix 统计按 drain 语义客户端累加 |
| `runners/vllm_runner.py` | 引擎加 `disable_log_stats=False`（vLLM 0.26 默认 True，stats 为 None）；warmup 日志改累计值 |
| `validate.py` | hit 检查改 before/after delta 逻辑；cpu_hit 增加 eviction events 判据 |
| `profiling/load_driver.py` | ThreadPoolExecutor 按 concurrency_ceiling 建专用池（修复默认池封顶）；active_concurrency_mean 改用 Little's law |
| `run_exp3.py` | control mode 支持 FROZEN_RATES（按位置选 low/sat/overload） |
| `run_exp3.sbatch` / `submit_exp3.sh` | FROZEN_RATES 参数全链路贯通 |

⚠️ **这些改动尚未提交，也未经新代码实际运行验证**（冒烟测试 1269314 跑的是旧代码，日志仍为旧格式）。

---

## 2. Exp1：Context Length Scaling（qwen）

设计：4096/8192/16384/32768 × recompute/gpu_hit/cpu_hit × 10 reps，prefix 固定 50%。

| context | recompute | gpu_hit | cpu_hit |
|---|---|---|---|
| 4096 | ✅ summary + 10 raw | ❌ validation FAILED | ❌ validation FAILED |
| 8192 | ✅ summary + 10 raw | ❌ validation FAILED | ❌ validation FAILED |
| 16384 | ✅ summary + 10 raw | ❌ 无输出（任务 FAILED） | ❌ 无输出（任务 FAILED） |
| 32768 | ✅ summary + 10 raw | ❌ 无输出（任务 FAILED） | ❌ 无输出（任务 FAILED） |

**可用数据**：4 个 recompute summary（median TTFT 282.7 / 564.7 / 1175.6 / 2609.9 ms）+ 40 个 raw rep。

**问题**：gpu_hit/cpu_hit 的 validation `gpu_resident_hit` / `cpu_resident_hit` 恒 FAILED（`queries=0, hits=0`），测量被中止 → 无 summary、无 raw。4096/8192 的目录里只有 metadata + validation.json；16384/32768 任务本身也 FAILED（CUDA fork 问题，已被 ff60752 修复）。

⚠️ **注意**：之前 sacct 显示的 "COMPLETED"（如 1269283/1269284）只是 python 退出码 0——验证门失败后测量中止，**并没有产出数据**。所有 gpu_hit/cpu_hit 数据点实际全部缺失。

**root cause（另一个会话已定位）**：vLLM 0.26 V1 引擎 `disable_log_stats=True` 默认 + `KVCacheManager` 内部对象在 EngineCore 子进程、客户端访问不到 → stats 全空。修复方案已在工作区（见 §1），未验证。

---

## 3. Exp2：Shared-Prefix Scaling（qwen）

设计：32768 ctx × 0/25/50/75/87.5% prefix × recompute/gpu_hit/cpu_hit（0% 只跑 recompute）。

| prefix ratio | recompute | gpu_hit | cpu_hit |
|---|---|---|---|
| 0% | ✅ summary + 10 raw | —（设计跳过） | —（设计跳过） |
| 25% | ✅ summary + 10 raw | ❌ 任务 FAILED | ❌ validation FAILED |
| 50% | ✅ summary + 10 raw | ❌ 任务 FAILED | ❌ validation FAILED |
| 75% | ✅ summary + 10 raw | ❌ 任务 FAILED | ❌ validation FAILED |
| 87.5% | ✅ summary + 10 raw | ❌ validation FAILED | ❌ validation FAILED |

**可用数据**：5 个 recompute summary + 50 个 raw rep。

**问题**：与 Exp1 相同——所有 gpu_hit/cpu_hit 无测量数据（验证门 FAILED 或任务崩溃）。

---

## 4. Exp3：Request-Rate Scaling（qwen）

**状态**：只有一次冒烟测试 `results/exp3/qwen/8192-cpu_hit/`（任务 1269314，COMPLETED 退出码 0），validation FAILED、无 summary、无 raw。**正式实验未跑，当前提交任务也跑不出有效数据（见下方前置条件）。**

### 当前可跑性（2026-08-09 核对）

| 模式 | 可跑性 | 说明 |
|---|---|---|
| recompute（control） | ✅ 能出有效数据 | 跳过 cache-hit 验证门，不依赖 stats（exp1/2 已证实） |
| gpu_hit（control） | ❌ 大概率无数据 | 依赖验证门通过；新 stats 修复未实际验证 |
| cpu_hit（primary） | ❌ 必然无数据 | 三个障碍叠加，见下 |

**cpu_hit 的三个障碍（按顺序）**：

1. **验证门未验证**：冒烟测试 1269314 跑的是旧代码（日志旧格式 `Warmup: no prefix cache hit detected`）；新修复（patch `get_output()` + `disable_log_stats=False`）还没跑过一次。直接提交大概率重复冒烟结果，浪费 GPU 任务（sbatch 限时 2h）。
2. **KV offload 未激活（确定）**：`run_exp3.sbatch` 中 `KV_OFFLOAD="${KV_OFFLOAD:-0}"` 默认 0 → 不传 `--kv-offloading-size` → vLLM `vllm/v1/kv_offload/` 根本没启用。无 offload 则 cpu_hit 的 warmup→evict→restore 链路第一步就断。
3. **eviction filler 填不满 KV cache**：`evict_prefix_to_cpu` 只发 8×4096=32K tokens，而冒烟日志显示 KV cache 容量 **471,258 tokens** → 不会触发 eviction（`preempted_reqs=0` 已证实）。此问题在 `vllm_runner.py`，exp1/2/3 的 cpu_hit 全受影响。

**跑 exp3 的前置顺序**（避免浪费 GPU）：

1. 新代码（7 文件未提交改动）先跑一次 **gpu_hit 单点冒烟**（最小配置），确认验证门通过；
2. 以 `KV_OFFLOAD=<GB>`（如 8192 MB 量级）提交 cpu_hit，确认 eviction 真的触发（`preempted_hits` / `kv_eviction_events` 非零），必要时加大/修正 filler 逻辑；
3. 之后才全量：`bash submit_exp3.sh qwen primary` → capacity calibration → 再用 `FROZEN_RATES` 跑 recompute/gpu_hit control（工作区代码已支持）。

**当前 sbatch 已具备**：`KV_OFFLOAD` 环境变量贯通（`--kv-offloading-size`）、`FROZEN_RATES` 贯通（`--skip-calibration --frozen-rates`）、`CONTROL=1`（`--control-mode`）、flashinfer/spawn 修复已内置。

---

## 5. Exp4：Cross-Model Synthesis（分析脚本）

**代码状态**：✅ 已实现并提交（`3357adb` + `5e449eb`），纯标准库 synthesis + matplotlib figures（未装 matplotlib）。

**已用真实数据验证**：synthesis 正确解析 exp1 4 点 + exp2 5 点，正式输出已生成到 `results/exp4/processed/`：

| 输出 | 行数 | 说明 |
|---|---|---|
| `context_scaling.csv` | 4 | 只有 recompute（hit 数据缺失） |
| `reuse_benefit.csv` | 5 | 只有 recompute 列，benefit/speedup 全空 |
| `load_sensitivity.csv` | 0 | exp3 未跑 |
| `bottleneck_composition.csv` | 3 | 只有 recompute TTFT，`parts_available=False` |
| `synthesis_report.json` | — | 数据可用性记录 |

**当前能力**：只能出 recompute 单模式的 TTFT 曲线；跨模型对比缺 Gemma 4 数据。

---

## 6. 总体阻塞与后续路线

### 核心阻塞（一个，影响 exp1/2/3 的 hit 模式）
> **vLLM 0.26 stats 采集不可用** → gpu_hit/cpu_hit 验证门恒 FAILED → 无任何 hit 测量数据。

根因链：`disable_log_stats=True`（默认）→ SchedulerStats 不发；`KVCacheManager` 在 EngineCore 子进程、客户端路径找不到 → prefix stats 恒 0。

工作区已有修复（另一个会话）：`disable_log_stats=False` + patch `get_output()` 捕获 SchedulerStats + validate delta 逻辑。**下一步：跑一次新代码的 gpu_hit 冒烟，确认验证门通过。**

### 次生问题
- eviction filler 量不足：`evict_prefix_to_cpu` 只发 8×4096=32K tokens，而 KV cache 有 471K tokens，填不满、不会触发 eviction（`preempted_reqs=0` 证实）。且 `kv_offloading_size` 默认 0 → KV offload 未激活（另一个会话已确认 vllm/v1/kv_offload 需要 KVTransferConfig 激活）。
- `active_concurrency_mean` 旧公式量纲错误（已修，未验证）。

### 数据补齐顺序建议
1. 提交并冒烟验证另一个会话的 7 个文件改动（gpu_hit 单点）
2. 重跑 exp1 gpu_hit/cpu_hit 4 点（16384/32768 需要确认 eviction 修复）
3. 重跑 exp2 gpu_hit/cpu_hit 4 点
4. 跑 exp3（先 calibration 后 control）
5. 跑 Gemma 4（exp1/2/3 同构）
6. 重跑 `exp4_synthesis.py` → 自动填充全部跨模型指标 → 装 matplotlib 出图

---

## 7. 数据文件清单（服务器路径）

- 根目录：`/share01/hpc/humxlab_intern/yanglihan/dl-stack/projects/strata-modern-kv/experiments/modern-kv-state-bottleneck/`
- 结果：`results/exp1|exp2|exp3|exp4/`
- 日志：`~/yanglihan/logs/ylh-*.out/.err`
- 模型：`/share01/hpc/humxlab_intern/yanglihan/dl-stack/cache/huggingface/models--Qwen--Qwen3.5-9B`
- 环境：`~/yanglihan/dl-stack/envs/qwen/`（vLLM 0.26.0，无 matplotlib/pandas）
