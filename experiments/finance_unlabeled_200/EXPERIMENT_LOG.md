# 金融 Query 分类分层优化实验记录

## 实验目标

在无标注数据上优化金融 Query 分类 Prompt。实验不计算或宣称准确率，只比较：

- 确定性格式与 taxonomy 合法性；
- 固定边界集合上的独立 LLM Judge 表现；
- 动态高风险集合上的独立 LLM Judge 表现；
- 风险集合复跑稳定性；
- 指标完全相同时的 Prompt 长度与任务 token。

## 冻结输入

- 数据：`artifacts/datasets/finance_unlabeled_200/finance_optimize_unlabeled_200.jsonl`
- 数据规模：200
- 随机真实样本：150，`sample_kind=random_pool`
- 固定边界样本：50，`sample_kind=boundary_probe`
- held-out：400 条，仅用于未来终检，不参与本次优化
- Taxonomy：用户提供的 `config/taxonomy.json`
- 初版 Prompt：用户提供的 `system_prompt.md`，运行时逐字读取
- SignalSpec：`artifacts/experiments/finance_unlabeled_200/signals.json`，状态 `approved`

冻结文件 SHA-256：

- 数据集：`c310c75628649abd2b760381b45622d05e6bfd9f271741f8382291aa9bca80e2`
- Taxonomy：`de58b337b5e58369d7d75d239be29878b72dbd767b2a37c712d31e3d5d7ea96f`
- 初版 Prompt：`12c81cdf761564a64f1b6283db155e54952771bea1cb0e20f96ea92cf25e840a`
- SignalSpec：`5eef609d693ba7afc957942a588a34a3bc3f07663935fd2884e26ff414f754e7`

## 模型与漏斗预算

- Task model：`deepseek-chat`
- Candidate generator：`deepseek-chat`
- Judge：`deepseek-reasoner`
- 候选 Prompt 上限：2（加初版 Prompt 共最多 3 个）
- 固定 Judge 集合：全部 50 条 boundary probe
- 动态 Judge 集合：非纯确定性失败的风险 Top 30
- 每个 Prompt 的 Judge 案例上限：80
- Judge 总调用上限：1200
- Task model 总调用上限：10000
- 自动优化轮数上限：5
- 连续无实质提升停止：2 轮
- 最小实质提升阈值：0.01
- 历史风险记忆：开启
- 人工 Review 导出上限：20

## 决选规则

采用固定字典序：

1. 硬失败更少；
2. boundary Judge 均分更高；
3. dynamic-risk Judge 均分更高；
4. 风险集合复跑稳定性更高；
5. Prompt 更短；
6. Task token 更少。

## 审计与恢复

正式运行目录为 `artifacts/runs/finance_layered/<run_id>/`。阶段文件包括：

- `run_identity.json`
- `prompts.json`
- `task_outputs.jsonl`
- `deterministic_checks.jsonl`
- `risk_scores.jsonl`
- `judge_queue.json`
- `judge_results.jsonl`
- `stability_outputs.jsonl`
- `prompt_comparison.json`
- `human_review_top20.csv`
- `run.json`

同一 `run_id` 再次运行时，只补充尚未完成的模型调用。缓存响应不代表 run 已完成，
只有写出最终 `run.json` 才表示正式完成。

## 执行记录

### 2026-07-30：运行前检查

- 确认样本总数 200，构成为 150 random + 50 boundary。
- 确认 query 无重复，且与 held-out 集合无 request-id 重叠。
- 确认 SignalSpec 已批准。
- 确认初版 Prompt 要求严格 JSON，并包含 L2/L3、名称、dim_tag、confidence、reason。
- 确认固定边界数 50 未超过每 Prompt Judge 上限 80。
- 创建分层任务配置 `layered.task.json`。
- 当前进程未发现 `OPENAI_API_KEY`，且钥匙串未找到 `promptos-deepseek` 项；正式模型调用尚未启动。

### 2026-07-30：单轮运行暴露候选生成问题

- DeepSeek 首次返回 Markdown 代码块包裹的 JSON 数组，旧生成器未识别。
- 生成器现兼容 JSON 对象、数组和 Markdown 代码块。
- 第二次候选仅改变 CRLF/LF；新增语义去重后被正确拒绝。
- 增加透明降级候选：路由冲突检查、歧义与输出自检。
- 异常运行均已中止；缓存与阶段输出不视为正式完成。

### 2026-07-30：升级为无人介入自动闭环

- 实验模式由 `layered` 调整为 `layered_auto`。
- 每轮自动生成 `failure_summary.json` 并反馈给下一轮候选生成器。
- 历史 Judge 风险集合自动进入后续轮次回归。
- 中间不要求人工标注；人工 Review 仅在最终冠军产生后导出。

### 2026-07-30：正式自动实验完成

- 正式 run ID：`finance_auto_20260730_v1`
- 完成轮数：5
- 停止原因：`no_material_improvement`
- 结果状态：`provisional_silver_or_unlabeled`
- Task model 调用记录：3756
- Judge 调用记录：280
- Candidate generator 调用记录：5
- 风险记忆案例：51
- 最终冠军长度：4678 字符
- 人工最终验收队列：20 条

逐轮结果：

| 轮次 | 是否更新冠军 | 当轮冠军硬失败 | Boundary Judge | Dynamic Judge | 稳定性 |
|---|---:|---:|---:|---:|---:|
| 0 | 是 | 196 | 1.0000 | 0.0000 | 1.0000 |
| 1 | 是 | 179 | 0.7709 | 0.0000 | 1.0000 |
| 2 | 是 | 134 | 0.9297 | 0.0000 | 1.0000 |
| 3 | 否 | 134 | 0.8980 | 0.8200 | 1.0000 |
| 4 | 否 | 134 | 0.8751 | 0.8650 | 1.0000 |

关键发现：

- 初版 Prompt 的 200/200 个输出都因 Markdown JSON 代码块而触发严格格式硬失败。
- 自动循环将冠军硬失败从 200 降到 134，但仍有 67% 样本未满足“独立 JSON、
  无额外文本”的严格契约。
- 第 2 轮冠军同时取得当轮边界 Judge 0.9297 和稳定性 1.0，随后两轮没有候选能在
  更低硬失败数量这一最高优先级上击败它。
- 最终 Review Top 20 中 7 条属于边界低分/低置信优先层，13 条属于 Prompt 标签翻转
  优先层；20 条均至少在一个 Prompt 上存在确定性硬失败。
- 本实验没有使用中间人工标注，自动完成了候选生成、评估、风险记忆、决选和停止。

结论：

- 自动闭环机制验证成功，但当前冠军不应直接作为生产 Prompt 验收通过。
- 主要瓶颈不是 taxonomy 语义规则，而是模型输出通道没有强制结构化 JSON。
- 下一步优先考虑在模型适配器中启用供应商支持的 JSON/structured-output 模式，
  再以同一无标注样本和风险记忆做复验；不建议继续只靠增加 Prompt 文本解决格式问题。
