# PromptOS

本地优先、可审计的通用 Prompt 自动化调优框架。它不假设任务是分类：输入可以是 `answer`、证据、表格行或任意 JSON；输出可以是文本、真假判断、结构化 JSON 或工具调用结果。

PromptOS 将产品经理用自然语言写下的验收标准转为可审核的 **Signal System**，在预算内搜索 Prompt 候选，并保存完整实验血缘。

## 质量与标注原则

- 人工确认的数据是 `gold_human`，可作为晋级与最终评测依据。
- LLM 对无标注数据生成的是 `silver_auto`，只用于早期筛选和优化，不可宣称为准确率。
- 无标注阶段可衡量“符合 Signal System 的程度”，不能宣称“质量/准确率提升”。
- SignalSpec 必须人工审核并显式批准，才能启动优化；每个 run 都冻结其版本、数据哈希与模型/评分细节。

## 快速开始（完全离线）

需要 Python 3.10 或更高版本。推荐先安装为可编辑包：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

安装后可以直接使用 `promptos`；从源码运行时也可以继续使用
`PYTHONPATH=src python3 -m promptos.cli`。

```bash
promptos draft-signals \
  --acceptance "必须准确转换输入并只返回结果" \
  --output artifacts/signals.json
promptos approve-signals --signals artifacts/signals.json
promptos optimize \
  --dataset examples/uppercase.jsonl --inputs text --expected-field expected \
  --signals artifacts/signals.json --prompt "处理输入" --runs artifacts/runs
```

`optimize` 是兼容模式，会对全量样本调用 Judge。单轮诊断可使用
`optimize-layered`；希望人工只做最终验收时，推荐使用 `optimize-auto`。

该示例使用本地演示模型。SDK 中也提供 `OpenAICompatibleModel` 与 `LLMJudge`：它们读取运行时环境变量 `OPENAI_API_KEY` 和可选的 `OPENAI_BASE_URL`，不把密钥写入任何文件。

可设置 `PROMPTOS_CACHE_DIR=/local/cache/path` 启用无密钥响应缓存。缓存键包含端点与完整请求体，不包含认证头；网络调用会对 429、5xx 和临时网络错误做指数退避重试。

## 推荐：分层风险优化

分层模式先让任务模型处理所有样本并执行不调用 LLM 的确定性检查，再把固定
`boundary_probe` 与动态风险 Top-K 合并为统一 Judge 集合。所有 Prompt 候选都在
同一集合上比较；格式、Schema 或 taxonomy 硬失败不会浪费 Judge 调用。

```bash
promptos optimize-layered \
  --dataset samples.jsonl --inputs query \
  --signals artifacts/signals.json \
  --plugin finance_classification --taxonomy /absolute/path/taxonomy.json \
  --model deepseek-chat --judge-model deepseek-reasoner \
  --max-candidates 2 --dynamic-top-k 30 \
  --judge-max-cases-per-prompt 80 --human-review-top-k 20 \
  --runs artifacts/runs
```

金融数据每行的 `metadata.sample_kind` 为 `boundary_probe` 时固定进入 Judge，
`metadata.boundary_note` 会连同风险原因、业务 metadata 和各 Prompt 的对比输出传给
Judge。默认策略是最多 2 个候选、动态 Top 30、每个 Prompt 最多 Judge 80 条、人工
Review 最多 20 条。若固定边界样本已经超过 80，程序会在任何模型调用前拒绝运行。

每个阶段会立即保存，指定同一 run ID 可以断点恢复：

```bash
promptos optimize-layered ... --resume-run layered_abc123
```

运行目录包含 `task_outputs.jsonl`、`deterministic_checks.jsonl`、
`risk_scores.jsonl`、`judge_queue.json`、`judge_results.jsonl`、
`prompt_comparison.json`、`human_review_top20.csv` 和 `run.json`。
`risk_scores.jsonl` 保留每条风险信号的来源与解释。冠军严格按“硬失败更少 →
边界 Judge 更好 → 动态风险 Judge 更好 → 稳定性更高 → Prompt/Token 更少”的
字典序产生，不使用隐藏加权分数。无金标实验仍标记为
`provisional_silver_or_unlabeled`。

## 无人介入的多轮自动优化

`optimize-auto` 将分层评估作为每轮评估引擎，自动执行：

```text
评估当前 Prompt → 汇总失败模式 → 生成候选 → 分层复评
→ 更新冠军 → 回归历史风险集合 → 判断停止 → 下一轮
```

中间不需要人工标注或批准候选。`failure_summary.json` 会把确定性失败、风险信号、
Judge 低分理由整理成下一轮反馈；已发现的高风险案例进入风险记忆，防止后续修复造成
回退。人工 Review CSV 只在整个自动循环结束后导出。

```bash
promptos optimize-auto \
  --dataset samples.jsonl --inputs query \
  --signals artifacts/signals.json --prompt-file initial_prompt.md \
  --plugin finance_classification --taxonomy taxonomy.json \
  --model deepseek-chat --judge-model deepseek-reasoner \
  --max-rounds 5 --stop-after-no-improvement 2 \
  --minimum-improvement 0.01 \
  --max-candidates 2 --dynamic-top-k 30 \
  --judge-max-cases-per-prompt 80 \
  --task-model-max-calls 10000 --judge-model-max-calls 1200 \
  --runs artifacts/runs
```

自动运行目录按 `round_00/`、`round_01/` 保存每轮完整证据，并在根目录输出
`champion_prompt.md`、`final_prompt_comparison.json`、
`human_review_top20.csv`、`auto_state.json` 和最终 `run.json`。指定
`--resume-run` 可从未完成轮次继续。

## 无标注数据与人工审核

```bash
PYTHONPATH=src python3 -m promptos.cli review-queue \
  --dataset my_answers.jsonl --inputs answer,evidence \
  --output artifacts/review-queue.json
```

v1 的队列会保守地保留无标注案例并明确其原因。生产 judge 可为高不确定性、模型分歧、格式失败和业务高风险案例补充排序；judge 输出必须记录模型、rubric、原始响应与 `silver_auto` 血缘。人工确认后，才可将样本升级为 `gold_human`。

人工审核结果以 JSONL 导入；每行含 `sample_id`、`value`、`source`、`rationale`。`silver_auto` 必须带 `judge_run_id`；`gold_human` 必须带非空 `reviewer`。这使标注来源始终可追溯。没有 `gold_human` 的运行会标记为 `provisional_silver_or_unlabeled`，不会被描述为准确率验证。

## 冻结金标切分

对已人工标注的数据，先切分并保存 manifest；`final_test` 不参与候选选择。

```bash
PYTHONPATH=src python3 -m promptos.cli split-dataset \
  --dataset labeled.jsonl --inputs answer,evidence --expected-field verdict \
  --seed 20260729 --output artifacts/split-v1
```

对无标注数据，以独立 judge 生成待审核 silver 标注（务必选用独立于任务模型的 `--model`）：

```bash
PYTHONPATH=src python3 -m promptos.cli annotate-silver \
  --dataset unlabeled.jsonl --inputs answer,evidence --signals artifacts/signals.json \
  --model judge-model --judge-run-id judge_20260729 --output artifacts/silver.jsonl
```

默认并发为 3，可通过 `--workers` 调整；遇到 429 时降低并发。单条失败会写入同目录的 `.errors.json`，不会丢失已成功完成的标注。

### 人工审核闭环

人工审核是唯一将 silver 变成 gold 的途径：

```bash
# 导出可编辑 CSV；可选 --annotations 附带 silver judge 的建议值和理由
PYTHONPATH=src python3 -m promptos.cli export-review \
  --dataset unlabeled.jsonl --inputs answer,evidence \
  --queue artifacts/review-queue.json --annotations artifacts/silver.jsonl \
  --output artifacts/review.csv

# 审核者填写 decision（approve/edit/reject）、reviewer、reviewed_value 与理由后导入
PYTHONPATH=src python3 -m promptos.cli import-review \
  --review-csv artifacts/review.csv --output artifacts/gold-annotations.jsonl

# 合并为带完整来源的规范数据集，再用于切分和优化
PYTHONPATH=src python3 -m promptos.cli apply-review \
  --dataset unlabeled.jsonl --inputs answer,evidence \
  --annotations artifacts/gold-annotations.jsonl --output artifacts/reviewed.jsonl
```

`approve` 采纳 silver 建议，`edit` 必须填写人工修订值，`reject` 永不进入 gold。导入时强制要求 reviewer、唯一 sample ID 和合法 decision；拒绝项另存为审计文件。

评选出 champion 后，才可对冻结的 final set 做一次性终检；相同 Prompt、SignalSpec 与 final set 组合不能重复运行。

```bash
PYTHONPATH=src python3 -m promptos.cli final-evaluate \
  --dataset artifacts/split-v1/final_test.jsonl \
  --split-manifest artifacts/split-v1/split_manifest.json \
  --inputs answer,evidence --signals artifacts/signals.json \
  --prompt "…已选中的 Prompt…" --runs artifacts/runs
```

## 核心概念

- `TaskSpec`：任意输入字段、输出说明和可选 JSON schema。
- `SignalSpec`：验收标准、硬约束、带权软信号及审核状态。
- `TaskModel` / `PromptGenerator` / `Judge`：独立可替换的模型与评估接口。
- `RunStore`：写入不可变的 `run.json`、候选逐样本结果和报告。

## 金融分类插件

原有金融分类项目现可作为独立插件使用，位于 `promptos.plugins.finance_classification`。它负责 taxonomy 加载、金融分类输出 schema、默认 Prompt 与金融专用 SignalSpec；通用优化核心仍不依赖 L2/L3 或 taxonomy。

```python
from pathlib import Path
from promptos.plugins.finance_classification import FinanceTaxonomy, default_prompt, signal_spec, task_spec

taxonomy = FinanceTaxonomy.load(Path("your-taxonomy.json"))
task = task_spec()
prompt = default_prompt(taxonomy)
signals = signal_spec(taxonomy).approved()
```

`your-taxonomy.json` 使用既有金融 taxonomy 结构：顶层 `L2_list`，每项含 `id`、`name` 和 `L3_list`。业务数据与 taxonomy 不随通用包分发。

## 可选真实网关验证

日常 [CI](.github/workflows/ci.yml) 只运行 mock 与离线测试。`.github/workflows/integration.yml` 仅在手动触发或工作日定时运行，并使用内部 GitHub 的 `PROMPTOS_INTEGRATION_API_KEY`、`PROMPTOS_INTEGRATION_BASE_URL` secrets 和 `PROMPTOS_INTEGRATION_MODEL` variable 发出一个极小请求。

它适用于任意 OpenAI 兼容网关（包括内部代理），不会在源码、日志或常规 CI 中保存密钥。

## 任务配置文件

用 `task.json` 将任务、数据、SignalSpec、模型和预算放进版本控制；它不得包含 API Key、token 或 secret。参考 [examples/uppercase.task.json](examples/uppercase.task.json)：

```bash
PYTHONPATH=src python3 -m promptos.cli validate-config --config examples/uppercase.task.json
PYTHONPATH=src python3 -m promptos.cli run-config --config examples/uppercase.task.json
```

路径均相对 `task.json` 本身解析。`run-config --prompt "…"` 或 `--model MODEL` 仅覆盖本次运行；其他配置保持冻结。真实网关地址可放在 `models.base_url`，鉴权仍只使用环境变量。

金融分层任务的核心配置如下；`taxonomy_path` 同样相对配置文件解析：

```json
{
  "plugin": {
    "name": "finance_classification",
    "taxonomy_path": "./taxonomy.json"
  },
  "evaluation": {
    "mode": "layered",
    "max_candidates": 2,
    "fixed_sample_kinds": ["boundary_probe"],
    "dynamic_top_k": 30,
    "judge_max_cases_per_prompt": 80,
    "human_review_top_k": 20
  }
}
```

配置为 `evaluation.mode: "layered"` 时，`run-config` 自动路由到单轮分层优化；
配置为 `layered_auto` 时进入无人介入的多轮闭环；`legacy` 或省略该字段时继续使用
旧流程。自动模式还支持：

```json
{
  "evaluation": {
    "mode": "layered_auto",
    "max_rounds": 5,
    "stop_after_no_improvement": 2,
    "minimum_improvement": 0.01,
    "retain_risk_memory": true
  }
}
```

## 开发

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## 项目状态与路线图

当前版本为 `0.1.0`（Alpha）。已包含 SignalSpec 编译与审批、Prompt
候选搜索、人工审核闭环、冻结数据切分、一次性终检、响应缓存、调用预算、
并发 silver 标注、实验血缘和金融分类插件。

后续重点：

- 主动学习队列的 judge 驱动排序
- 多评审一致性与分歧分析
- 调用前成本估算及更细粒度的预算策略
- 更多独立任务插件（事实核验、结构化抽取等）

项目尚未承诺稳定 API；升级次版本前请查看变更记录。欢迎通过 Issue 提交问题，
贡献方式见 [CONTRIBUTING.md](CONTRIBUTING.md)，安全问题请遵循
[SECURITY.md](SECURITY.md)。

## License

MIT
