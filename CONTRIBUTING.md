# Contributing to PromptOS

感谢你参与 PromptOS。提交改动前，请先搜索已有 Issue，较大的功能建议先开
Issue 对齐范围。

## 本地开发

需要 Python 3.10 或更高版本。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
ruff check src tests
coverage run -m unittest discover -s tests -v
coverage report
python -m build
```

真实 OpenAI 兼容网关测试默认跳过，不应成为普通 Pull Request 的前置条件。
不要在测试、Issue、日志或提交中加入 API Key。需要运行集成测试时，仅通过
环境变量配置 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 和
`PROMPTOS_INTEGRATION_MODEL`。

## Pull Request

- 保持改动聚焦，并为行为变化补充测试。
- 用户可见的功能或命令变化应同步更新 README。
- 保留 `gold_human`、`silver_auto` 和 `unlabeled` 的来源边界。
- 确认不会把数据集、运行产物、凭据或个人信息提交进仓库。

提交 Pull Request 即表示你同意按本项目的 MIT License 贡献代码。
