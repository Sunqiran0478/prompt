# Finance unlabeled-200 experiment

这是个人实验 case，不是 PromptOS 核心功能或通用数据管道。

## 数据构成

- 150 条从金融 query 池随机抽取的无标注样本
- 50 条人工编写的边界 probe
- 排除 held-out request ID
- 输出不包含 user ID、enterprise ID 或原始 request ID

## 构建

此实验脚本额外需要 `pandas` 和 `pyarrow`：

```bash
python -m pip install pandas pyarrow
python experiments/finance_unlabeled_200/build_sample.py \
  --pool /path/to/finance_l2_conversation.parquet \
  --probe /path/to/probe_50.parquet \
  --held-out /path/to/held_out_400.parquet \
  --output-dir artifacts/datasets/finance_unlabeled_200 \
  --total 200 \
  --boundary 50 \
  --seed 20260729
```

生成的 JSONL、Parquet 和 manifest 位于被 Git 忽略的 `artifacts/`，不会作为
项目包内容发布。

## 准备实验输入

Taxonomy 和初始 Prompt 可能包含业务信息，因此示例配置只引用本地、被 Git
忽略的实验目录。运行前请准备：

```text
artifacts/experiments/finance_unlabeled_200/taxonomy.json
artifacts/experiments/finance_unlabeled_200/initial_prompt.md
artifacts/experiments/finance_unlabeled_200/signals.json
```

然后使用 `layered.task.json` 启动单轮分层实验，或使用
`layered_auto_4rounds.task.json` 启动四轮自动实验。配置文件中不应保存 API Key
或个人电脑的绝对路径。
