"""Versioned JSON task configuration for reproducible PromptOS runs."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetConfig:
    path: Path
    input_fields: list[str]
    expected_field: str | None = None
    annotations: Path | None = None


@dataclass(frozen=True)
class ModelConfig:
    task_model: str | None = None
    judge_model: str | None = None
    generator_model: str | None = None
    base_url: str | None = None
    task_response_format: str | None = None
    task_max_tokens: int | None = None


@dataclass(frozen=True)
class OptimizationConfig:
    initial_prompt: str
    initial_prompt_path: Path | None = None
    rounds: int = 3
    max_calls: int = 100
    max_cost_usd: float = 10.0
    max_seconds: float = 900.0


@dataclass(frozen=True)
class PluginConfig:
    name: str | None = None
    taxonomy_path: Path | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    mode: str = "legacy"
    max_candidates: int = 2
    fixed_sample_kinds: tuple[str, ...] = ("boundary_probe",)
    dynamic_top_k: int = 30
    judge_max_cases_per_prompt: int = 80
    human_review_top_k: int = 20
    task_model_max_calls: int = 2000
    judge_model_max_calls: int = 240
    max_seconds: float = 7200.0
    max_rounds: int = 5
    stop_after_no_improvement: int = 2
    minimum_improvement: float = 0.01
    retain_risk_memory: bool = True
    first_round_hard_failure_stop_count: int | None = None
    judge_workers: int = 1


@dataclass(frozen=True)
class TaskConfig:
    path: Path
    name: str
    input_fields: list[str]
    output_description: str
    output_schema: dict[str, Any]
    dataset: DatasetConfig
    signal_spec: Path
    models: ModelConfig
    optimization: OptimizationConfig
    plugin: PluginConfig
    evaluation: EvaluationConfig
    runs_dir: Path
    metadata: dict[str, Any] = field(default_factory=dict)


def _resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def load_task_config(path: Path) -> TaskConfig:
    """Load and validate the portable JSON config without storing credentials."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Task config must be valid JSON: {error}") from error
    if raw.get("version") != 1:
        raise ValueError("Task config requires version: 1.")
    root = path.parent
    task = raw.get("task", {})
    dataset = raw.get("dataset", {})
    optimization = raw.get("optimization", {})
    plugin = raw.get("plugin", {})
    evaluation = raw.get("evaluation", {})
    missing = [name for name, value in {
        "task.name": task.get("name"), "task.input_fields": task.get("input_fields"),
        "task.output_description": task.get("output_description"), "dataset.path": dataset.get("path"),
        "signals.path": raw.get("signals", {}).get("path"),
    }.items() if not value]
    evaluation_mode = str(evaluation.get("mode", "legacy"))
    initial_prompt_path = _resolve(root, optimization.get("initial_prompt_path"))
    if not optimization.get("initial_prompt") and initial_prompt_path is None and not (
        plugin.get("name") == "finance_classification"
        and evaluation_mode in {"layered", "layered_auto"}
    ):
        missing.append("optimization.initial_prompt or optimization.initial_prompt_path")
    if missing:
        raise ValueError(f"Task config is missing required values: {', '.join(missing)}")
    input_fields = task["input_fields"]
    if not isinstance(input_fields, list) or not all(isinstance(field, str) and field for field in input_fields):
        raise ValueError("task.input_fields must be a non-empty list of strings.")
    models = raw.get("models", {})
    if any(
        key.lower() in {
            "key", "token", "secret", "api_key", "auth_token",
            "access_token", "bearer_token", "client_secret",
        }
        or key.lower().endswith(("_api_key", "_token", "_secret"))
        for key in models
    ):
        raise ValueError("Do not store credentials in task config; use environment variables.")
    task_response_format = models.get("task_response_format")
    if task_response_format not in {None, "text", "json_object"}:
        raise ValueError("models.task_response_format must be text or json_object.")
    task_max_tokens = models.get("task_max_tokens")
    if task_max_tokens is not None and int(task_max_tokens) <= 0:
        raise ValueError("models.task_max_tokens must be positive.")
    mode = evaluation_mode
    if mode not in {"legacy", "layered", "layered_auto"}:
        raise ValueError("evaluation.mode must be legacy, layered, or layered_auto.")
    fixed_sample_kinds = evaluation.get("fixed_sample_kinds", ["boundary_probe"])
    if not isinstance(fixed_sample_kinds, list) or not all(isinstance(item, str) and item for item in fixed_sample_kinds):
        raise ValueError("evaluation.fixed_sample_kinds must be a list of non-empty strings.")
    plugin_name = plugin.get("name")
    if plugin_name not in {None, "finance_classification"}:
        raise ValueError(f"Unsupported plugin: {plugin_name}")
    taxonomy_path = _resolve(root, plugin.get("taxonomy_path"))
    if plugin_name == "finance_classification" and taxonomy_path is None:
        raise ValueError("finance_classification plugin requires taxonomy_path.")
    evaluation_config = EvaluationConfig(
        mode, int(evaluation.get("max_candidates", 2)), tuple(fixed_sample_kinds),
        int(evaluation.get("dynamic_top_k", 30)),
        int(evaluation.get("judge_max_cases_per_prompt", 80)),
        int(evaluation.get("human_review_top_k", 20)),
        int(evaluation.get("task_model_max_calls", 2000)),
        int(evaluation.get("judge_model_max_calls", 240)),
        float(evaluation.get("max_seconds", 7200.0)),
        int(evaluation.get("max_rounds", 5)),
        int(evaluation.get("stop_after_no_improvement", 2)),
        float(evaluation.get("minimum_improvement", 0.01)),
        bool(evaluation.get("retain_risk_memory", True)),
        (
            int(evaluation["first_round_hard_failure_stop_count"])
            if evaluation.get("first_round_hard_failure_stop_count") is not None
            else None
        ),
        int(evaluation.get("judge_workers", 1)),
    )
    counts = (
        evaluation_config.max_candidates,
        evaluation_config.dynamic_top_k,
        evaluation_config.judge_max_cases_per_prompt,
        evaluation_config.human_review_top_k,
        evaluation_config.task_model_max_calls,
        evaluation_config.judge_model_max_calls,
        evaluation_config.max_rounds,
        evaluation_config.stop_after_no_improvement,
        evaluation_config.judge_workers,
    )
    if (
        any(value < 0 for value in counts)
        or evaluation_config.max_seconds <= 0
        or evaluation_config.minimum_improvement < 0
        or evaluation_config.max_rounds <= 0
        or evaluation_config.stop_after_no_improvement <= 0
        or evaluation_config.judge_workers <= 0
        or (
            evaluation_config.first_round_hard_failure_stop_count is not None
            and evaluation_config.first_round_hard_failure_stop_count < 0
        )
    ):
        raise ValueError(
            "Layered evaluation counts and minimum_improvement must be non-negative, "
            "and max_seconds must be positive."
        )
    return TaskConfig(
        path=path, name=str(task["name"]), input_fields=input_fields,
        output_description=str(task["output_description"]), output_schema=task.get("output_schema", {}),
        dataset=DatasetConfig(_resolve(root, dataset["path"]), input_fields, dataset.get("expected_field"),
                              _resolve(root, dataset.get("annotations"))),
        signal_spec=_resolve(root, raw["signals"]["path"]),
        models=ModelConfig(
            models.get("task_model"),
            models.get("judge_model"),
            models.get("generator_model"),
            models.get("base_url"),
            task_response_format,
            int(task_max_tokens) if task_max_tokens is not None else None,
        ),
        optimization=OptimizationConfig(str(optimization.get("initial_prompt", "")), initial_prompt_path,
                                        int(optimization.get("rounds", 3)),
                                        int(optimization.get("max_calls", 100)), float(optimization.get("max_cost_usd", 10.0)),
                                        float(optimization.get("max_seconds", 900.0))),
        plugin=PluginConfig(plugin_name, taxonomy_path),
        evaluation=evaluation_config,
        runs_dir=_resolve(root, raw.get("runs_dir", "runs")), metadata=raw.get("metadata", {}),
    )
