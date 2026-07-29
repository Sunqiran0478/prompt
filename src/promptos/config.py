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


@dataclass(frozen=True)
class OptimizationConfig:
    initial_prompt: str
    rounds: int = 3
    max_calls: int = 100
    max_cost_usd: float = 10.0
    max_seconds: float = 900.0


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
    missing = [name for name, value in {
        "task.name": task.get("name"), "task.input_fields": task.get("input_fields"),
        "task.output_description": task.get("output_description"), "dataset.path": dataset.get("path"),
        "signals.path": raw.get("signals", {}).get("path"), "optimization.initial_prompt": optimization.get("initial_prompt"),
    }.items() if not value]
    if missing:
        raise ValueError(f"Task config is missing required values: {', '.join(missing)}")
    input_fields = task["input_fields"]
    if not isinstance(input_fields, list) or not all(isinstance(field, str) and field for field in input_fields):
        raise ValueError("task.input_fields must be a non-empty list of strings.")
    models = raw.get("models", {})
    if any("key" in key.lower() or "token" in key.lower() or "secret" in key.lower() for key in models):
        raise ValueError("Do not store credentials in task config; use environment variables.")
    return TaskConfig(
        path=path, name=str(task["name"]), input_fields=input_fields,
        output_description=str(task["output_description"]), output_schema=task.get("output_schema", {}),
        dataset=DatasetConfig(_resolve(root, dataset["path"]), input_fields, dataset.get("expected_field"),
                              _resolve(root, dataset.get("annotations"))),
        signal_spec=_resolve(root, raw["signals"]["path"]),
        models=ModelConfig(models.get("task_model"), models.get("judge_model"), models.get("generator_model"), models.get("base_url")),
        optimization=OptimizationConfig(str(optimization["initial_prompt"]), int(optimization.get("rounds", 3)),
                                        int(optimization.get("max_calls", 100)), float(optimization.get("max_cost_usd", 10.0)),
                                        float(optimization.get("max_seconds", 900.0))),
        runs_dir=_resolve(root, raw.get("runs_dir", "runs")), metadata=raw.get("metadata", {}),
    )
