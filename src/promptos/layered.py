"""Cost-aware layered evaluation for prompt optimization.

All samples receive task-model outputs and deterministic checks. Only a shared
fixed-plus-dynamic risk set receives semantic LLM judging. Stage artifacts are
append-safe and make interrupted runs resumable by run ID.
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

from .core import ModelResponse, PromptGenerator, Sample, SignalSpec, TaskModel, TaskSpec


@dataclass(frozen=True)
class Violation:
    code: str
    message: str
    severity: str = "hard"


@dataclass(frozen=True)
class RiskSignal:
    name: str
    score: float
    reason: str
    source: str = "automatic"


@dataclass(frozen=True)
class DeterministicCheck:
    sample_id: str
    prompt_index: int
    parsed_output: Any | None
    violations: list[Violation]


@dataclass(frozen=True)
class LayeredEvaluationPolicy:
    max_candidates: int = 2
    fixed_sample_kinds: tuple[str, ...] = ("boundary_probe",)
    dynamic_top_k: int = 30
    judge_max_cases_per_prompt: int = 80
    human_review_top_k: int = 20
    task_model_max_calls: int = 2000
    judge_model_max_calls: int = 240
    max_seconds: float = 7200.0
    low_confidence_threshold: float = 0.65
    distribution_concentration_threshold: float = 0.40

    def validate(self) -> None:
        integer_values = {
            "max_candidates": self.max_candidates,
            "dynamic_top_k": self.dynamic_top_k,
            "judge_max_cases_per_prompt": self.judge_max_cases_per_prompt,
            "human_review_top_k": self.human_review_top_k,
            "task_model_max_calls": self.task_model_max_calls,
            "judge_model_max_calls": self.judge_model_max_calls,
        }
        if any(value < 0 for value in integer_values.values()):
            raise ValueError(f"Layered policy counts cannot be negative: {integer_values}")
        if not 0 <= self.low_confidence_threshold <= 1:
            raise ValueError("low_confidence_threshold must be between 0 and 1.")
        if not 0 < self.distribution_concentration_threshold <= 1:
            raise ValueError("distribution_concentration_threshold must be in (0, 1].")
        if self.max_seconds <= 0:
            raise ValueError("max_seconds must be positive.")


class OutputValidator(Protocol):
    def validate(self, sample: Sample, output: str) -> tuple[Any | None, list[Violation]]: ...


class RiskAnalyzer(Protocol):
    """Task-specific extension point for transparent, non-LLM risk signals."""

    def analyze(
        self,
        prompts: list[str],
        samples: list[Sample],
        checks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]: ...


class CaseSelector(Protocol):
    """Extension point for selecting one shared judge set for every prompt."""

    def select(
        self,
        samples: list[Sample],
        risks: list[dict[str, Any]],
        policy: LayeredEvaluationPolicy,
    ) -> dict[str, Any]: ...


class ContextualJudge(Protocol):
    def score(
        self,
        signal_spec: SignalSpec,
        sample: Sample,
        output: str,
        context: dict[str, Any] | None = None,
    ) -> Any: ...


class JsonOutputValidator:
    """Dependency-free JSON and shallow JSON-schema validation."""

    def __init__(self, schema: dict[str, Any] | None = None):
        self.schema = schema or {}

    def validate(self, sample: Sample, output: str) -> tuple[Any | None, list[Violation]]:
        del sample
        text = output.strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return None, [Violation("valid_json", "Output is not a standalone JSON value.")]
        violations: list[Violation] = []
        expected_type = self.schema.get("type")
        if expected_type == "object" and not isinstance(value, dict):
            violations.append(Violation("valid_schema", "Output must be a JSON object."))
            return value, violations
        if isinstance(value, dict):
            required = set(self.schema.get("required", []))
            missing = sorted(required - set(value))
            if missing:
                violations.append(Violation("valid_schema", f"Missing required fields: {missing}"))
            properties = self.schema.get("properties", {})
            if self.schema.get("additionalProperties") is False:
                extras = sorted(set(value) - set(properties))
                if extras:
                    violations.append(Violation("valid_schema", f"Unexpected fields: {extras}"))
            for name, rule in properties.items():
                if name not in value:
                    continue
                item = value[name]
                type_name = rule.get("type")
                valid_type = (
                    type_name == "string" and isinstance(item, str)
                    or type_name == "number" and isinstance(item, (int, float)) and not isinstance(item, bool)
                    or type_name == "integer" and isinstance(item, int) and not isinstance(item, bool)
                    or type_name == "boolean" and isinstance(item, bool)
                    or type_name == "object" and isinstance(item, dict)
                    or type_name == "array" and isinstance(item, list)
                    or type_name is None
                )
                if not valid_type:
                    violations.append(Violation("valid_schema", f"Field {name} has an invalid type."))
                    continue
                if isinstance(item, (int, float)) and not isinstance(item, bool):
                    if "minimum" in rule and item < rule["minimum"]:
                        violations.append(Violation("valid_schema", f"Field {name} is below minimum."))
                    if "maximum" in rule and item > rule["maximum"]:
                        violations.append(Violation("valid_schema", f"Field {name} is above maximum."))
        return value, violations


class CompositeOutputValidator:
    def __init__(self, validators: Iterable[OutputValidator]):
        self.validators = list(validators)

    def validate(self, sample: Sample, output: str) -> tuple[Any | None, list[Violation]]:
        parsed: Any | None = None
        violations: list[Violation] = []
        for validator in self.validators:
            value, found = validator.validate(sample, output)
            if parsed is None and value is not None:
                parsed = value
            violations.extend(found)
        unique = {(item.code, item.message, item.severity): item for item in violations}
        return parsed, list(unique.values())


@dataclass
class LayeredBudget:
    task_model_max_calls: int
    judge_model_max_calls: int
    max_seconds: float
    task_calls: int = 0
    judge_calls: int = 0
    generator_calls: int = 0
    task_cost_usd: float = 0.0
    judge_cost_usd: float = 0.0
    generator_cost_usd: float = 0.0
    started_at: float = field(default_factory=time.monotonic)

    def check_time(self) -> None:
        if time.monotonic() - self.started_at >= self.max_seconds:
            raise RuntimeError("Layered budget exhausted: maximum runtime reached.")

    def record_task(self, response: ModelResponse) -> None:
        if self.task_calls >= self.task_model_max_calls:
            raise RuntimeError("Layered budget exhausted: task-model call limit reached.")
        self.task_calls += 1
        self.task_cost_usd += response.cost_usd

    def record_judge(self, response: ModelResponse | None) -> None:
        if self.judge_calls >= self.judge_model_max_calls:
            raise RuntimeError("Layered budget exhausted: judge-model call limit reached.")
        self.judge_calls += 1
        if response:
            self.judge_cost_usd += response.cost_usd

    def record_generator(self, response: ModelResponse | None) -> None:
        self.generator_calls += 1
        if response:
            self.generator_cost_usd += response.cost_usd


@dataclass(frozen=True)
class LayeredOptimizationResult:
    run_id: str
    run_dir: Path
    champion_prompt: str
    champion_status: str
    comparison: list[dict[str, Any]]
    judge_case_ids: list[str]
    budget: dict[str, Any]


def _json_line_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _label_key(parsed: Any | None, output: str) -> str:
    if isinstance(parsed, dict):
        preferred = [str(parsed.get(name, "")) for name in ("L2_id", "L3_id") if name in parsed]
        if preferred:
            return "|".join(preferred)
        return json.dumps(parsed, ensure_ascii=False, sort_keys=True)
    return output.strip()


def _confidence(parsed: Any | None) -> float | None:
    if not isinstance(parsed, dict) or "confidence" not in parsed:
        return None
    try:
        value = float(parsed["confidence"])
    except (TypeError, ValueError):
        return None
    return value if 0 <= value <= 1 else None


class LayeredPromptOptimizer:
    def __init__(
        self,
        model: TaskModel,
        judge: ContextualJudge,
        generator: PromptGenerator,
        validator: OutputValidator,
        policy: LayeredEvaluationPolicy | None = None,
        risk_analyzer: RiskAnalyzer | None = None,
        case_selector: CaseSelector | None = None,
    ):
        self.model = model
        self.judge = judge
        self.generator = generator
        self.validator = validator
        self.policy = policy or LayeredEvaluationPolicy()
        self.risk_analyzer = risk_analyzer
        self.case_selector = case_selector
        self.policy.validate()

    @staticmethod
    def _auxiliary_response(component: Any) -> ModelResponse | None:
        model = getattr(component, "model", None)
        response = getattr(model, "last_response", None)
        if isinstance(response, ModelResponse):
            model.last_response = None
            return response
        return None

    def _generate_prompts(
        self,
        initial_prompt: str,
        task: TaskSpec,
        signals: SignalSpec,
        budget: LayeredBudget,
    ) -> list[str]:
        if self.policy.max_candidates == 0:
            return [initial_prompt]
        budget.check_time()
        proposed = self.generator.propose(
            initial_prompt,
            task,
            signals,
            "Generate concise candidates that improve deterministic compliance and ambiguous-boundary handling.",
        )
        budget.record_generator(self._auxiliary_response(self.generator))
        prompts = [initial_prompt]
        for item in proposed:
            value = str(item).strip()
            if value and value not in prompts:
                prompts.append(value)
            if len(prompts) >= self.policy.max_candidates + 1:
                break
        if len(prompts) == 1 and self.policy.max_candidates:
            raise RuntimeError(
                "Prompt generator produced no distinct candidates; refusing to run an optimization "
                "that only evaluates the initial prompt."
            )
        return prompts

    def _task_outputs(
        self,
        prompts: list[str],
        samples: list[Sample],
        path: Path,
        budget: LayeredBudget,
    ) -> list[dict[str, Any]]:
        rows = _json_line_records(path)
        completed = {(int(row["prompt_index"]), str(row["sample_id"])) for row in rows}
        for prompt_index, prompt in enumerate(prompts):
            for sample in samples:
                key = (prompt_index, sample.id)
                if key in completed:
                    continue
                budget.check_time()
                if budget.task_calls >= budget.task_model_max_calls:
                    raise RuntimeError("Layered budget exhausted: task-model call limit reached.")
                response = self.model.generate(prompt, sample.inputs)
                budget.record_task(response)
                rows.append({
                    "prompt_index": prompt_index,
                    "sample_id": sample.id,
                    "output": response.text,
                    "model": response.model,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                })
                _write_jsonl(path, rows)
        return rows

    def _checks(
        self,
        task_rows: list[dict[str, Any]],
        samples_by_id: dict[str, Sample],
        path: Path,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in task_rows:
            sample = samples_by_id[str(item["sample_id"])]
            parsed, violations = self.validator.validate(sample, str(item["output"]))
            rows.append({
                "prompt_index": int(item["prompt_index"]),
                "sample_id": sample.id,
                "parsed_output": parsed,
                "violations": [asdict(value) for value in violations],
                "hard_failure": any(value.severity == "hard" for value in violations),
                "label_key": _label_key(parsed, str(item["output"])),
                "confidence": _confidence(parsed),
            })
        _write_jsonl(path, rows)
        return rows

    def _risk_scores(
        self,
        prompts: list[str],
        samples: list[Sample],
        checks: list[dict[str, Any]],
        path: Path,
    ) -> list[dict[str, Any]]:
        if self.risk_analyzer is not None:
            rows = self.risk_analyzer.analyze(prompts, samples, checks)
            rows.sort(key=lambda item: (-float(item["risk_score"]), str(item["sample_id"])))
            _write_jsonl(path, rows)
            return rows
        by_sample: dict[str, list[dict[str, Any]]] = {}
        for item in checks:
            by_sample.setdefault(str(item["sample_id"]), []).append(item)
        distributions: dict[int, Counter[str]] = {}
        for item in checks:
            distributions.setdefault(int(item["prompt_index"]), Counter())[str(item["label_key"])] += 1
        sample_map = {sample.id: sample for sample in samples}
        rows: list[dict[str, Any]] = []
        for sample_id, entries in by_sample.items():
            entries.sort(key=lambda value: int(value["prompt_index"]))
            signals: list[RiskSignal] = []
            if any(entry["hard_failure"] for entry in entries):
                signals.append(RiskSignal("deterministic_failure", 1.0, "At least one prompt produced a hard violation."))
            confidences = [entry["confidence"] for entry in entries if entry["confidence"] is not None]
            if confidences and min(confidences) < self.policy.low_confidence_threshold:
                signals.append(RiskSignal(
                    "low_confidence",
                    0.5,
                    f"Minimum confidence {min(confidences):.3f} is below {self.policy.low_confidence_threshold:.3f}.",
                ))
            if any(
                isinstance(entry["parsed_output"], dict)
                and (
                    entry["parsed_output"].get("L3_id") == "Unknown"
                    or entry["parsed_output"].get("label") == "Unknown"
                )
                for entry in entries
            ):
                signals.append(RiskSignal("unknown_output", 0.4, "At least one prompt returned Unknown."))
            labels = {str(entry["label_key"]) for entry in entries}
            if len(labels) > 1:
                signals.append(RiskSignal("label_flip", 0.8, f"Prompts produced {len(labels)} distinct labels."))
            sample = sample_map[sample_id]
            business_risk = float(sample.metadata.get("business_risk", 0.0))
            if business_risk > 0:
                signals.append(RiskSignal("business_risk", min(1.0, business_risk), "Sample metadata marks business risk.", "metadata"))
            for entry in entries:
                prompt_index = int(entry["prompt_index"])
                share = distributions[prompt_index][str(entry["label_key"])] / max(len(samples), 1)
                if share >= self.policy.distribution_concentration_threshold:
                    signals.append(RiskSignal(
                        "distribution_concentration",
                        0.3,
                        f"Label occupies {share:.1%} of prompt {prompt_index} outputs.",
                    ))
                    break
            score = sum(signal.score for signal in signals)
            rows.append({
                "sample_id": sample_id,
                "sample_kind": sample.metadata.get("sample_kind", ""),
                "risk_score": round(score, 4),
                "risk_signals": [asdict(signal) for signal in signals],
                "hard_failure": any(entry["hard_failure"] for entry in entries),
                "prompt_count": len(prompts),
            })
        rows.sort(key=lambda item: (-float(item["risk_score"]), str(item["sample_id"])))
        _write_jsonl(path, rows)
        return rows

    def _select_cases(
        self,
        samples: list[Sample],
        risks: list[dict[str, Any]],
        path: Path,
    ) -> dict[str, Any]:
        if self.case_selector is not None:
            payload = self.case_selector.select(samples, risks, self.policy)
            fixed = list(payload.get("fixed_case_ids", []))
            dynamic = list(payload.get("dynamic_case_ids", []))
            payload = {
                "policy": asdict(self.policy),
                "fixed_case_ids": fixed,
                "dynamic_case_ids": dynamic,
                "case_ids": fixed + dynamic,
            }
            if len(payload["case_ids"]) > self.policy.judge_max_cases_per_prompt:
                raise ValueError("CaseSelector returned more cases than the per-prompt judge cap.")
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return payload
        fixed = [
            sample.id for sample in samples
            if sample.metadata.get("sample_kind") in set(self.policy.fixed_sample_kinds)
        ]
        if len(fixed) > self.policy.judge_max_cases_per_prompt:
            raise ValueError(
                f"Fixed judge set has {len(fixed)} cases, exceeding cap "
                f"{self.policy.judge_max_cases_per_prompt}."
            )
        fixed_set = set(fixed)
        dynamic_candidates = [
            item for item in risks
            if item["sample_id"] not in fixed_set
            and not item["hard_failure"]
            and float(item["risk_score"]) > 0
        ]
        available = self.policy.judge_max_cases_per_prompt - len(fixed)
        dynamic_limit = min(self.policy.dynamic_top_k, available)
        dynamic = [str(item["sample_id"]) for item in dynamic_candidates[:dynamic_limit]]
        payload = {
            "policy": asdict(self.policy),
            "fixed_case_ids": fixed,
            "dynamic_case_ids": dynamic,
            "case_ids": fixed + dynamic,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return payload

    def _judge(
        self,
        prompts: list[str],
        samples_by_id: dict[str, Sample],
        task_rows: list[dict[str, Any]],
        checks: list[dict[str, Any]],
        risks: list[dict[str, Any]],
        queue: dict[str, Any],
        signals: SignalSpec,
        path: Path,
        budget: LayeredBudget,
    ) -> list[dict[str, Any]]:
        rows = _json_line_records(path)
        completed = {(int(row["prompt_index"]), str(row["sample_id"])) for row in rows}
        output_map = {(int(row["prompt_index"]), str(row["sample_id"])): str(row["output"]) for row in task_rows}
        check_map = {(int(row["prompt_index"]), str(row["sample_id"])): row for row in checks}
        risk_map = {str(row["sample_id"]): row for row in risks}
        comparison_outputs = {
            sample_id: {
                str(prompt_index): output_map[(prompt_index, sample_id)]
                for prompt_index in range(len(prompts))
            }
            for sample_id in queue["case_ids"]
        }
        for prompt_index, _prompt in enumerate(prompts):
            for sample_id in queue["case_ids"]:
                key = (prompt_index, sample_id)
                if key in completed:
                    continue
                check = check_map[key]
                if check["hard_failure"]:
                    continue
                budget.check_time()
                if budget.judge_calls >= budget.judge_model_max_calls:
                    raise RuntimeError("Layered budget exhausted: judge-model call limit reached.")
                sample = samples_by_id[sample_id]
                context = {
                    "metadata": sample.metadata,
                    "boundary_note": sample.metadata.get("boundary_note"),
                    "risk_reasons": risk_map[sample_id]["risk_signals"],
                    "comparison_outputs": comparison_outputs[sample_id],
                    "prompt_index": prompt_index,
                }
                judgment = self.judge.score(signals, sample, output_map[key], context)
                response = self._auxiliary_response(self.judge)
                budget.record_judge(response)
                rows.append({
                    "prompt_index": prompt_index,
                    "sample_id": sample_id,
                    "set": "fixed" if sample_id in set(queue["fixed_case_ids"]) else "dynamic",
                    "judgment": asdict(judgment),
                    "context": context,
                })
                _write_jsonl(path, rows)
        return rows

    def _stability(
        self,
        prompts: list[str],
        samples_by_id: dict[str, Sample],
        task_rows: list[dict[str, Any]],
        queue: dict[str, Any],
        path: Path,
        budget: LayeredBudget,
    ) -> list[dict[str, Any]]:
        rows = _json_line_records(path)
        completed = {(int(row["prompt_index"]), str(row["sample_id"])) for row in rows}
        original = {(int(row["prompt_index"]), str(row["sample_id"])): str(row["output"]) for row in task_rows}
        for prompt_index, prompt in enumerate(prompts):
            for sample_id in queue["case_ids"]:
                key = (prompt_index, sample_id)
                if key in completed:
                    continue
                budget.check_time()
                if budget.task_calls >= budget.task_model_max_calls:
                    raise RuntimeError("Layered budget exhausted: task-model call limit reached.")
                response = self.model.generate(prompt, samples_by_id[sample_id].inputs)
                budget.record_task(response)
                first_parsed, _ = self.validator.validate(samples_by_id[sample_id], original[key])
                second_parsed, _ = self.validator.validate(samples_by_id[sample_id], response.text)
                rows.append({
                    "prompt_index": prompt_index,
                    "sample_id": sample_id,
                    "output": response.text,
                    "stable": _label_key(first_parsed, original[key]) == _label_key(second_parsed, response.text),
                })
                _write_jsonl(path, rows)
        return rows

    def _compare(
        self,
        prompts: list[str],
        checks: list[dict[str, Any]],
        judge_rows: list[dict[str, Any]],
        stability_rows: list[dict[str, Any]],
        queue: dict[str, Any],
        task_rows: list[dict[str, Any]],
        path: Path,
    ) -> list[dict[str, Any]]:
        fixed = set(queue["fixed_case_ids"])
        comparison: list[dict[str, Any]] = []
        for prompt_index, prompt in enumerate(prompts):
            prompt_checks = [row for row in checks if int(row["prompt_index"]) == prompt_index]
            prompt_judgments = [row for row in judge_rows if int(row["prompt_index"]) == prompt_index]
            fixed_scores = [
                float(row["judgment"]["score"]) for row in prompt_judgments if row["sample_id"] in fixed
            ]
            dynamic_scores = [
                float(row["judgment"]["score"]) for row in prompt_judgments if row["sample_id"] not in fixed
            ]
            stability = [
                bool(row["stable"]) for row in stability_rows if int(row["prompt_index"]) == prompt_index
            ]
            token_count = sum(
                int(row.get("input_tokens", 0)) + int(row.get("output_tokens", 0))
                for row in task_rows if int(row["prompt_index"]) == prompt_index
            )
            item = {
                "prompt_index": prompt_index,
                "prompt": prompt,
                "hard_failure_count": sum(bool(row["hard_failure"]) for row in prompt_checks),
                "boundary_judge_score": sum(fixed_scores) / len(fixed_scores) if fixed_scores else 0.0,
                "dynamic_judge_score": sum(dynamic_scores) / len(dynamic_scores) if dynamic_scores else 0.0,
                "stability_rate": sum(stability) / len(stability) if stability else 1.0,
                "prompt_chars": len(prompt),
                "task_tokens": token_count,
            }
            item["selection_key"] = [
                item["hard_failure_count"],
                -item["boundary_judge_score"],
                -item["dynamic_judge_score"],
                -item["stability_rate"],
                item["prompt_chars"],
                item["task_tokens"],
            ]
            comparison.append(item)
        comparison.sort(key=lambda item: tuple(item["selection_key"]))
        path.write_text(json.dumps({"ranking": comparison}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return comparison

    def _human_review(
        self,
        samples_by_id: dict[str, Sample],
        risks: list[dict[str, Any]],
        checks: list[dict[str, Any]],
        judge_rows: list[dict[str, Any]],
        task_rows: list[dict[str, Any]],
        queue: dict[str, Any],
        path: Path,
    ) -> None:
        risk_map = {str(row["sample_id"]): row for row in risks}
        check_by_sample: dict[str, list[dict[str, Any]]] = {}
        for row in checks:
            check_by_sample.setdefault(str(row["sample_id"]), []).append(row)
        judge_by_sample: dict[str, list[dict[str, Any]]] = {}
        for row in judge_rows:
            judge_by_sample.setdefault(str(row["sample_id"]), []).append(row)
        output_by_sample: dict[str, dict[str, str]] = {}
        for row in task_rows:
            output_by_sample.setdefault(str(row["sample_id"]), {})[str(row["prompt_index"])] = str(row["output"])
        candidates: list[dict[str, Any]] = []
        all_ids = set(queue["case_ids"]) | {
            sample_id for sample_id, values in check_by_sample.items()
            if any(value["hard_failure"] for value in values)
        }
        for sample_id in all_ids:
            sample = samples_by_id[sample_id]
            judgments = judge_by_sample.get(sample_id, [])
            judge_confidences = [
                float(row["judgment"]["confidence"])
                for row in judgments if row["judgment"].get("confidence") is not None
            ]
            judge_scores = [float(row["judgment"]["score"]) for row in judgments]
            risk_names = [item["name"] for item in risk_map[sample_id]["risk_signals"]]
            hard_failure = any(row["hard_failure"] for row in check_by_sample.get(sample_id, []))
            business_risk = float(sample.metadata.get("business_risk", 0.0))
            task_confidences = [
                float(row["confidence"])
                for row in check_by_sample.get(sample_id, [])
                if row.get("confidence") is not None
            ]
            high_confidence_conflict = (
                business_risk > 0
                and task_confidences
                and max(task_confidences) >= 0.8
                and judge_scores
                and min(judge_scores) < 0.5
                and judge_confidences
                and max(judge_confidences) >= 0.8
            )
            boundary_failure = (
                sample.metadata.get("sample_kind") in set(self.policy.fixed_sample_kinds)
                and judge_scores and min(judge_scores) < 0.5
            )
            low_judge_confidence = judge_confidences and min(judge_confidences) < 0.6
            if high_confidence_conflict:
                priority_band = 0
            elif boundary_failure or low_judge_confidence:
                priority_band = 1
            elif "label_flip" in risk_names:
                priority_band = 2
            else:
                priority_band = 3
            candidates.append({
                "priority_band": priority_band,
                "risk_score": float(risk_map[sample_id]["risk_score"]),
                "sample_id": sample_id,
                "inputs_json": json.dumps(sample.inputs, ensure_ascii=False),
                "sample_kind": sample.metadata.get("sample_kind", ""),
                "boundary_note": sample.metadata.get("boundary_note", ""),
                "risk_reasons": json.dumps(risk_names, ensure_ascii=False),
                "prompt_outputs_json": json.dumps(output_by_sample.get(sample_id, {}), ensure_ascii=False),
                "judge_results_json": json.dumps(judgments, ensure_ascii=False),
                "hard_failure": hard_failure,
                "decision": "",
                "reviewed_value": "",
                "reviewer": "",
                "review_rationale": "",
            })
        candidates.sort(key=lambda row: (row["priority_band"], -row["risk_score"], row["sample_id"]))
        selected = candidates[:self.policy.human_review_top_k]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as file:
            fieldnames = list(selected[0]) if selected else [
                "priority_band", "risk_score", "sample_id", "inputs_json", "sample_kind",
                "boundary_note", "risk_reasons", "prompt_outputs_json", "judge_results_json",
                "hard_failure", "decision", "reviewed_value", "reviewer", "review_rationale",
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected)

    def optimize(
        self,
        initial_prompt: str,
        task: TaskSpec,
        signals: SignalSpec,
        samples: Iterable[Sample],
        runs_root: Path,
        run_id: str | None = None,
        task_config: dict[str, Any] | None = None,
    ) -> LayeredOptimizationResult:
        if signals.status != "approved":
            raise ValueError("Only an approved SignalSpec can drive a layered optimization run.")
        frozen = list(samples)
        if not frozen:
            raise ValueError("At least one sample is required.")
        fixed_count = sum(
            sample.metadata.get("sample_kind") in set(self.policy.fixed_sample_kinds)
            for sample in frozen
        )
        if fixed_count > self.policy.judge_max_cases_per_prompt:
            raise ValueError(
                f"Fixed judge set has {fixed_count} cases, exceeding cap "
                f"{self.policy.judge_max_cases_per_prompt}."
            )
        identifier = run_id or f"layered_{uuid.uuid4().hex[:12]}"
        run_dir = runs_root / identifier
        run_dir.mkdir(parents=True, exist_ok=True)
        identity = {
            "task": asdict(task),
            "initial_prompt": initial_prompt,
            "signal_spec": {"id": signals.id, "version": signals.version},
            "sample_hash": hashlib.sha256(
                "\n".join(
                    json.dumps(asdict(sample), ensure_ascii=False, sort_keys=True)
                    for sample in frozen
                ).encode()
            ).hexdigest(),
            "policy": asdict(self.policy),
        }
        identity = json.loads(json.dumps(identity, ensure_ascii=False))
        identity_path = run_dir / "run_identity.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
                raise ValueError("Resume run identity does not match task, signals, samples, or policy.")
        else:
            identity_path.write_text(json.dumps(identity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        budget = LayeredBudget(
            self.policy.task_model_max_calls,
            self.policy.judge_model_max_calls,
            self.policy.max_seconds,
        )
        prompts_path = run_dir / "prompts.json"
        if prompts_path.exists():
            prompts = json.loads(prompts_path.read_text(encoding="utf-8"))["prompts"]
        else:
            prompts = self._generate_prompts(initial_prompt, task, signals, budget)
            prompts_path.write_text(json.dumps({"prompts": prompts}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        samples_by_id = {sample.id: sample for sample in frozen}
        if len(samples_by_id) != len(frozen):
            raise ValueError("Sample ids must be unique for layered evaluation.")
        task_rows = self._task_outputs(prompts, frozen, run_dir / "task_outputs.jsonl", budget)
        checks = self._checks(task_rows, samples_by_id, run_dir / "deterministic_checks.jsonl")
        risks = self._risk_scores(prompts, frozen, checks, run_dir / "risk_scores.jsonl")
        queue = self._select_cases(frozen, risks, run_dir / "judge_queue.json")
        judge_rows = self._judge(
            prompts, samples_by_id, task_rows, checks, risks, queue, signals,
            run_dir / "judge_results.jsonl", budget,
        )
        stability_rows = self._stability(
            prompts, samples_by_id, task_rows, queue,
            run_dir / "stability_outputs.jsonl", budget,
        )
        comparison = self._compare(
            prompts, checks, judge_rows, stability_rows, queue, task_rows,
            run_dir / "prompt_comparison.json",
        )
        self._human_review(
            samples_by_id, risks, checks, judge_rows, task_rows, queue,
            run_dir / "human_review_top20.csv",
        )
        champion = comparison[0]
        status = (
            "gold_validated"
            if any(sample.label_source == "gold_human" for sample in frozen)
            else "provisional_silver_or_unlabeled"
        )
        manifest = {
            "run_id": identifier,
            "mode": "layered",
            "task": asdict(task),
            "signal_spec": asdict(signals),
            "sample_count": len(frozen),
            "champion_prompt": champion["prompt"],
            "champion_status": status,
            "selection_policy": [
                "hard_failure_count",
                "boundary_judge_score",
                "dynamic_judge_score",
                "stability_rate",
                "prompt_chars",
                "task_tokens",
            ],
            "judge_case_count": len(queue["case_ids"]),
            "policy": asdict(self.policy),
            "budget": asdict(budget),
        }
        if task_config is not None:
            manifest["task_config"] = task_config
        (run_dir / "run.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return LayeredOptimizationResult(
            identifier, run_dir, champion["prompt"], status, comparison,
            list(queue["case_ids"]), asdict(budget),
        )
