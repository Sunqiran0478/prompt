"""Unattended multi-round prompt optimization built on layered evaluation."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable

from .core import PromptGenerator, Sample, SignalSpec, TaskModel, TaskSpec
from .layered import (
    ContextualJudge,
    LayeredBudget,
    LayeredEvaluationPolicy,
    LayeredPromptOptimizer,
    OutputValidator,
    _json_line_records,
    _prompt_key,
)
from .reporting import render_auto_report

try:
    _SOFTWARE_VERSION = version("promptos")
except PackageNotFoundError:
    _SOFTWARE_VERSION = "development"


@dataclass(frozen=True)
class AutoOptimizationPolicy:
    max_rounds: int = 5
    stop_after_no_improvement: int = 2
    minimum_improvement: float = 0.01
    retain_risk_memory: bool = True
    first_round_hard_failure_stop_count: int | None = None

    def validate(self) -> None:
        if self.max_rounds <= 0:
            raise ValueError("max_rounds must be positive.")
        if self.stop_after_no_improvement <= 0:
            raise ValueError("stop_after_no_improvement must be positive.")
        if self.minimum_improvement < 0:
            raise ValueError("minimum_improvement cannot be negative.")
        if (
            self.first_round_hard_failure_stop_count is not None
            and self.first_round_hard_failure_stop_count < 0
        ):
            raise ValueError("first_round_hard_failure_stop_count cannot be negative.")


@dataclass(frozen=True)
class AutoOptimizationResult:
    run_id: str
    run_dir: Path
    champion_prompt: str
    champion_status: str
    rounds_completed: int
    stop_reason: str
    risk_memory_case_ids: list[str]
    budget: dict[str, Any]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sample_hash(samples: list[Sample]) -> str:
    payload = "\n".join(
        json.dumps(asdict(sample), ensure_ascii=False, sort_keys=True)
        for sample in samples
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _failure_summary(round_dir: Path) -> dict[str, Any]:
    checks = _json_line_records(round_dir / "deterministic_checks.jsonl")
    risks = _json_line_records(round_dir / "risk_scores.jsonl")
    judgments = _json_line_records(round_dir / "judge_results.jsonl")
    violation_counts: Counter[str] = Counter()
    for row in checks:
        for violation in row["violations"]:
            violation_counts[str(violation["code"])] += 1
    risk_counts: Counter[str] = Counter()
    for row in risks:
        for signal in row["risk_signals"]:
            risk_counts[str(signal["name"])] += 1
    low_judgments = sorted(
        judgments,
        key=lambda row: (
            float(row["judgment"]["score"]),
            str(row["sample_id"]),
            int(row["prompt_index"]),
        ),
    )[:20]
    top_risks = sorted(
        risks,
        key=lambda row: (-float(row["risk_score"]), str(row["sample_id"])),
    )[:20]
    return {
        "hard_failure_count": sum(bool(row["hard_failure"]) for row in checks),
        "hard_violation_counts": dict(sorted(violation_counts.items())),
        "risk_signal_counts": dict(sorted(risk_counts.items())),
        "top_risk_cases": [
            {
                "sample_id": row["sample_id"],
                "risk_score": row["risk_score"],
                "reasons": row["risk_signals"],
            }
            for row in top_risks
        ],
        "lowest_judge_results": [
            {
                "sample_id": row["sample_id"],
                "prompt_index": row["prompt_index"],
                "score": row["judgment"]["score"],
                "confidence": row["judgment"].get("confidence"),
                "rationale": row["judgment"]["rationale"],
            }
            for row in low_judgments
        ],
    }


def _generation_feedback(summary: dict[str, Any], risk_memory: list[str]) -> str:
    compact = {
        "instruction": (
            "Create candidates that directly fix these observed failures without regressing "
            "previously selected risk cases. Do not copy the current prompt unchanged."
        ),
        "hard_violation_counts": summary["hard_violation_counts"],
        "risk_signal_counts": summary["risk_signal_counts"],
        "top_risk_cases": summary["top_risk_cases"][:10],
        "lowest_judge_results": summary["lowest_judge_results"][:10],
        "risk_memory_case_ids": risk_memory,
    }
    return json.dumps(compact, ensure_ascii=False)


def _improvement(
    comparison: list[dict[str, Any]],
    current_prompt: str,
    minimum: float,
) -> tuple[str, bool, dict[str, Any]]:
    current = next(
        row for row in comparison
        if _prompt_key(str(row["prompt"])) == _prompt_key(current_prompt)
    )
    proposed = comparison[0]
    changed = _prompt_key(str(proposed["prompt"])) != _prompt_key(current_prompt)
    hard_delta = int(current["hard_failure_count"]) - int(proposed["hard_failure_count"])
    metric_deltas = {
        "boundary_judge_score": (
            float(proposed["boundary_judge_score"])
            - float(current["boundary_judge_score"])
        ),
        "dynamic_judge_score": (
            float(proposed["dynamic_judge_score"])
            - float(current["dynamic_judge_score"])
        ),
        "stability_rate": (
            float(proposed["stability_rate"])
            - float(current["stability_rate"])
        ),
    }
    material = hard_delta > 0 or max(metric_deltas.values()) >= minimum
    accepted = changed and material
    selected = str(proposed["prompt"]) if accepted else current_prompt
    return selected, accepted, {
        "changed": changed,
        "accepted": accepted,
        "hard_failure_reduction": hard_delta,
        "metric_deltas": metric_deltas,
        "minimum_improvement": minimum,
        "ranked_winner_prompt_index": proposed["prompt_index"],
    }


class AutoPromptOptimizer:
    """Runs layered rounds until convergence or an explicit resource limit."""

    def __init__(
        self,
        model: TaskModel,
        judge: ContextualJudge,
        generator: PromptGenerator,
        validator: OutputValidator,
        layered_policy: LayeredEvaluationPolicy | None = None,
        auto_policy: AutoOptimizationPolicy | None = None,
    ):
        self.model = model
        self.judge = judge
        self.generator = generator
        self.validator = validator
        self.layered_policy = layered_policy or LayeredEvaluationPolicy()
        self.auto_policy = auto_policy or AutoOptimizationPolicy()
        self.layered_policy.validate()
        self.auto_policy.validate()

    def _rehydrate_budget(self, run_dir: Path) -> LayeredBudget:
        task_calls = 0
        judge_calls = 0
        generator_calls = 0
        task_cost = 0.0
        judge_cost = 0.0
        generator_cost = 0.0
        for round_dir in sorted(run_dir.glob("round_*")):
            task_rows = _json_line_records(round_dir / "task_outputs.jsonl")
            stability_rows = _json_line_records(round_dir / "stability_outputs.jsonl")
            judge_rows = _json_line_records(round_dir / "judge_results.jsonl")
            task_calls += len(task_rows) + len(stability_rows)
            shared_calls = {
                str(row["shared_call_id"]) for row in judge_rows if row.get("shared_call_id")
            }
            judge_calls += len(shared_calls) + sum(
                not row.get("shared_call_id") for row in judge_rows
            )
            task_cost += sum(
                float(row.get("cost_usd", 0.0))
                for row in task_rows + stability_rows
            )
            seen_shared: set[str] = set()
            for row in judge_rows:
                shared_id = str(row.get("shared_call_id", ""))
                if shared_id and shared_id in seen_shared:
                    continue
                if shared_id:
                    seen_shared.add(shared_id)
                judge_cost += float(
                    row.get("judgment", {}).get("raw", {}).get(
                        "_promptos_judge_cost_usd", 0.0
                    )
                )
            if (round_dir / "prompts.json").exists():
                generator_calls += 1
                metadata = (
                    json.loads((round_dir / "generator_metadata.json").read_text(encoding="utf-8"))
                    if (round_dir / "generator_metadata.json").exists()
                    else {}
                )
                generator_cost += float(metadata.get("cost_usd") or 0.0)
        return LayeredBudget(
            self.layered_policy.task_model_max_calls,
            self.layered_policy.judge_model_max_calls,
            self.layered_policy.max_seconds,
            task_calls=task_calls,
            judge_calls=judge_calls,
            generator_calls=generator_calls,
            task_cost_usd=task_cost,
            judge_cost_usd=judge_cost,
            generator_cost_usd=generator_cost,
            started_at=time.monotonic(),
        )

    def optimize(
        self,
        initial_prompt: str,
        task: TaskSpec,
        signals: SignalSpec,
        samples: Iterable[Sample],
        runs_root: Path,
        run_id: str | None = None,
        task_config: dict[str, Any] | None = None,
    ) -> AutoOptimizationResult:
        frozen = list(samples)
        if not frozen:
            raise ValueError("At least one sample is required.")
        identifier = run_id or f"auto_{uuid.uuid4().hex[:12]}"
        run_dir = runs_root / identifier
        run_dir.mkdir(parents=True, exist_ok=True)
        identity = {
            "task": asdict(task),
            "initial_prompt": initial_prompt,
            "signal_spec": {"id": signals.id, "version": signals.version},
            "sample_hash": _sample_hash(frozen),
            "software_version": _SOFTWARE_VERSION,
            "layered_policy": asdict(self.layered_policy),
            "auto_policy": asdict(self.auto_policy),
        }
        identity = json.loads(json.dumps(identity, ensure_ascii=False))
        identity_path = run_dir / "run_identity.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
                raise ValueError("Resume auto-run identity does not match its frozen inputs.")
        else:
            _write_json(identity_path, identity)
        final_path = run_dir / "run.json"
        if final_path.exists():
            final = json.loads(final_path.read_text(encoding="utf-8"))
            # Existing pre-report runs are intentionally not retrofitted. Runs created
            # with report support already have a manifest and may be regenerated safely.
            if (run_dir / "report_manifest.json").exists():
                render_auto_report(
                    run_dir,
                    frozen,
                    completed=True,
                    stop_reason=final.get("stop_reason"),
                )
            return AutoOptimizationResult(
                identifier,
                run_dir,
                final["champion_prompt"],
                final["champion_status"],
                int(final["rounds_completed"]),
                final["stop_reason"],
                list(final["risk_memory_case_ids"]),
                final["budget"],
            )
        state_path = run_dir / "auto_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
        else:
            state = {
                "current_prompt": initial_prompt,
                "rounds_completed": 0,
                "no_improvement_rounds": 0,
                "risk_memory_case_ids": [],
                "round_history": [],
                "next_feedback": "",
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            _write_json(state_path, state)
        budget = self._rehydrate_budget(run_dir)
        stop_reason = "max_rounds"
        final_round_dir: Path | None = None
        optimizer = LayeredPromptOptimizer(
            self.model,
            self.judge,
            self.generator,
            self.validator,
            self.layered_policy,
        )
        while int(state["rounds_completed"]) < self.auto_policy.max_rounds:
            round_index = int(state["rounds_completed"])
            round_id = f"round_{round_index:02d}"
            result = optimizer.optimize(
                str(state["current_prompt"]),
                task,
                signals,
                frozen,
                run_dir,
                run_id=round_id,
                task_config=task_config,
                generation_feedback=str(state.get("next_feedback", "")) or None,
                risk_memory_case_ids=state["risk_memory_case_ids"],
                export_human_review=False,
                shared_budget=budget,
                comparative_judge=round_index >= 1,
                identical_calibration_k=10,
            )
            final_round_dir = result.run_dir
            summary = _failure_summary(result.run_dir)
            _write_json(result.run_dir / "failure_summary.json", summary)
            queue = json.loads(
                (result.run_dir / "judge_queue.json").read_text(encoding="utf-8")
            )
            memory = set(state["risk_memory_case_ids"])
            if self.auto_policy.retain_risk_memory:
                memory.update(queue["case_ids"])
            selected_prompt, accepted, decision = _improvement(
                result.comparison,
                str(state["current_prompt"]),
                self.auto_policy.minimum_improvement,
            )
            decision.update({
                "round": round_index,
                "selected_prompt": selected_prompt,
                "risk_memory_size": len(memory),
            })
            _write_json(result.run_dir / "round_decision.json", decision)
            state["current_prompt"] = selected_prompt
            state["rounds_completed"] = round_index + 1
            state["no_improvement_rounds"] = (
                0 if accepted else int(state["no_improvement_rounds"]) + 1
            )
            state["risk_memory_case_ids"] = sorted(memory)
            state["next_feedback"] = _generation_feedback(
                summary,
                state["risk_memory_case_ids"],
            )
            state["round_history"].append(decision)
            _write_json(state_path, state)
            render_auto_report(
                run_dir,
                frozen,
                completed=False,
            )
            if (
                round_index == 0
                and self.auto_policy.first_round_hard_failure_stop_count is not None
                and int(result.comparison[0]["hard_failure_count"])
                == self.auto_policy.first_round_hard_failure_stop_count
            ):
                stop_reason = "first_round_hard_failure_count_matched_stop_condition"
                break
            if int(state["no_improvement_rounds"]) >= self.auto_policy.stop_after_no_improvement:
                stop_reason = "no_material_improvement"
                break
        if final_round_dir is None:
            raise RuntimeError("Auto optimization completed no rounds.")
        samples_by_id = {sample.id: sample for sample in frozen}
        risks = _json_line_records(final_round_dir / "risk_scores.jsonl")
        checks = _json_line_records(final_round_dir / "deterministic_checks.jsonl")
        judge_rows = _json_line_records(final_round_dir / "judge_results.jsonl")
        task_rows = _json_line_records(final_round_dir / "task_outputs.jsonl")
        queue = json.loads(
            (final_round_dir / "judge_queue.json").read_text(encoding="utf-8")
        )
        optimizer._human_review(
            samples_by_id,
            risks,
            checks,
            judge_rows,
            task_rows,
            queue,
            run_dir / "human_review_top20.csv",
        )
        (run_dir / "champion_prompt.md").write_text(
            str(state["current_prompt"]).rstrip() + "\n",
            encoding="utf-8",
        )
        shutil.copyfile(
            final_round_dir / "prompt_comparison.json",
            run_dir / "final_prompt_comparison.json",
        )
        status = (
            "gold_validated"
            if any(sample.label_source == "gold_human" for sample in frozen)
            else "provisional_silver_or_unlabeled"
        )
        manifest = {
            "run_id": identifier,
            "mode": "layered_auto",
            "champion_prompt": state["current_prompt"],
            "champion_status": status,
            "rounds_completed": state["rounds_completed"],
            "stop_reason": stop_reason,
            "risk_memory_case_ids": state["risk_memory_case_ids"],
            "round_history": state["round_history"],
            "layered_policy": asdict(self.layered_policy),
            "auto_policy": asdict(self.auto_policy),
            "budget": asdict(budget),
            "started_at": state.get("started_at"),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        if task_config is not None:
            manifest["task_config"] = task_config
        _write_json(final_path, manifest)
        render_auto_report(
            run_dir,
            frozen,
            completed=True,
            stop_reason=stop_reason,
        )
        return AutoOptimizationResult(
            identifier,
            run_dir,
            str(state["current_prompt"]),
            status,
            int(state["rounds_completed"]),
            stop_reason,
            list(state["risk_memory_case_ids"]),
            asdict(budget),
        )
