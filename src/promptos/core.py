"""Provider-agnostic primitives for prompt optimization and evaluation."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol


LabelSource = str  # "unlabeled" | "silver_auto" | "gold_human"


@dataclass(frozen=True)
class Sample:
    """An arbitrary task input with optional reference output.

    `inputs` deliberately has no prescribed field names: it may contain an
    answer, source documents, a user query, a table row, or tool output.
    """

    id: str
    inputs: dict[str, Any]
    expected: Any | None = None
    label_source: LabelSource = "unlabeled"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    input_fields: list[str]
    output_description: str
    output_schema: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HardConstraint:
    name: str
    description: str


@dataclass(frozen=True)
class SoftSignal:
    name: str
    criterion: str
    weight: float


@dataclass(frozen=True)
class SignalSpec:
    """An inspectable, immutable quality contract for one experiment."""

    id: str
    version: int
    acceptance_criteria: str
    hard_constraints: list[HardConstraint]
    soft_signals: list[SoftSignal]
    status: str = "draft"  # draft | approved
    created_by: str = "signal_compiler"

    def validate(self) -> None:
        if not self.acceptance_criteria.strip():
            raise ValueError("SignalSpec requires acceptance criteria.")
        if not self.soft_signals:
            raise ValueError("SignalSpec requires at least one soft signal.")
        total = sum(signal.weight for signal in self.soft_signals)
        if total <= 0:
            raise ValueError("Signal weights must sum to a positive value.")
        if any(signal.weight < 0 for signal in self.soft_signals):
            raise ValueError("Signal weights cannot be negative.")

    def approved(self) -> "SignalSpec":
        self.validate()
        return SignalSpec(self.id, self.version, self.acceptance_criteria,
                          self.hard_constraints, self.soft_signals, "approved", self.created_by)


@dataclass(frozen=True)
class ModelResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


class TaskModel(Protocol):
    def generate(self, prompt: str, inputs: dict[str, Any]) -> ModelResponse: ...


class Judge(Protocol):
    def score(self, signal_spec: SignalSpec, sample: Sample, output: str) -> "Judgment": ...


class PromptGenerator(Protocol):
    def propose(self, prompt: str, task: TaskSpec, signal_spec: SignalSpec, feedback: str) -> Iterable[str]: ...


@dataclass(frozen=True)
class Judgment:
    score: float
    signal_scores: dict[str, float]
    hard_failures: list[str]
    rationale: str
    confidence: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SampleResult:
    sample_id: str
    output: str
    judgment: Judgment
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass(frozen=True)
class Evaluation:
    prompt: str
    score: float
    results: list[SampleResult]


@dataclass
class Budget:
    max_calls: int = 100
    max_cost_usd: float = 10.0
    max_seconds: float = 900.0
    calls: int = 0
    cost_usd: float = 0.0
    started_at: float = field(default_factory=time.monotonic)

    def check(self) -> None:
        if self.calls >= self.max_calls:
            raise RuntimeError("Budget exhausted: maximum model calls reached.")
        if self.cost_usd >= self.max_cost_usd:
            raise RuntimeError("Budget exhausted: maximum cost reached.")
        if time.monotonic() - self.started_at >= self.max_seconds:
            raise RuntimeError("Budget exhausted: maximum runtime reached.")

    def record(self, response: ModelResponse) -> None:
        self.calls += 1
        self.cost_usd += response.cost_usd


@dataclass(frozen=True)
class OptimizationResult:
    run_id: str
    champion_prompt: str
    champion_score: float
    champion_status: str
    evaluations: list[Evaluation]
    budget: dict[str, Any]


class SignalEvaluator:
    """Applies a user-approved SignalSpec through a judge implementation."""

    def __init__(self, judge: Judge):
        self.judge = judge

    def evaluate(self, signal_spec: SignalSpec, sample: Sample, output: str) -> Judgment:
        signal_spec.validate()
        judgment = self.judge.score(signal_spec, sample, output)
        if not 0 <= judgment.score <= 1:
            raise ValueError("Judges must return scores between 0 and 1.")
        return judgment


class PromptOptimizer:
    """Budget-bounded greedy search. Approval prevents accidental metric drift."""

    def __init__(self, model: TaskModel, evaluator: SignalEvaluator, generator: PromptGenerator):
        self.model, self.evaluator, self.generator = model, evaluator, generator

    @staticmethod
    def _record_auxiliary(component: Any, budget: Budget) -> None:
        """Count a judge/generator model call when the adapter exposes it."""
        model = getattr(component, "model", None)
        response = getattr(model, "last_response", None)
        if isinstance(response, ModelResponse):
            budget.record(response)
            model.last_response = None

    def evaluate(self, prompt: str, samples: Iterable[Sample], signals: SignalSpec, budget: Budget) -> Evaluation:
        results: list[SampleResult] = []
        for sample in samples:
            budget.check()
            response = self.model.generate(prompt, sample.inputs)
            budget.record(response)
            budget.check()
            judgment = self.evaluator.evaluate(signals, sample, response.text)
            self._record_auxiliary(self.evaluator.judge, budget)
            results.append(SampleResult(sample.id, response.text, judgment, response.model,
                                        response.input_tokens, response.output_tokens, response.cost_usd))
        if not results:
            raise ValueError("At least one evaluation sample is required.")
        return Evaluation(prompt, sum(item.judgment.score for item in results) / len(results), results)

    def optimize(self, initial_prompt: str, task: TaskSpec, signals: SignalSpec, samples: Iterable[Sample], budget: Budget, rounds: int = 3) -> OptimizationResult:
        if signals.status != "approved":
            raise ValueError("Only an approved SignalSpec can drive an optimization run.")
        frozen = list(samples)
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        current = self.evaluate(initial_prompt, frozen, signals, budget)
        history = [current]
        seen = {initial_prompt.strip()}
        for _ in range(rounds):
            feedback = f"Current aggregate signal score is {current.score:.3f}. Improve hard-constraint compliance and weak signals."
            budget.check()
            candidates = [item.strip() for item in self.generator.propose(current.prompt, task, signals, feedback) if item.strip() not in seen]
            self._record_auxiliary(self.generator, budget)
            if not candidates:
                break
            contenders = [self.evaluate(candidate, frozen, signals, budget) for candidate in candidates]
            history.extend(contenders)
            seen.update(item.prompt.strip() for item in contenders)
            best = max(contenders, key=lambda item: item.score)
            if best.score <= current.score:
                break
            current = best
        status = "gold_validated" if any(sample.label_source == "gold_human" for sample in frozen) else "provisional_silver_or_unlabeled"
        return OptimizationResult(run_id, current.prompt, current.score, status, history, asdict(budget))


class RunStore:
    """Append-only local experiment store; secrets are never accepted as fields."""

    def __init__(self, root: Path):
        self.root = root

    @staticmethod
    def dataset_hash(samples: Iterable[Sample]) -> str:
        canonical = "\n".join(json.dumps(asdict(sample), ensure_ascii=False, sort_keys=True) for sample in samples)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def save(self, result: OptimizationResult, task: TaskSpec, signals: SignalSpec, samples: list[Sample], task_config: dict[str, Any] | None = None) -> Path:
        run_dir = self.root / result.run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest = {
            "run_id": result.run_id, "task": asdict(task), "signal_spec": asdict(signals),
            "dataset_hash": self.dataset_hash(samples), "sample_count": len(samples),
            "label_sources": {source: sum(sample.label_source == source for sample in samples)
                              for source in ("unlabeled", "silver_auto", "gold_human")},
            "champion_prompt": result.champion_prompt, "champion_score": result.champion_score,
            "champion_status": result.champion_status,
            "budget": result.budget,
        }
        if task_config is not None:
            manifest["task_config"] = task_config
        (run_dir / "run.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with (run_dir / "evaluations.jsonl").open("w", encoding="utf-8") as file:
            for evaluation in result.evaluations:
                file.write(json.dumps(asdict(evaluation), ensure_ascii=False) + "\n")
        (run_dir / "report.md").write_text(
            f"# Experiment {result.run_id}\n\n- Champion score: {result.champion_score:.4f}\n- Champion status: {result.champion_status}\n- Signal spec: {signals.id} v{signals.version}\n- Dataset hash: `{manifest['dataset_hash']}`\n",
            encoding="utf-8",
        )
        return run_dir
