"""CLI for SignalSpec drafting, approval, review queues, and optimization."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .adapters import LLMJudge, LLMPromptGenerator, LLMSignalCompiler, LLMSilverAnnotator, OpenAICompatibleModel, ReferenceJudge, RuleBasedDemoModel, TemplatePromptGenerator
from .core import Budget, HardConstraint, OptimizationResult, PromptOptimizer, RunStore, Sample, SignalEvaluator, SignalSpec, SoftSignal, TaskSpec
from .datasets import split_gold_samples, write_split
from .config import load_task_config
from .auto import AutoOptimizationPolicy, AutoPromptOptimizer
from .layered import JsonOutputValidator, LayeredEvaluationPolicy, LayeredPromptOptimizer
from .provenance import apply_annotations, export_review_csv, import_review_csv, load_annotations, select_review_cases, write_annotations, write_samples


def _json(path: Path, value: Any | None = None) -> Any:
    if value is None:
        return json.loads(path.read_text(encoding="utf-8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_samples(path: Path, input_fields: list[str], expected_field: str | None = None) -> list[Sample]:
    if path.suffix.lower() == ".csv":
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
    else:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    samples = []
    for index, row in enumerate(rows):
        canonical_inputs = row.get("inputs") if isinstance(row.get("inputs"), dict) else None
        inputs = canonical_inputs or {field: row[field] for field in input_fields if field in row}
        if any(field not in inputs for field in input_fields):
            raise ValueError(f"Record {index} is missing one of {input_fields}.")
        expected = row.get("expected") if canonical_inputs is not None else (row.get(expected_field) if expected_field else None)
        label_source = row.get("label_source", "gold_human" if expected is not None else "unlabeled")
        samples.append(Sample(str(row.get("id", index)), inputs, expected, label_source, row.get("metadata", {})))
    return samples


def as_signals(value: dict[str, Any]) -> SignalSpec:
    return SignalSpec(value["id"], int(value["version"]), value["acceptance_criteria"],
        [HardConstraint(**item) for item in value.get("hard_constraints", [])],
        [SoftSignal(**item) for item in value["soft_signals"]], value.get("status", "draft"), value.get("created_by", "unknown"))


def draft_signals(args: argparse.Namespace) -> int:
    model = OpenAICompatibleModel(args.model) if args.model else None
    spec = LLMSignalCompiler(model).compile(args.acceptance, args.id, args.version)
    _json(Path(args.output), asdict(spec))
    print(f"Draft SignalSpec saved to {args.output}. Review and run approve-signals before optimization.")
    return 0


def approve_signals(args: argparse.Namespace) -> int:
    path = Path(args.signals)
    spec = as_signals(_json(path)).approved()
    _json(path, asdict(spec))
    print(f"Approved SignalSpec {spec.id} v{spec.version}.")
    return 0


def build_review_queue(args: argparse.Namespace) -> int:
    samples = load_samples(Path(args.dataset), args.inputs.split(","), args.expected_field)
    queue = select_review_cases(samples, args.limit)
    _json(Path(args.output), {"queue": queue, "policy": "v1 requires independent judge or human review; auto labels are silver, never gold."})
    print(f"Wrote {len(queue)} review candidates to {args.output}.")
    return 0


def export_review(args: argparse.Namespace) -> int:
    samples = load_samples(Path(args.dataset), args.inputs.split(","), args.expected_field)
    queue_data = _json(Path(args.queue))
    queue = queue_data.get("queue", queue_data) if isinstance(queue_data, dict) else queue_data
    annotations = load_annotations(Path(args.annotations)) if args.annotations else []
    count = export_review_csv(samples, queue, annotations, Path(args.output))
    print(f"Exported {count} review rows to {args.output}. Complete decision/reviewer fields, then run import-review.")
    return 0


def import_review(args: argparse.Namespace) -> int:
    annotations, rejected = import_review_csv(Path(args.review_csv))
    write_annotations(Path(args.output), annotations)
    if rejected:
        _json(Path(args.output).with_suffix(".rejected.json"), {"rejected": rejected})
    print(f"Imported {len(annotations)} gold_human annotations to {args.output}; rejected={len(rejected)}.")
    return 0


def apply_review(args: argparse.Namespace) -> int:
    samples = load_samples(Path(args.dataset), args.inputs.split(","), args.expected_field)
    annotations = load_annotations(Path(args.annotations))
    write_samples(Path(args.output), apply_annotations(samples, annotations))
    print(f"Wrote canonical reviewed dataset to {args.output}.")
    return 0


def split_dataset(args: argparse.Namespace) -> int:
    samples = load_samples(Path(args.dataset), args.inputs.split(","), args.expected_field)
    splits = split_gold_samples(samples, args.seed, args.optimize_ratio, args.validation_ratio)
    directory = write_split(Path(args.output), splits, args.seed)
    print(json.dumps({"split_dir": str(directory), "counts": {name: len(items) for name, items in splits.items()}}, ensure_ascii=False))
    return 0


def annotate_silver(args: argparse.Namespace) -> int:
    signals = as_signals(_json(Path(args.signals)))
    if signals.status != "approved":
        raise ValueError("Approve the SignalSpec before generating silver labels.")
    samples = load_samples(Path(args.dataset), args.inputs.split(","), args.expected_field)
    model = OpenAICompatibleModel(args.model)
    annotator = LLMSilverAnnotator(model)
    targets = [item for item in samples if item.expected is None][:args.limit]
    annotations, errors = [], []

    def label(sample: Sample) -> dict[str, Any]:
        value, rationale = annotator.annotate(signals, sample)
        return {"sample_id": sample.id, "value": value, "source": "silver_auto", "judge_run_id": args.judge_run_id,
                "rationale": rationale, "judge_model": args.model}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(label, sample): sample.id for sample in targets}
        for future in as_completed(futures):
            try:
                annotations.append(future.result())
            except Exception as error:
                errors.append({"sample_id": futures[future], "error": str(error)})
    annotations.sort(key=lambda item: item["sample_id"])
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in annotations), encoding="utf-8")
    if errors:
        _json(output.with_suffix(".errors.json"), {"errors": errors})
    print(f"Wrote {len(annotations)} silver_auto annotations to {output}; failures={len(errors)}. They require human review before becoming gold_human.")
    return 0


def optimize(args: argparse.Namespace) -> int:
    signals = as_signals(_json(Path(args.signals)))
    samples = load_samples(Path(args.dataset), args.inputs.split(","), args.expected_field)
    if args.annotations:
        samples = apply_annotations(samples, load_annotations(Path(args.annotations)))
    task = TaskSpec(args.task, args.inputs.split(","), args.output_description, getattr(args, "output_schema", {}))
    budget = Budget(args.max_calls, args.max_cost, args.max_seconds)
    if args.model:
        base_url = getattr(args, "base_url", None)
        task_model = OpenAICompatibleModel(args.model, base_url=base_url)
        judge_model = OpenAICompatibleModel(args.judge_model or args.model, base_url=base_url)
        generator = LLMPromptGenerator(OpenAICompatibleModel(args.generator_model or args.model, base_url=base_url, temperature=0.7))
        evaluator = SignalEvaluator(LLMJudge(judge_model))
    else:
        task_model, evaluator, generator = RuleBasedDemoModel(), SignalEvaluator(ReferenceJudge()), TemplatePromptGenerator()
    optimizer = PromptOptimizer(task_model, evaluator, generator)
    result = optimizer.optimize(args.prompt, task, signals, samples, budget, args.rounds)
    run_dir = RunStore(Path(args.runs)).save(result, task, signals, samples, getattr(args, "config_snapshot", None))
    print(json.dumps({"run_id": result.run_id, "champion_score": result.champion_score,
                      "champion_status": result.champion_status, "run_dir": str(run_dir)}, ensure_ascii=False))
    return 0


def optimize_layered(args: argparse.Namespace) -> int:
    signals = as_signals(_json(Path(args.signals)))
    samples = load_samples(Path(args.dataset), args.inputs.split(","), args.expected_field)
    if args.annotations:
        samples = apply_annotations(samples, load_annotations(Path(args.annotations)))
    task = TaskSpec(args.task, args.inputs.split(","), args.output_description, getattr(args, "output_schema", {}))
    prompt = args.prompt
    prompt_file = getattr(args, "prompt_file", None)
    if prompt_file:
        if prompt:
            raise ValueError("Use either --prompt or --prompt-file, not both.")
        prompt = Path(prompt_file).read_text(encoding="utf-8")
    validator = JsonOutputValidator(task.output_schema)
    if getattr(args, "plugin", None):
        if args.plugin != "finance_classification":
            raise ValueError(f"Unsupported plugin: {args.plugin}")
        if not args.taxonomy:
            raise ValueError("finance_classification requires --taxonomy.")
        from .plugins.finance_classification import (
            FinanceOutputValidator,
            FinanceTaxonomy,
            default_prompt,
            task_spec as finance_task_spec,
        )
        taxonomy = FinanceTaxonomy.load(Path(args.taxonomy))
        task = finance_task_spec()
        validator = FinanceOutputValidator(taxonomy)
        prompt = prompt or default_prompt(taxonomy)
    if not prompt:
        raise ValueError("Layered optimization requires --prompt unless a plugin supplies one.")
    if not args.model:
        raise ValueError("Layered optimization requires an explicit task model.")
    base_url = getattr(args, "base_url", None)
    task_model = OpenAICompatibleModel(args.model, base_url=base_url)
    judge = LLMJudge(OpenAICompatibleModel(args.judge_model or args.model, base_url=base_url))
    generator = LLMPromptGenerator(
        OpenAICompatibleModel(args.generator_model or args.model, base_url=base_url, temperature=0.7)
    )
    fixed_sample_kinds = tuple(
        kind.strip()
        for kind in args.fixed_sample_kinds.split(",")
        if kind.strip()
    )
    policy = LayeredEvaluationPolicy(
        max_candidates=args.max_candidates,
        fixed_sample_kinds=fixed_sample_kinds,
        dynamic_top_k=args.dynamic_top_k,
        judge_max_cases_per_prompt=args.judge_max_cases_per_prompt,
        human_review_top_k=args.human_review_top_k,
        task_model_max_calls=args.task_model_max_calls,
        judge_model_max_calls=args.judge_model_max_calls,
        max_seconds=args.max_seconds,
    )
    result = LayeredPromptOptimizer(task_model, judge, generator, validator, policy).optimize(
        prompt,
        task,
        signals,
        samples,
        Path(args.runs),
        run_id=args.resume_run,
        task_config=getattr(args, "config_snapshot", None),
    )
    print(json.dumps({
        "run_id": result.run_id,
        "champion_status": result.champion_status,
        "judge_case_count": len(result.judge_case_ids),
        "run_dir": str(result.run_dir),
    }, ensure_ascii=False))
    return 0


def optimize_auto(args: argparse.Namespace) -> int:
    signals = as_signals(_json(Path(args.signals)))
    samples = load_samples(Path(args.dataset), args.inputs.split(","), args.expected_field)
    if args.annotations:
        samples = apply_annotations(samples, load_annotations(Path(args.annotations)))
    task = TaskSpec(
        args.task,
        args.inputs.split(","),
        args.output_description,
        getattr(args, "output_schema", {}),
    )
    prompt = args.prompt
    prompt_file = getattr(args, "prompt_file", None)
    if prompt_file:
        if prompt:
            raise ValueError("Use either --prompt or --prompt-file, not both.")
        prompt = Path(prompt_file).read_text(encoding="utf-8")
    validator = JsonOutputValidator(task.output_schema)
    if getattr(args, "plugin", None):
        if args.plugin != "finance_classification":
            raise ValueError(f"Unsupported plugin: {args.plugin}")
        if not args.taxonomy:
            raise ValueError("finance_classification requires --taxonomy.")
        from .plugins.finance_classification import (
            FinanceOutputValidator,
            FinanceTaxonomy,
            default_prompt,
            task_spec as finance_task_spec,
        )
        taxonomy = FinanceTaxonomy.load(Path(args.taxonomy))
        task = finance_task_spec()
        validator = FinanceOutputValidator(taxonomy)
        prompt = prompt or default_prompt(taxonomy)
    if not prompt:
        raise ValueError("Auto optimization requires --prompt unless a plugin supplies one.")
    if not args.model:
        raise ValueError("Auto optimization requires an explicit task model.")
    base_url = getattr(args, "base_url", None)
    task_model = OpenAICompatibleModel(args.model, base_url=base_url)
    judge = LLMJudge(
        OpenAICompatibleModel(args.judge_model or args.model, base_url=base_url)
    )
    generator = LLMPromptGenerator(
        OpenAICompatibleModel(
            args.generator_model or args.model,
            base_url=base_url,
            temperature=0.7,
        )
    )
    fixed_sample_kinds = tuple(
        kind.strip()
        for kind in args.fixed_sample_kinds.split(",")
        if kind.strip()
    )
    layered_policy = LayeredEvaluationPolicy(
        max_candidates=args.max_candidates,
        fixed_sample_kinds=fixed_sample_kinds,
        dynamic_top_k=args.dynamic_top_k,
        judge_max_cases_per_prompt=args.judge_max_cases_per_prompt,
        human_review_top_k=args.human_review_top_k,
        task_model_max_calls=args.task_model_max_calls,
        judge_model_max_calls=args.judge_model_max_calls,
        max_seconds=args.max_seconds,
    )
    auto_policy = AutoOptimizationPolicy(
        max_rounds=args.max_rounds,
        stop_after_no_improvement=args.stop_after_no_improvement,
        minimum_improvement=args.minimum_improvement,
        retain_risk_memory=args.retain_risk_memory,
    )
    result = AutoPromptOptimizer(
        task_model,
        judge,
        generator,
        validator,
        layered_policy,
        auto_policy,
    ).optimize(
        prompt,
        task,
        signals,
        samples,
        Path(args.runs),
        run_id=args.resume_run,
        task_config=getattr(args, "config_snapshot", None),
    )
    print(json.dumps({
        "run_id": result.run_id,
        "champion_status": result.champion_status,
        "rounds_completed": result.rounds_completed,
        "stop_reason": result.stop_reason,
        "risk_memory_case_count": len(result.risk_memory_case_ids),
        "run_dir": str(result.run_dir),
    }, ensure_ascii=False))
    return 0


def run_config(args: argparse.Namespace) -> int:
    config = load_task_config(Path(args.config))
    if config.evaluation.mode in {"layered", "layered_auto"}:
        values = {
            "dataset": str(config.dataset.path), "inputs": ",".join(config.input_fields),
            "expected_field": config.dataset.expected_field,
            "annotations": str(config.dataset.annotations) if config.dataset.annotations else None,
            "signals": str(config.signal_spec), "prompt": config.optimization.initial_prompt or None,
            "prompt_file": (
                str(config.optimization.initial_prompt_path)
                if config.optimization.initial_prompt_path else None
            ),
            "task": config.name, "output_description": config.output_description,
            "output_schema": config.output_schema, "model": config.models.task_model,
            "judge_model": config.models.judge_model, "generator_model": config.models.generator_model,
            "base_url": config.models.base_url, "plugin": config.plugin.name,
            "taxonomy": str(config.plugin.taxonomy_path) if config.plugin.taxonomy_path else None,
            "max_candidates": config.evaluation.max_candidates,
            "fixed_sample_kinds": ",".join(config.evaluation.fixed_sample_kinds),
            "dynamic_top_k": config.evaluation.dynamic_top_k,
            "judge_max_cases_per_prompt": config.evaluation.judge_max_cases_per_prompt,
            "human_review_top_k": config.evaluation.human_review_top_k,
            "task_model_max_calls": config.evaluation.task_model_max_calls,
            "judge_model_max_calls": config.evaluation.judge_model_max_calls,
            "max_seconds": config.evaluation.max_seconds, "runs": str(config.runs_dir),
            "max_rounds": config.evaluation.max_rounds,
            "stop_after_no_improvement": config.evaluation.stop_after_no_improvement,
            "minimum_improvement": config.evaluation.minimum_improvement,
            "retain_risk_memory": config.evaluation.retain_risk_memory,
            "resume_run": getattr(args, "resume_run", None),
            "config_snapshot": json.loads(config.path.read_text(encoding="utf-8")),
        }
        if args.prompt:
            values["prompt"] = args.prompt
            values["prompt_file"] = None
        if args.model:
            values["model"] = args.model
        function = optimize_auto if config.evaluation.mode == "layered_auto" else optimize_layered
        return function(argparse.Namespace(**values))
    values = {
        "dataset": str(config.dataset.path), "inputs": ",".join(config.input_fields),
        "expected_field": config.dataset.expected_field, "annotations": str(config.dataset.annotations) if config.dataset.annotations else None,
        "signals": str(config.signal_spec),
        "prompt": (
            args.prompt
            or config.optimization.initial_prompt
            or (
                config.optimization.initial_prompt_path.read_text(encoding="utf-8")
                if config.optimization.initial_prompt_path else ""
            )
        ),
        "task": config.name,
        "output_description": config.output_description, "output_schema": config.output_schema,
        "model": config.models.task_model, "judge_model": config.models.judge_model,
        "generator_model": config.models.generator_model, "base_url": config.models.base_url,
        "rounds": config.optimization.rounds, "max_calls": config.optimization.max_calls,
        "max_cost": config.optimization.max_cost_usd, "max_seconds": config.optimization.max_seconds,
        "runs": str(config.runs_dir), "config_snapshot": json.loads(config.path.read_text(encoding="utf-8")),
    }
    # Explicit CLI prompt/model overrides are permitted; config remains recorded in the run metadata via task fields.
    if args.model:
        values["model"] = args.model
    return optimize(argparse.Namespace(**values))


def validate_config(args: argparse.Namespace) -> int:
    config = load_task_config(Path(args.config))
    print(json.dumps({"task": config.name, "dataset": str(config.dataset.path), "signals": str(config.signal_spec),
                      "runs_dir": str(config.runs_dir), "model": config.models.task_model,
                      "evaluation_mode": config.evaluation.mode, "plugin": config.plugin.name,
                      "taxonomy": str(config.plugin.taxonomy_path) if config.plugin.taxonomy_path else None,
                      "credentials": "environment-only"}, ensure_ascii=False, indent=2))
    return 0


def final_evaluate(args: argparse.Namespace) -> int:
    """Run a locked final set once; no candidate generation is permitted here."""
    signals = as_signals(_json(Path(args.signals)))
    if signals.status != "approved":
        raise ValueError("Only approved SignalSpecs may evaluate a final test set.")
    samples = load_samples(Path(args.dataset), args.inputs.split(","), args.expected_field)
    if any(sample.label_source != "gold_human" for sample in samples):
        raise ValueError("final-evaluate only accepts a frozen gold_human final test dataset.")
    lock = _json(Path(args.split_manifest))
    digest = RunStore.dataset_hash(samples)
    if not lock.get("locked") or lock.get("splits", {}).get("final_test", {}).get("hash") != digest:
        raise ValueError("Dataset does not match the locked final_test hash in split_manifest.json.")
    key = hashlib.sha256(f"{digest}:{signals.id}:{signals.version}:{args.prompt}".encode()).hexdigest()
    registry_path = Path(args.runs) / "final_test_registry.json"
    registry = _json(registry_path) if registry_path.exists() else {"completed": {}}
    if key in registry["completed"]:
        raise RuntimeError("This prompt/signal/final-test combination has already been evaluated; final tests are one-shot.")
    task = TaskSpec(args.task, args.inputs.split(","), args.output_description)
    budget = Budget(args.max_calls, args.max_cost, args.max_seconds)
    if args.model:
        model = OpenAICompatibleModel(args.model)
        evaluator = SignalEvaluator(LLMJudge(OpenAICompatibleModel(args.judge_model or args.model)))
    else:
        model, evaluator = RuleBasedDemoModel(), SignalEvaluator(ReferenceJudge())
    evaluation = PromptOptimizer(model, evaluator, TemplatePromptGenerator()).evaluate(args.prompt, samples, signals, budget)
    result = OptimizationResult(f"final_{uuid.uuid4().hex[:12]}", args.prompt, evaluation.score, "gold_final_test", [evaluation], budget.__dict__)
    run_dir = RunStore(Path(args.runs)).save(result, task, signals, samples)
    registry["completed"][key] = {"run_id": result.run_id, "dataset_hash": digest, "signal_spec": f"{signals.id}:{signals.version}"}
    _json(registry_path, registry)
    print(json.dumps({"run_id": result.run_id, "final_score": evaluation.score, "run_dir": str(run_dir)}, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="PromptOS: local, auditable prompt optimization.")
    sub = parser.add_subparsers(dest="command", required=True)
    draft = sub.add_parser("draft-signals", help="Turn acceptance criteria into a reviewable SignalSpec draft")
    draft.add_argument("--acceptance", required=True); draft.add_argument("--output", required=True); draft.add_argument("--id", default="signals"); draft.add_argument("--version", type=int, default=1); draft.add_argument("--model", help="OpenAI-compatible model used to draft the rubric"); draft.set_defaults(func=draft_signals)
    approve = sub.add_parser("approve-signals", help="Explicitly approve a reviewed SignalSpec")
    approve.add_argument("--signals", required=True); approve.set_defaults(func=approve_signals)
    queue = sub.add_parser("review-queue", help="Create a conservative unlabeled-data review queue")
    queue.add_argument("--dataset", required=True); queue.add_argument("--inputs", required=True); queue.add_argument("--expected-field"); queue.add_argument("--limit", type=int, default=50); queue.add_argument("--output", required=True); queue.set_defaults(func=build_review_queue)
    review_export = sub.add_parser("export-review", help="Export queue and optional silver proposals as a human review CSV")
    review_export.add_argument("--dataset", required=True); review_export.add_argument("--inputs", required=True); review_export.add_argument("--expected-field"); review_export.add_argument("--queue", required=True); review_export.add_argument("--annotations", help="silver_auto JSONL proposals"); review_export.add_argument("--output", required=True); review_export.set_defaults(func=export_review)
    review_import = sub.add_parser("import-review", help="Validate human review CSV and emit gold_human JSONL annotations")
    review_import.add_argument("--review-csv", required=True); review_import.add_argument("--output", required=True); review_import.set_defaults(func=import_review)
    review_apply = sub.add_parser("apply-review", help="Apply validated annotations to a canonical JSONL dataset")
    review_apply.add_argument("--dataset", required=True); review_apply.add_argument("--inputs", required=True); review_apply.add_argument("--expected-field"); review_apply.add_argument("--annotations", required=True); review_apply.add_argument("--output", required=True); review_apply.set_defaults(func=apply_review)
    split = sub.add_parser("split-dataset", help="Deterministically freeze gold optimize/validation/final-test sets")
    split.add_argument("--dataset", required=True); split.add_argument("--inputs", required=True); split.add_argument("--expected-field", required=True); split.add_argument("--seed", default="promptos-v1"); split.add_argument("--optimize-ratio", type=float, default=0.6); split.add_argument("--validation-ratio", type=float, default=0.2); split.add_argument("--output", required=True); split.set_defaults(func=split_dataset)
    silver = sub.add_parser("annotate-silver", help="Use an independent judge to create reviewable silver labels")
    silver.add_argument("--dataset", required=True); silver.add_argument("--inputs", required=True); silver.add_argument("--expected-field"); silver.add_argument("--signals", required=True); silver.add_argument("--model", required=True); silver.add_argument("--judge-run-id", required=True); silver.add_argument("--limit", type=int, default=50); silver.add_argument("--workers", type=int, default=3, help="parallel judge calls; reduce after 429 responses"); silver.add_argument("--output", required=True); silver.set_defaults(func=annotate_silver)
    config_check = sub.add_parser("validate-config", help="Validate and show resolved task.json paths without running models")
    config_check.add_argument("--config", required=True); config_check.set_defaults(func=validate_config)
    config_run = sub.add_parser("run-config", help="Run an optimization experiment from a versioned task.json")
    config_run.add_argument("--config", required=True); config_run.add_argument("--prompt", help="Optional one-run prompt override"); config_run.add_argument("--model", help="Optional one-run task model override"); config_run.add_argument("--resume-run", help="Resume a layered run ID"); config_run.set_defaults(func=run_config)
    run = sub.add_parser("optimize", help="Legacy mode: judge every sample in a budget-bounded experiment")
    run.add_argument("--dataset", required=True); run.add_argument("--inputs", required=True); run.add_argument("--expected-field"); run.add_argument("--annotations", help="JSONL annotations; gold_human requires reviewer, silver_auto requires judge_run_id"); run.add_argument("--signals", required=True); run.add_argument("--prompt", required=True); run.add_argument("--task", default="generic"); run.add_argument("--output-description", default="Return the requested task result."); run.add_argument("--model", help="OpenAI-compatible task model; omit for local demo"); run.add_argument("--judge-model"); run.add_argument("--generator-model"); run.add_argument("--base-url", help="OpenAI-compatible endpoint; credentials remain in environment"); run.add_argument("--rounds", type=int, default=3); run.add_argument("--max-calls", type=int, default=100); run.add_argument("--max-cost", type=float, default=10.0); run.add_argument("--max-seconds", type=float, default=900.0); run.add_argument("--runs", default="runs"); run.set_defaults(func=optimize)
    layered = sub.add_parser("optimize-layered", help="Cost-aware fixed-plus-dynamic risk funnel")
    layered.add_argument("--dataset", required=True); layered.add_argument("--inputs", required=True)
    layered.add_argument("--expected-field"); layered.add_argument("--annotations")
    layered.add_argument("--signals", required=True); layered.add_argument("--prompt")
    layered.add_argument("--prompt-file", help="Read the initial prompt verbatim from a UTF-8 file")
    layered.add_argument("--task", default="generic")
    layered.add_argument("--output-description", default="Return the requested task result.")
    layered.add_argument("--model", required=True); layered.add_argument("--judge-model")
    layered.add_argument("--generator-model"); layered.add_argument("--base-url")
    layered.add_argument("--plugin", choices=["finance_classification"]); layered.add_argument("--taxonomy")
    layered.add_argument("--max-candidates", type=int, default=2)
    layered.add_argument("--fixed-sample-kinds", default="boundary_probe")
    layered.add_argument("--dynamic-top-k", type=int, default=30)
    layered.add_argument("--judge-max-cases-per-prompt", type=int, default=80)
    layered.add_argument("--human-review-top-k", type=int, default=20)
    layered.add_argument("--task-model-max-calls", type=int, default=2000)
    layered.add_argument("--judge-model-max-calls", type=int, default=240)
    layered.add_argument("--max-seconds", type=float, default=7200.0)
    layered.add_argument("--runs", default="runs"); layered.add_argument("--resume-run")
    layered.set_defaults(func=optimize_layered)
    auto = sub.add_parser("optimize-auto", help="Unattended multi-round layered prompt optimization")
    auto.add_argument("--dataset", required=True); auto.add_argument("--inputs", required=True)
    auto.add_argument("--expected-field"); auto.add_argument("--annotations")
    auto.add_argument("--signals", required=True); auto.add_argument("--prompt")
    auto.add_argument("--prompt-file", help="Read the initial prompt verbatim from a UTF-8 file")
    auto.add_argument("--task", default="generic")
    auto.add_argument("--output-description", default="Return the requested task result.")
    auto.add_argument("--model", required=True); auto.add_argument("--judge-model")
    auto.add_argument("--generator-model"); auto.add_argument("--base-url")
    auto.add_argument("--plugin", choices=["finance_classification"]); auto.add_argument("--taxonomy")
    auto.add_argument("--max-candidates", type=int, default=2)
    auto.add_argument("--fixed-sample-kinds", default="boundary_probe")
    auto.add_argument("--dynamic-top-k", type=int, default=30)
    auto.add_argument("--judge-max-cases-per-prompt", type=int, default=80)
    auto.add_argument("--human-review-top-k", type=int, default=20)
    auto.add_argument("--task-model-max-calls", type=int, default=10000)
    auto.add_argument("--judge-model-max-calls", type=int, default=1200)
    auto.add_argument("--max-seconds", type=float, default=28800.0)
    auto.add_argument("--max-rounds", type=int, default=5)
    auto.add_argument("--stop-after-no-improvement", type=int, default=2)
    auto.add_argument("--minimum-improvement", type=float, default=0.01)
    auto.add_argument(
        "--retain-risk-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    auto.add_argument("--runs", default="runs"); auto.add_argument("--resume-run")
    auto.set_defaults(func=optimize_auto)
    final = sub.add_parser("final-evaluate", help="Evaluate a frozen final_test set exactly once")
    final.add_argument("--dataset", required=True); final.add_argument("--split-manifest", required=True); final.add_argument("--inputs", required=True); final.add_argument("--expected-field"); final.add_argument("--signals", required=True); final.add_argument("--prompt", required=True); final.add_argument("--task", default="generic"); final.add_argument("--output-description", default="Return the requested task result."); final.add_argument("--model"); final.add_argument("--judge-model"); final.add_argument("--max-calls", type=int, default=100); final.add_argument("--max-cost", type=float, default=10.0); final.add_argument("--max-seconds", type=float, default=900.0); final.add_argument("--runs", default="runs"); final.set_defaults(func=final_evaluate)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
