"""Deterministic, zero-LLM reporting for layered_auto runs."""
from __future__ import annotations

import csv
import difflib
import hashlib
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .core import Sample

REPORT_SCHEMA_VERSION = 1


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            if line.strip():
                rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _label(parsed: Any) -> tuple[str, str]:
    if not isinstance(parsed, dict):
        return "", ""
    return str(parsed.get("L2_id", parsed.get("label", ""))), str(parsed.get("L3_id", ""))


def _safe_config(value: Any) -> Any:
    """Defense-in-depth redaction for reportable config snapshots."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in {"authorization", "api_key", "token", "secret"} or lowered.endswith(
                ("_api_key", "_token", "_secret")
            ):
                result[key] = "[REDACTED]"
            else:
                result[key] = _safe_config(item)
        return result
    if isinstance(value, list):
        return [_safe_config(item) for item in value]
    return value


def _change_summary(before: str, after: str) -> dict[str, Any]:
    diff = list(difflib.ndiff(before.splitlines(), after.splitlines()))
    added = [line[2:] for line in diff if line.startswith("+ ")]
    removed = [line[2:] for line in diff if line.startswith("- ")]
    headings = sorted({
        line.strip() for line in added + removed
        if line.strip().startswith("#")
    })
    return {
        "added_lines": len(added),
        "removed_lines": len(removed),
        "changed_headings": headings,
    }


def _generator_metadata(round_dir: Path) -> dict[str, Any]:
    return _read_json(round_dir / "generator_metadata.json", {
        "model": "unavailable",
        "input_tokens": None,
        "output_tokens": None,
        "cost_usd": None,
        "cached": None,
    })


def _write_prompt_evolution(round_dir: Path) -> None:
    identity = _read_json(round_dir / "run_identity.json", {})
    prompts = _read_json(round_dir / "prompts.json", {}).get("prompts", [])
    comparison = _read_json(round_dir / "prompt_comparison.json", {}).get("ranking", [])
    decision = _read_json(round_dir / "round_decision.json", {})
    incoming = str(identity.get("initial_prompt", ""))
    selected = str(decision.get("selected_prompt", incoming))
    rank_by_index = {
        int(row["prompt_index"]): (rank, row)
        for rank, row in enumerate(comparison, start=1)
        if "prompt_index" in row
    }
    lines = [
        "# Prompt evolution",
        "",
        f"- Round: `{round_dir.name}`",
        f"- Accepted update: `{decision.get('accepted', 'unavailable')}`",
        f"- Ranked winner prompt index: `{decision.get('ranked_winner_prompt_index', 'unavailable')}`",
        f"- Generator evidence: `{_json_cell(_generator_metadata(round_dir))}`",
        "",
        "## Incoming champion",
        "",
        f"- SHA-256: `{_hash_text(incoming)}`",
        "",
        f"<details><summary>完整 Prompt</summary><pre>{html.escape(incoming)}</pre></details>",
        "",
        "## Candidates",
        "",
    ]
    for index, prompt in enumerate(prompts):
        rank, metrics = rank_by_index.get(index, ("unavailable", {}))
        change = _change_summary(incoming, str(prompt))
        headings = ", ".join(change["changed_headings"]) or "none"
        if _hash_text(str(prompt)) == _hash_text(selected):
            outcome = "selected as outgoing champion"
        elif rank == 1 and not decision.get("accepted"):
            outcome = "ranked winner not accepted because no material improvement was recorded"
        else:
            outcome = "rejected by lexicographic selection key"
        lines.extend([
            f"### Prompt {index}",
            "",
            f"- Rank: `{rank}`",
            f"- SHA-256: `{_hash_text(str(prompt))}`",
            f"- Characters: `{len(str(prompt))}`",
            f"- Task tokens: `{metrics.get('task_tokens', 'unavailable')}`",
            f"- Change summary: +{change['added_lines']} / -{change['removed_lines']} lines; headings: {headings}",
            f"- Selection key: `{_json_cell(metrics.get('selection_key', []))}`",
            f"- Outcome: {outcome}",
            "",
            f"<details><summary>完整 Prompt</summary><pre>{html.escape(str(prompt))}</pre></details>",
            "",
        ])
    cumulative = _change_summary(incoming, selected)
    lines.extend([
        "## Outgoing champion",
        "",
        f"- SHA-256: `{_hash_text(selected)}`",
        f"- Decision: `accepted={decision.get('accepted', 'unavailable')}`",
        f"- Change summary: +{cumulative['added_lines']} / -{cumulative['removed_lines']} lines",
        "",
        f"<details><summary>完整 Prompt</summary><pre>{html.escape(selected)}</pre></details>",
        "",
    ])
    _atomic_text(round_dir / "prompt_evolution.md", "\n".join(lines))


_RISK_COLUMNS = [
    "round", "sample_id", "sample_kind", "query", "boundary_note", "selection_sources",
    "risk_score", "risk_signals", "prompt_index", "prompt_hash", "task_model",
    "L2_id", "L2_name", "L3_id", "L3_name", "confidence", "reason", "raw_output",
    "hard_failure", "violations", "judge_selected", "judge_set", "judge_skipped_reason",
    "judge_model", "judge_score", "judge_confidence", "judge_signal_scores",
    "judge_hard_failures", "judge_rationale", "judge_tokens", "judge_cost_usd",
    "judge_cached", "stability_label", "stable",
]


def _write_risk_details(round_dir: Path, samples: dict[str, Sample]) -> None:
    identity = _read_json(round_dir / "run_identity.json", {})
    prompts = _read_json(round_dir / "prompts.json", {}).get("prompts", [])
    queue = _read_json(round_dir / "judge_queue.json", {})
    risks = {str(row["sample_id"]): row for row in _read_jsonl(round_dir / "risk_scores.jsonl")}
    tasks = {
        (int(row["prompt_index"]), str(row["sample_id"])): row
        for row in _read_jsonl(round_dir / "task_outputs.jsonl")
    }
    checks = {
        (int(row["prompt_index"]), str(row["sample_id"])): row
        for row in _read_jsonl(round_dir / "deterministic_checks.jsonl")
    }
    judgments = {
        (int(row["prompt_index"]), str(row["sample_id"])): row
        for row in _read_jsonl(round_dir / "judge_results.jsonl")
    }
    stability = {
        (int(row["prompt_index"]), str(row["sample_id"])): row
        for row in _read_jsonl(round_dir / "stability_outputs.jsonl")
    }
    fixed = set(map(str, queue.get("fixed_case_ids", [])))
    dynamic = set(map(str, queue.get("dynamic_case_ids", [])))
    memory = (
        set(map(str, queue.get("memory_case_ids", [])))
        | set(map(str, identity.get("risk_memory_case_ids", [])))
    )
    queued = set(map(str, queue.get("case_ids", [])))
    risk_ids = {
        sample_id for sample_id, row in risks.items()
        if float(row.get("risk_score", 0.0)) > 0
    } | queued
    rows: list[dict[str, Any]] = []
    for sample_id in sorted(risk_ids):
        sample = samples.get(sample_id)
        sources = []
        if sample_id in fixed:
            sources.append("fixed_boundary")
        if sample_id in dynamic:
            sources.append("dynamic_top_k")
        if sample_id in memory:
            sources.append("risk_memory")
        risk = risks.get(sample_id, {})
        for prompt_index, prompt in enumerate(prompts):
            task = tasks.get((prompt_index, sample_id), {})
            check = checks.get((prompt_index, sample_id), {})
            parsed = check.get("parsed_output")
            judgment_row = judgments.get((prompt_index, sample_id))
            judgment = judgment_row.get("judgment", {}) if judgment_row else {}
            raw_judge = judgment.get("raw", {}) if isinstance(judgment, dict) else {}
            repeat = stability.get((prompt_index, sample_id), {})
            repeat_parsed = None
            try:
                repeat_parsed = json.loads(str(repeat.get("output", "")))
            except json.JSONDecodeError:
                pass
            hard_failure = bool(check.get("hard_failure", False))
            skipped = ""
            if not judgment_row:
                skipped = "deterministic_hard_failure" if hard_failure else "not_selected_for_judge"
            rows.append({
                "round": round_dir.name,
                "sample_id": sample_id,
                "sample_kind": sample.metadata.get("sample_kind", "") if sample else risk.get("sample_kind", ""),
                "query": _json_cell(sample.inputs if sample else {}),
                "boundary_note": sample.metadata.get("boundary_note", "") if sample else "",
                "selection_sources": "|".join(sources),
                "risk_score": risk.get("risk_score", 0.0),
                "risk_signals": _json_cell(risk.get("risk_signals", [])),
                "prompt_index": prompt_index,
                "prompt_hash": _hash_text(str(prompt)),
                "task_model": task.get("model", "unavailable"),
                "L2_id": parsed.get("L2_id", "") if isinstance(parsed, dict) else "",
                "L2_name": parsed.get("L2_name", "") if isinstance(parsed, dict) else "",
                "L3_id": parsed.get("L3_id", "") if isinstance(parsed, dict) else "",
                "L3_name": parsed.get("L3_name", "") if isinstance(parsed, dict) else "",
                "confidence": parsed.get("confidence", "") if isinstance(parsed, dict) else "",
                "reason": parsed.get("reason", "") if isinstance(parsed, dict) else "",
                "raw_output": task.get("output", ""),
                "hard_failure": hard_failure,
                "violations": _json_cell(check.get("violations", [])),
                "judge_selected": bool(judgment_row),
                "judge_set": judgment_row.get("set", "") if judgment_row else "",
                "judge_skipped_reason": skipped,
                "judge_model": raw_judge.get("_promptos_judge_model", "unavailable") if judgment_row else "",
                "judge_score": judgment.get("score", "") if judgment_row else "",
                "judge_confidence": judgment.get("confidence", "") if judgment_row else "",
                "judge_signal_scores": _json_cell(judgment.get("signal_scores", {})) if judgment_row else "",
                "judge_hard_failures": _json_cell(judgment.get("hard_failures", [])) if judgment_row else "",
                "judge_rationale": judgment.get("rationale", "") if judgment_row else "",
                "judge_tokens": raw_judge.get("_promptos_judge_tokens", "") if judgment_row else "",
                "judge_cost_usd": raw_judge.get("_promptos_judge_cost_usd", "") if judgment_row else "",
                "judge_cached": raw_judge.get("_promptos_judge_cached", "") if judgment_row else "",
                "stability_label": "|".join(_label(repeat_parsed)),
                "stable": repeat.get("stable", ""),
            })
    temporary = round_dir / ".risk_case_details.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=_RISK_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(round_dir / "risk_case_details.csv")


def _round_summary(round_dir: Path) -> dict[str, Any]:
    identity = _read_json(round_dir / "run_identity.json", {})
    queue = _read_json(round_dir / "judge_queue.json", {})
    comparison = _read_json(round_dir / "prompt_comparison.json", {}).get("ranking", [])
    decision = _read_json(round_dir / "round_decision.json", {})
    decision_summary = {
        key: value for key, value in decision.items() if key != "selected_prompt"
    }
    if "selected_prompt" in decision:
        selected_prompt = str(decision["selected_prompt"])
        decision_summary["selected_prompt_sha256"] = _hash_text(selected_prompt)
        decision_summary["selected_prompt_chars"] = len(selected_prompt)
    risks = _read_jsonl(round_dir / "risk_scores.jsonl")
    checks = _read_jsonl(round_dir / "deterministic_checks.jsonl")
    judges = _read_jsonl(round_dir / "judge_results.jsonl")
    task_rows = _read_jsonl(round_dir / "task_outputs.jsonl")
    stability = _read_jsonl(round_dir / "stability_outputs.jsonl")
    risk_counts = Counter(
        signal["name"] for row in risks for signal in row.get("risk_signals", [])
    )
    violation_counts = Counter(
        violation["code"] for row in checks for violation in row.get("violations", [])
    )
    observed_task = Counter(str(row.get("model", "unavailable")) for row in task_rows)
    observed_judge = Counter(
        str(row.get("judgment", {}).get("raw", {}).get("_promptos_judge_model", "unavailable"))
        for row in judges
    )
    labels: dict[int, Counter[str]] = defaultdict(Counter)
    unknown_by_prompt: Counter[int] = Counter()
    low_confidence_by_prompt: Counter[int] = Counter()
    labels_by_sample: dict[str, set[str]] = defaultdict(set)
    for row in checks:
        prompt_index = int(row["prompt_index"])
        label = "|".join(_label(row.get("parsed_output")))
        labels[prompt_index][label] += 1
        labels_by_sample[str(row["sample_id"])].add(label)
        parsed = row.get("parsed_output")
        if isinstance(parsed, dict) and parsed.get("L3_id") == "Unknown":
            unknown_by_prompt[prompt_index] += 1
        confidence = row.get("confidence")
        if confidence is not None and float(confidence) < 0.65:
            low_confidence_by_prompt[prompt_index] += 1
    judge_confidence = Counter()
    signal_totals: Counter[str] = Counter()
    signal_counts: Counter[str] = Counter()
    for row in judges:
        value = row.get("judgment", {}).get("confidence")
        if value is None:
            judge_confidence["unavailable"] += 1
        elif float(value) < 0.6:
            judge_confidence["low_<0.6"] += 1
        elif float(value) < 0.8:
            judge_confidence["medium_0.6-0.8"] += 1
        else:
            judge_confidence["high_>=0.8"] += 1
        for name, score in row.get("judgment", {}).get("signal_scores", {}).items():
            signal_totals[str(name)] += float(score)
            signal_counts[str(name)] += 1
    signal_means = {
        name: signal_totals[name] / signal_counts[name] for name in sorted(signal_totals)
    }
    low_judgments = sorted(
        judges,
        key=lambda row: (
            float(row.get("judgment", {}).get("score", 0.0)),
            str(row.get("sample_id", "")),
            int(row.get("prompt_index", 0)),
        ),
    )[:5]
    task_and_stability = task_rows + stability
    task_tokens = sum(
        int(row.get("input_tokens", 0)) + int(row.get("output_tokens", 0))
        for row in task_and_stability
    )
    unique_judge_rows = []
    seen_shared: set[str] = set()
    for row in judges:
        shared_id = str(row.get("shared_call_id", ""))
        if shared_id and shared_id in seen_shared:
            continue
        if shared_id:
            seen_shared.add(shared_id)
        unique_judge_rows.append(row)
    judge_tokens = sum(
        int(row.get("judgment", {}).get("raw", {}).get("_promptos_judge_tokens", 0))
        for row in unique_judge_rows
    )
    cached_task = sum(bool(row.get("cached")) for row in task_and_stability)
    cached_judge = sum(
        bool(row.get("judgment", {}).get("raw", {}).get("_promptos_judge_cached"))
        for row in unique_judge_rows
    )
    return {
        "round": round_dir.name,
        "queue": {
            "fixed": len(queue.get("fixed_case_ids", [])),
            "dynamic": len(queue.get("dynamic_case_ids", [])),
            "memory": len(set(queue.get("memory_case_ids", [])) | set(identity.get("risk_memory_case_ids", []))),
            "total": len(queue.get("case_ids", [])),
        },
        "comparison": comparison,
        "decision": decision_summary,
        "risk_signal_counts": dict(risk_counts),
        "violation_counts": dict(violation_counts),
        "observed_task_models": dict(observed_task),
        "observed_judge_models": dict(observed_judge),
        "generator": _generator_metadata(round_dir),
        "judge_count": len(unique_judge_rows),
        "candidate_judgment_count": len(judges),
        "judge_mean": (
            sum(float(row["judgment"]["score"]) for row in judges) / len(judges)
            if judges else None
        ),
        "judge_confidence_distribution": dict(judge_confidence),
        "judge_signal_means": signal_means,
        "lowest_judgments": [
            {
                "sample_id": row.get("sample_id"),
                "prompt_index": row.get("prompt_index"),
                "score": row.get("judgment", {}).get("score"),
                "confidence": row.get("judgment", {}).get("confidence"),
                "rationale": row.get("judgment", {}).get("rationale"),
            }
            for row in low_judgments
        ],
        "stable_count": sum(bool(row.get("stable")) for row in stability),
        "stability_count": len(stability),
        "label_distributions": {str(key): dict(value) for key, value in labels.items()},
        "unknown_by_prompt": dict(unknown_by_prompt),
        "low_confidence_by_prompt": dict(low_confidence_by_prompt),
        "label_flip_count": sum(len(values) > 1 for values in labels_by_sample.values()),
        "usage": {
            "task_and_stability_tokens": task_tokens,
            "judge_tokens": judge_tokens,
            "task_and_stability_cost_usd": sum(float(row.get("cost_usd", 0.0)) for row in task_and_stability),
            "judge_cost_usd": sum(
                float(row.get("judgment", {}).get("raw", {}).get("_promptos_judge_cost_usd", 0.0))
                for row in unique_judge_rows
            ),
            "cached_task_and_stability": cached_task,
            "cached_judge": cached_judge,
        },
    }


def render_auto_report(
    run_dir: Path,
    samples: Iterable[Sample],
    *,
    completed: bool,
    stop_reason: str | None = None,
) -> None:
    """Render audit artifacts without model calls; safe to repeat during resume."""
    sample_list = list(samples)
    sample_map = {sample.id: sample for sample in sample_list}
    identity = _safe_config(_read_json(run_dir / "run_identity.json", {}))
    state = _read_json(run_dir / "auto_state.json", {})
    final = _safe_config(_read_json(run_dir / "run.json", {}))
    warnings: list[str] = []
    round_dirs = sorted(
        path for path in run_dir.glob("round_*")
        if path.is_dir() and (path / "round_decision.json").exists()
    )
    summaries = []
    source_hashes: dict[str, str] = {}
    for round_dir in round_dirs:
        _write_risk_details(round_dir, sample_map)
        _write_prompt_evolution(round_dir)
        summaries.append(_round_summary(round_dir))
        for name in (
            "run_identity.json", "prompts.json", "task_outputs.jsonl",
            "deterministic_checks.jsonl", "risk_scores.jsonl", "judge_queue.json",
            "judge_results.jsonl", "stability_outputs.jsonl",
            "prompt_comparison.json", "round_decision.json",
        ):
            path = round_dir / name
            if path.exists():
                source_hashes[str(path.relative_to(run_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
            else:
                warnings.append(f"{round_dir.name}/{name} unavailable")
    configured_models = identity.get("task_config", {}).get("models", {})
    if not configured_models:
        configured_models = final.get("task_config", {}).get("models", {})
    sample_kinds = Counter(
        str(sample.metadata.get("sample_kind", "unspecified")) for sample in sample_list
    )
    status = final.get(
        "champion_status",
        "provisional_silver_or_unlabeled"
        if not any(sample.label_source == "gold_human" for sample in sample_list)
        else "gold_validated",
    )
    lines = [
        "# PromptOS automated experiment report",
        "",
        "## Run identity",
        "",
        f"- Run ID: `{final.get('run_id', run_dir.name)}`",
        "- Mode: `layered_auto`",
        f"- Report status: `{'complete' if completed else 'incomplete'}`",
        f"- Champion status: `{status}`",
        f"- Rounds completed: `{state.get('rounds_completed', len(summaries))}`",
        f"- Stop reason: `{stop_reason or final.get('stop_reason', 'unavailable')}`",
        f"- Started at: `{final.get('started_at', state.get('started_at', 'unavailable'))}`",
        f"- Completed at: `{final.get('completed_at', 'unavailable')}`",
        f"- PromptOS version: `{identity.get('software_version', 'unavailable')}`",
        f"- Task config version: `{final.get('task_config', {}).get('version', 'unavailable')}`",
        f"- Sample count: `{len(sample_list)}`; composition: `{_json_cell(dict(sample_kinds))}`",
        f"- Frozen sample hash: `{identity.get('sample_hash', 'unavailable')}`",
        f"- SignalSpec: `{_json_cell(identity.get('signal_spec', {}))}`",
        "",
        "## Models and policy",
        "",
        f"- Configured models: `{_json_cell(configured_models or {'status': 'unavailable'})}`",
        f"- Layered policy: `{_json_cell(identity.get('layered_policy', {}))}`",
        f"- Auto policy: `{_json_cell(identity.get('auto_policy', {}))}`",
        "",
        "## Round overview",
        "",
        "| Round | Updated | Hard failures | Boundary Judge | Dynamic Judge | Stability | Fixed/Dynamic/Memory | Judge model |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for summary in summaries:
        comparison = summary["comparison"]
        winner = comparison[0] if comparison else {}
        dynamic_count = summary["queue"]["dynamic"]
        dynamic_display = (
            f"{float(winner.get('dynamic_judge_score', 0.0)):.4f}"
            if dynamic_count else "N/A (n=0)"
        )
        judge_models = ", ".join(summary["observed_judge_models"]) or "unavailable"
        lines.append(
            f"| {summary['round']} | {summary['decision'].get('accepted', 'unavailable')} "
            f"| {winner.get('hard_failure_count', 'unavailable')} "
            f"| {float(winner.get('boundary_judge_score', 0.0)):.4f} "
            f"| {dynamic_display} | {float(winner.get('stability_rate', 0.0)):.4f} "
            f"| {summary['queue']['fixed']}/{summary['queue']['dynamic']}/{summary['queue']['memory']} "
            f"| {judge_models} |"
        )
    for summary in summaries:
        lines.extend([
            "",
            f"## {summary['round']}",
            "",
            f"- Decision: `{_json_cell(summary['decision'])}`",
            f"- Queue: `{_json_cell(summary['queue'])}`",
            f"- Risk signals: `{_json_cell(summary['risk_signal_counts'])}`",
            f"- Hard violations: `{_json_cell(summary['violation_counts'])}`",
            f"- Observed Task models: `{_json_cell(summary['observed_task_models'])}`",
            f"- Observed Judge models: `{_json_cell(summary['observed_judge_models'])}`",
            f"- Generator evidence: `{_json_cell(summary['generator'])}`",
            f"- Usage and cache: `{_json_cell(summary['usage'])}`",
            f"- Judge results: `{summary['judge_count']}`; mean score: "
            f"`{summary['judge_mean'] if summary['judge_mean'] is not None else 'N/A (n=0)'}`",
            f"- Judge confidence distribution: `{_json_cell(summary['judge_confidence_distribution'])}`",
            f"- Judge signal means: `{_json_cell(summary['judge_signal_means'])}`",
            f"- Lowest Judge cases: `{_json_cell(summary['lowest_judgments'])}`",
            f"- Stability: `{summary['stable_count']}/{summary['stability_count']}`",
            f"- Label distributions: `{_json_cell(summary['label_distributions'])}`",
            f"- Unknown by Prompt: `{_json_cell(summary['unknown_by_prompt'])}`",
            f"- Low confidence by Prompt: `{_json_cell(summary['low_confidence_by_prompt'])}`",
            f"- Prompt label-flip samples: `{summary['label_flip_count']}`",
            f"- Full risk evidence: [{summary['round']}/risk_case_details.csv]"
            f"({summary['round']}/risk_case_details.csv)",
            f"- Full prompts and changes: [{summary['round']}/prompt_evolution.md]"
            f"({summary['round']}/prompt_evolution.md)",
            "",
            "### Candidate ranking",
            "",
            "| Rank | Prompt | Hard failures | Boundary | Dynamic | Stability | Chars | Tokens | Selection key |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ])
        for rank, row in enumerate(summary["comparison"], start=1):
            dynamic = (
                f"{float(row.get('dynamic_judge_score', 0.0)):.4f}"
                if summary["queue"]["dynamic"] else "N/A (n=0)"
            )
            lines.append(
                f"| {rank} | {row.get('prompt_index')} | {row.get('hard_failure_count')} "
                f"| {float(row.get('boundary_judge_score', 0.0)):.4f} | {dynamic} "
                f"| {float(row.get('stability_rate', 0.0)):.4f} "
                f"| {row.get('prompt_chars')} | {row.get('task_tokens')} "
                f"| `{_json_cell(row.get('selection_key', []))}` |"
            )
    initial_prompt = str(identity.get("initial_prompt", ""))
    champion = str(final.get("champion_prompt", state.get("current_prompt", initial_prompt)))
    cumulative = _change_summary(initial_prompt, champion)
    review_path = run_dir / "human_review_top20.csv"
    review_rows: list[dict[str, str]] = []
    if review_path.exists():
        with review_path.open(encoding="utf-8", newline="") as file:
            review_rows = list(csv.DictReader(file))
    review_count = len(review_rows)
    priority_counts = Counter(row.get("priority_band", "unavailable") for row in review_rows)
    review_top = [
        {
            "sample_id": row.get("sample_id"),
            "priority_band": row.get("priority_band"),
            "risk_score": row.get("risk_score"),
            "risk_reasons": row.get("risk_reasons"),
            "hard_failure": row.get("hard_failure"),
        }
        for row in review_rows
    ]
    lines.extend([
        "",
        "## Final decision and review",
        "",
        f"- Initial-to-champion change: +{cumulative['added_lines']} / -{cumulative['removed_lines']} lines.",
        f"- Human review queue: `{review_count if review_path.exists() else 'unavailable'}` cases.",
        f"- Human review priority distribution: `{_json_cell(dict(priority_counts))}`",
        f"- Human review cases: `{_json_cell(review_top)}`",
        f"- Budget: `{_json_cell(final.get('budget', {}))}`",
        "",
        "## Limitations",
        "",
        "- This report does not call any model and adds no inference cost.",
        f"- Result status is `{status}`.",
        "- Without a frozen gold-human test set, this report does not calculate or imply Accuracy, Precision, Recall, or F1.",
        "- Raw queries are retained locally for audit and may contain sensitive experiment data.",
        "",
    ])
    _atomic_text(run_dir / "experiment_report.md", "\n".join(lines))
    manifest = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "complete" if completed else "incomplete",
        "completed_rounds": [path.name for path in round_dirs],
        "source_hashes": dict(sorted(source_hashes.items())),
        "warnings": sorted(set(warnings)),
        "contains_raw_queries": True,
        "model_calls_added": 0,
    }
    _atomic_json(run_dir / "report_manifest.json", manifest)
