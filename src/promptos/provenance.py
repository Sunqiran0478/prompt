"""Review import and immutable label provenance helpers."""
from __future__ import annotations

import json
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .core import Sample


@dataclass(frozen=True)
class Annotation:
    sample_id: str
    value: Any
    source: str  # silver_auto | gold_human
    reviewer: str | None = None
    judge_run_id: str | None = None
    rationale: str = ""
    judge_model: str | None = None

    def validate(self) -> None:
        if self.source not in {"silver_auto", "gold_human"}:
            raise ValueError("Annotation source must be silver_auto or gold_human.")
        if self.source == "gold_human" and not self.reviewer:
            raise ValueError("gold_human annotations require a reviewer identifier.")
        if self.source == "silver_auto" and not self.judge_run_id:
            raise ValueError("silver_auto annotations require a judge_run_id.")


def apply_annotations(samples: list[Sample], annotations: list[Annotation]) -> list[Sample]:
    by_id = {item.sample_id: item for item in annotations}
    unknown = set(by_id) - {sample.id for sample in samples}
    if unknown:
        raise ValueError(f"Annotations reference unknown sample IDs: {sorted(unknown)}")
    output = []
    for sample in samples:
        annotation = by_id.get(sample.id)
        if annotation is None:
            output.append(sample)
            continue
        annotation.validate()
        metadata = {**sample.metadata, "annotation": asdict(annotation)}
        output.append(Sample(sample.id, sample.inputs, annotation.value, annotation.source, metadata))
    return output


def load_annotations(path: Path) -> list[Annotation]:
    return [Annotation(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_review_cases(samples: list[Sample], limit: int) -> list[dict[str, Any]]:
    """Rank unlabelled cases from model-produced diagnostic metadata.

    Callers may supply uncertainty, disagreement, business_risk and
    format_failure in sample.metadata. The formula is deliberately transparent
    so a product team can audit or replace it.
    """
    ranked = []
    for sample in samples:
        if sample.expected is not None:
            continue
        meta = sample.metadata
        parts = {
            "uncertainty": float(meta.get("uncertainty", 0.0)),
            "disagreement": float(meta.get("disagreement", 0.0)),
            "business_risk": float(meta.get("business_risk", 0.0)),
            "format_failure": float(bool(meta.get("format_failure", False))),
        }
        importance = 0.35 * parts["uncertainty"] + 0.30 * parts["disagreement"] + 0.25 * parts["business_risk"] + 0.10 * parts["format_failure"]
        reason = max(parts, key=parts.get) if any(parts.values()) else "unlabeled_requires_independent_review"
        ranked.append({"sample_id": sample.id, "inputs": sample.inputs, "importance": round(importance, 4),
                       "reason": reason, "label_source": "unlabeled"})
    return sorted(ranked, key=lambda item: (-item["importance"], item["sample_id"]))[:limit]


REVIEW_COLUMNS = ["sample_id", "inputs_json", "proposed_value", "proposed_source", "judge_run_id", "proposed_rationale",
                  "decision", "reviewed_value", "reviewer", "review_rationale"]


def export_review_csv(samples: list[Sample], queue: list[dict[str, Any]], annotations: list[Annotation], path: Path) -> int:
    """Export a human-editable review sheet. No automatic label becomes gold here."""
    by_id = {sample.id: sample for sample in samples}
    proposals = {item.sample_id: item for item in annotations}
    selected = [item["sample_id"] for item in queue]
    unknown = set(selected) - set(by_id)
    if unknown:
        raise ValueError(f"Review queue references unknown samples: {sorted(unknown)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for sample_id in selected:
            sample, proposal = by_id[sample_id], proposals.get(sample_id)
            writer.writerow({"sample_id": sample_id, "inputs_json": json.dumps(sample.inputs, ensure_ascii=False),
                             "proposed_value": json.dumps(proposal.value, ensure_ascii=False) if proposal else "",
                             "proposed_source": proposal.source if proposal else "unlabeled",
                             "judge_run_id": proposal.judge_run_id if proposal else "",
                             "proposed_rationale": proposal.rationale if proposal else "",
                             "decision": "", "reviewed_value": "", "reviewer": "", "review_rationale": ""})
    return len(selected)


def import_review_csv(path: Path) -> tuple[list[Annotation], list[dict[str, str]]]:
    """Convert human decisions to gold annotations; rejected rows never enter gold."""
    annotations, rejected = [], []
    with path.open(encoding="utf-8", newline="") as file:
        for line, row in enumerate(csv.DictReader(file), 2):
            decision = (row.get("decision") or "").strip().lower()
            if not decision:
                continue
            if decision == "reject":
                rejected.append({"sample_id": row.get("sample_id", ""), "reason": row.get("review_rationale", "rejected by reviewer")})
                continue
            if decision not in {"approve", "edit"}:
                raise ValueError(f"Line {line}: decision must be approve, edit, or reject.")
            reviewer = (row.get("reviewer") or "").strip()
            if not reviewer:
                raise ValueError(f"Line {line}: approved/edit decisions require reviewer.")
            raw_value = row.get("proposed_value") if decision == "approve" else row.get("reviewed_value")
            if not raw_value:
                raise ValueError(f"Line {line}: {decision} requires a value.")
            try:
                value = json.loads(raw_value)
            except json.JSONDecodeError:
                value = raw_value
            annotation = Annotation((row.get("sample_id") or "").strip(), value, "gold_human", reviewer=reviewer,
                                    rationale=(row.get("review_rationale") or "").strip())
            if not annotation.sample_id:
                raise ValueError(f"Line {line}: sample_id is required.")
            annotation.validate()
            annotations.append(annotation)
    seen = set()
    duplicates = [item.sample_id for item in annotations if item.sample_id in seen or seen.add(item.sample_id)]
    if duplicates:
        raise ValueError(f"A review sheet cannot contain multiple final decisions for one sample: {sorted(set(duplicates))}")
    return annotations, rejected


def write_annotations(path: Path, annotations: list[Annotation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(asdict(item), ensure_ascii=False) + "\n" for item in annotations), encoding="utf-8")


def write_samples(path: Path, samples: list[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(asdict(item), ensure_ascii=False) + "\n" for item in samples), encoding="utf-8")
