"""Finance intent-classification plugin migrated from the legacy DSPy project.

The plugin owns taxonomy-specific concepts; PromptOS core remains task-neutral.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from promptos.core import HardConstraint, SignalSpec, SoftSignal, TaskSpec


@dataclass(frozen=True)
class FinanceTaxonomy:
    version: str
    l2_by_l3: dict[str, str]
    names: dict[str, str]
    context: str

    @classmethod
    def load(cls, path: Path) -> "FinanceTaxonomy":
        raw = json.loads(path.read_text(encoding="utf-8"))
        l2_by_l3, names, lines = {}, {}, []
        for category in raw["L2_list"]:
            l2_id, l2_name = category["id"], category["name"]
            names[l2_id] = l2_name
            lines.append(f"{l2_id} {l2_name}: {category.get('definition', '').strip()}")
            for leaf in category.get("L3_list", category.get("l3_list", [])):
                l3_id, l3_name = leaf["id"], leaf["name"]
                l2_by_l3[l3_id] = l2_id
                names[l3_id] = l3_name
                lines.append(f"- {l3_id} {l3_name}: {leaf.get('definition', '').strip()}")
        return cls(str(raw.get("version", "unknown")), l2_by_l3, names, "\n".join(lines))

    def valid(self, l2_id: str, l3_id: str) -> bool:
        return l3_id == "Unknown" or self.l2_by_l3.get(l3_id) == l2_id


def task_spec() -> TaskSpec:
    return TaskSpec(
        name="finance_classification",
        input_fields=["query"],
        output_description="Classify a financial query into a taxonomy L2/L3 intent and return strict JSON.",
        output_schema={
            "type": "object",
            "required": ["L2_id", "L3_id", "confidence", "reason"],
            "properties": {
                "L2_id": {"type": "string"}, "L3_id": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string"},
            },
        },
    )


def default_prompt(taxonomy: FinanceTaxonomy) -> str:
    return f"""You are a financial intent classifier. Classify the user's query using taxonomy version {taxonomy.version}.
Return JSON only with L2_id, L3_id, confidence (0 to 1), and a concise reason.
If the query is non-financial or lacks enough information, return L3_id as Unknown.

Taxonomy:
{taxonomy.context}"""


def signal_spec(taxonomy: FinanceTaxonomy, version: int = 1) -> SignalSpec:
    return SignalSpec(
        id="finance-classification-quality", version=version,
        acceptance_criteria="Correctly identify the user's final financial intent while returning a valid taxonomy label.",
        hard_constraints=[
            HardConstraint("valid_json", "Output must be parseable JSON with required fields."),
            HardConstraint("valid_taxonomy", "L3 must belong to the selected L2, unless L3 is Unknown."),
        ],
        soft_signals=[
            SoftSignal("intent_alignment", "Does the classification match the user's final requested action?", 0.6),
            SoftSignal("boundary_handling", "Are ambiguous, non-financial, and insufficient-context queries handled conservatively?", 0.25),
            SoftSignal("reason_quality", "Is the reason concise and tied to the query?", 0.15),
        ],
        status="draft", created_by=f"finance_classification:{taxonomy.version}",
    )


def validate_output(value: dict[str, Any], taxonomy: FinanceTaxonomy) -> list[str]:
    """Return violations suitable for a task-specific hard-constraint evaluator."""
    required = {"L2_id", "L3_id", "confidence", "reason"}
    if not required.issubset(value):
        return ["valid_json"]
    try:
        confidence = float(value["confidence"])
    except (TypeError, ValueError):
        return ["valid_json"]
    if not 0 <= confidence <= 1:
        return ["valid_json"]
    return [] if taxonomy.valid(str(value["L2_id"]), str(value["L3_id"])) else ["valid_taxonomy"]
