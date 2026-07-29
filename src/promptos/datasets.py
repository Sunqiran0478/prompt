"""Deterministic dataset splitting with a persisted anti-leakage manifest."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .core import RunStore, Sample


def _bucket(sample_id: str, seed: str) -> float:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()
    return int(digest[:12], 16) / float(16 ** 12)


def split_gold_samples(samples: list[Sample], seed: str, optimize_ratio: float = 0.6, validation_ratio: float = 0.2) -> dict[str, list[Sample]]:
    if not 0 < optimize_ratio < 1 or not 0 < validation_ratio < 1 or optimize_ratio + validation_ratio >= 1:
        raise ValueError("Split ratios must be positive and leave a final_test remainder.")
    if any(sample.label_source != "gold_human" or sample.expected is None for sample in samples):
        raise ValueError("Only gold_human samples with expected outputs may enter optimize/validation/final_test splits.")
    if len(samples) < 3:
        raise ValueError("At least three gold_human samples are needed for three isolated partitions.")
    # Sorting by a seeded hash avoids data-order dependence; explicit quotas avoid
    # accidental empty validation/final partitions on small, valuable gold sets.
    ordered = sorted(samples, key=lambda sample: (_bucket(sample.id, seed), sample.id))
    optimize_n = max(1, round(len(samples) * optimize_ratio))
    validation_n = max(1, round(len(samples) * validation_ratio))
    while optimize_n + validation_n >= len(samples):
        if optimize_n >= validation_n and optimize_n > 1:
            optimize_n -= 1
        elif validation_n > 1:
            validation_n -= 1
        else:
            raise ValueError("Unable to allocate a final_test partition.")
    output = {"optimize": ordered[:optimize_n], "validation": ordered[optimize_n:optimize_n + validation_n],
              "final_test": ordered[optimize_n + validation_n:]}
    return output


def write_split(root: Path, splits: dict[str, list[Sample]], seed: str) -> Path:
    root.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, samples in splits.items():
        path = root / f"{name}.jsonl"
        path.write_text("".join(json.dumps(asdict(sample), ensure_ascii=False) + "\n" for sample in samples), encoding="utf-8")
        hashes[name] = RunStore.dataset_hash(samples)
    manifest = {"seed": seed, "locked": True, "splits": {name: {"count": len(samples), "hash": hashes[name]} for name, samples in splits.items()},
                "rule": "final_test is never used for candidate selection; rerun splitting only with a new output directory."}
    (root / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return root
