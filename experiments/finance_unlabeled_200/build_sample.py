"""Build this experiment's privacy-minimized unlabeled finance sample.

The output contains 150 randomly sampled production-pool queries plus all 50
handwritten boundary probes. Held-out request IDs are excluded before sampling.
Boundary descriptions are retained only as metadata and are never promoted to
reference labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text(value: object) -> str:
    return " ".join(str(value).split()).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--held-out", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total", type=int, default=200)
    parser.add_argument("--boundary", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    if args.total <= 0 or args.boundary <= 0 or args.boundary >= args.total:
        raise ValueError("Require total > boundary > 0.")

    pool = pd.read_parquet(args.pool)
    probes = pd.read_parquet(args.probe)
    held_out = pd.read_parquet(args.held_out)
    required_pool = {"conversationRequestId", "cleaned_query", "userQuery"}
    required_probe = {"query", "expected_boundary"}
    if missing := required_pool - set(pool.columns):
        raise ValueError(f"Pool is missing columns: {sorted(missing)}")
    if missing := required_probe - set(probes.columns):
        raise ValueError(f"Probe set is missing columns: {sorted(missing)}")
    if "conversationRequestId" not in held_out:
        raise ValueError("Held-out set requires conversationRequestId.")
    if len(probes) != args.boundary:
        raise ValueError(f"Expected exactly {args.boundary} boundary probes, found {len(probes)}.")

    held_ids = set(held_out["conversationRequestId"].dropna().astype(str))
    candidates = pool[~pool["conversationRequestId"].astype(str).isin(held_ids)].copy()
    candidates["query"] = candidates["cleaned_query"].fillna(candidates["userQuery"]).map(normalized_text)
    candidates = candidates[candidates["query"] != ""].drop_duplicates("query")

    probe_queries = probes["query"].map(normalized_text)
    candidates = candidates[~candidates["query"].isin(set(probe_queries))]
    random_count = args.total - args.boundary
    if len(candidates) < random_count:
        raise ValueError(f"Only {len(candidates)} eligible unique pool rows; need {random_count}.")
    random_rows = candidates.sample(n=random_count, random_state=args.seed, replace=False).reset_index(drop=True)

    records: list[dict[str, object]] = []
    for index, row in random_rows.iterrows():
        records.append({
            "id": f"finance_pool_{index:03d}",
            "query": row["query"],
            "label_source": "unlabeled",
            "metadata": {
                "sample_kind": "random_pool",
                "source": args.pool.name,
            },
        })
    for index, row in probes.reset_index(drop=True).iterrows():
        records.append({
            "id": f"finance_boundary_{index:03d}",
            "query": normalized_text(row["query"]),
            "label_source": "unlabeled",
            "metadata": {
                "sample_kind": "boundary_probe",
                "source": args.probe.name,
                "boundary_note": normalized_text(row["expected_boundary"]),
            },
        })

    output = pd.DataFrame(records)
    if len(output) != args.total or output["id"].duplicated().any() or output["query"].duplicated().any():
        raise RuntimeError("Output invariants failed: count, IDs, or query uniqueness.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"finance_optimize_unlabeled_{args.total}"
    parquet_path = args.output_dir / f"{stem}.parquet"
    jsonl_path = args.output_dir / f"{stem}.jsonl"
    manifest_path = args.output_dir / f"{stem}.manifest.json"
    output.to_parquet(parquet_path, index=False)
    with jsonl_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "dataset": stem,
        "label_source": "unlabeled",
        "seed": args.seed,
        "counts": {"total": len(output), "random_pool": random_count, "boundary_probe": args.boundary},
        "held_out_overlap_by_request_id": 0,
        "query_duplicates": 0,
        "privacy": {
            "included_fields": ["id", "query", "label_source", "metadata"],
            "excluded_source_fields": ["userId", "enterpriseId", "conversationRequestId"],
        },
        "sources": {
            "pool": {"name": args.pool.name, "sha256": file_sha256(args.pool)},
            "probe": {"name": args.probe.name, "sha256": file_sha256(args.probe)},
            "held_out_exclusion": {"name": args.held_out.name, "sha256": file_sha256(args.held_out)},
        },
        "outputs": {"parquet": parquet_path.name, "jsonl": jsonl_path.name},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), **manifest["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
