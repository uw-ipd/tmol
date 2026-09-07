"""Summarize paired per-term rotamer-scoring profiles."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def close_enough(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-5, abs_tol=1e-3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = [
        json.loads(path.read_text()) for path in sorted(args.input.glob("*.json"))
    ]
    groups: dict[tuple, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    metadata: dict[tuple, dict] = {}
    for record in records:
        common = (
            record["dataset_id"],
            record["modality"],
            record["device"],
            record["batch_size"],
        )
        metadata[common] = record
        for term in record["terms"]:
            groups[(*common, term["term"])][record["label"]].append(term)

    rows = []
    for key, labels in sorted(groups.items()):
        baseline = labels.get("baseline", [])
        candidate = labels.get("candidate", [])
        baseline_seconds = [term["median_seconds"] for term in baseline]
        candidate_seconds = [term["median_seconds"] for term in candidate]
        baseline_time = (
            statistics.median(baseline_seconds) if baseline_seconds else None
        )
        candidate_time = (
            statistics.median(candidate_seconds) if candidate_seconds else None
        )
        checksums = [term["checksum"] for term in [*baseline, *candidate]]
        checksum_consistent = bool(checksums) and all(
            close_enough(checksums[0], value) for value in checksums[1:]
        )
        common = key[:4]
        record = metadata[common]
        rows.append(
            {
                "dataset_id": key[0],
                "modality": key[1],
                "polymer_residues": record["polymer_residues"],
                "device": key[2],
                "batch_size": key[3],
                "n_rotamers": record["n_rotamers"],
                "term": key[4],
                "baseline_replicates": len(baseline_seconds),
                "candidate_replicates": len(candidate_seconds),
                "baseline_seconds": baseline_time,
                "candidate_seconds": candidate_time,
                "candidate_speedup": (
                    baseline_time / candidate_time
                    if baseline_time is not None and candidate_time is not None
                    else None
                ),
                "checksum_consistent": checksum_consistent,
            }
        )

    if not rows:
        parser.error(f"no JSON rotamer-profile records found under {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "rotamer_term_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    report = [
        "# Rotamer-score term A/B",
        "",
        "Speedup is baseline time divided by candidate time; values above 1 are faster.",
        "Each timing is the median across independent A–B–B–A process records.",
        "",
        "| Dataset | Residues | Rotamers | Term | Baseline (s) | Candidate (s) | Speedup | Replicates A/B | Checksum |",
        "|---|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        baseline = row["baseline_seconds"]
        candidate = row["candidate_seconds"]
        speedup = row["candidate_speedup"]
        report.append(
            "| {dataset_id} | {polymer_residues} | {n_rotamers} | {term} | {baseline} | {candidate} | {speedup} | {baseline_replicates}/{candidate_replicates} | {checksum} |".format(
                **row,
                baseline=f"{baseline:.6g}" if baseline is not None else "pending",
                candidate=f"{candidate:.6g}" if candidate is not None else "pending",
                speedup=f"{speedup:.3f}x" if speedup is not None else "pending",
                checksum="match" if row["checksum_consistent"] else "MISMATCH",
            )
        )
    (args.output / "rotamer_term_report.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
