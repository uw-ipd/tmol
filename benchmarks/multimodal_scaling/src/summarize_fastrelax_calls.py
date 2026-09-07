"""Summarize phase timings and call counts from matched FastRelax runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for path in sorted(args.input.glob("*.json")):
        record = json.loads(path.read_text())
        key = (
            record["dataset_id"],
            record["modality"],
            record["engine_version"],
        )
        groups[key].append(record)
    if not groups:
        parser.error(f"no JSON FastRelax call-count records found under {args.input}")

    rows = []
    for (dataset_id, modality, label), records in groups.items():
        counts = [record["whole_pose_call_counts"] for record in records]
        rows.append(
            {
                "dataset_id": dataset_id,
                "modality": modality,
                "polymer_residues": records[0]["polymer_residues"],
                "label": label,
                "replicates": len(records),
                "median_total_seconds": statistics.median(
                    record["seconds_per_structure_samples"][0]
                    for record in records
                ),
                "median_packing_seconds": statistics.median(
                    count["packing_seconds"] for count in counts
                ),
                "median_minimization_seconds": statistics.median(
                    count["minimization_seconds"] for count in counts
                ),
                "median_score_calls": statistics.median(
                    count["total"] for count in counts
                ),
                "median_gradient_calls": statistics.median(
                    count["coords_require_grad"] for count in counts
                ),
                "median_final_score": statistics.median(
                    record["validation_score_mean"] for record in records
                ),
            }
        )
    baseline_totals = {
        (row["dataset_id"], row["modality"]): row["median_total_seconds"]
        for row in rows
        if row["label"] == "baseline"
    }
    for row in rows:
        baseline_total = baseline_totals.get(
            (row["dataset_id"], row["modality"])
        )
        row["speedup_over_baseline"] = (
            baseline_total / row["median_total_seconds"]
            if baseline_total is not None
            else None
        )

    order = {label: index for index, label in enumerate(
        ("baseline", "optimizer-only", "shared-ordered", "candidate")
    )}
    rows.sort(
        key=lambda row: (
            row["modality"],
            row["polymer_residues"],
            row["dataset_id"],
            order.get(row["label"], len(order)),
            row["label"],
        )
    )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "fastrelax_call_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    report = [
        "# FastRelax phase and call-count A/B",
        "",
        "Speedup is baseline total time divided by variant total time.",
        "",
        "| Dataset | Modality | Residues | Variant | Replicates | Total (s) | Packing (s) | Minimization (s) | Score calls | Gradient calls | Final score | Speedup |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        speedup = row["speedup_over_baseline"]
        report.append(
            "| {dataset_id} | {modality} | {polymer_residues} | {label} | {replicates} | {median_total_seconds:.4f} | {median_packing_seconds:.4f} | {median_minimization_seconds:.4f} | {median_score_calls:g} | {median_gradient_calls:g} | {median_final_score:.6f} | {speedup} |".format(
                **row,
                speedup=f"{speedup:.3f}x" if speedup is not None else "pending",
            )
        )
    (args.output / "fastrelax_call_report.md").write_text(
        "\n".join(report) + "\n"
    )


if __name__ == "__main__":
    main()
