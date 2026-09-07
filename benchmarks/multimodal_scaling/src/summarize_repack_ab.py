"""Summarize fixed-input repacking A/B records."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    labels: dict[str, list[dict]] = {"baseline": [], "candidate": []}
    for path in sorted(args.input.glob("*.json")):
        label = path.name.split("-", 1)[0]
        if label in labels:
            labels[label].append(json.loads(path.read_text()))
    if not all(labels.values()):
        parser.error(f"incomplete baseline/candidate repack records under {args.input}")

    rows = []
    for label, records in labels.items():
        rows.append(
            {
                "label": label,
                "replicates": len(records),
                "median_seconds": statistics.median(
                    record["timing"]["median_seconds"] for record in records
                ),
                "median_first_call_seconds": statistics.median(
                    record["first_call_seconds"] for record in records
                ),
            }
        )
    baseline = rows[0]["median_seconds"]
    for row in rows:
        row["speedup_over_baseline"] = baseline / row["median_seconds"]

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "repack_ab_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    report = [
        "# Fixed-input repacking A/B",
        "",
        "Speedup is baseline time divided by variant time.",
        "",
        "| Variant | Replicates | Median (s) | First call (s) | Speedup |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            "| {label} | {replicates} | {median_seconds:.4f} | {median_first_call_seconds:.4f} | {speedup_over_baseline:.3f}x |".format(
                **row
            )
        )
    (args.output / "repack_ab_report.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
