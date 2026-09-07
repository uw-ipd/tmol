"""Summarize paired baseline/candidate multimodal benchmark records."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def median_seconds(record: dict) -> float:
    return statistics.median(record["seconds_per_structure_samples"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = [
        json.loads(path.read_text()) for path in sorted(args.input.glob("*.json"))
    ]
    groups: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        key = (
            record["protocol"],
            record["modality"],
            record["dataset_id"],
            record["device"],
            record["batch_size"],
        )
        groups[key][record["engine_version"]].append(record)

    rows = []
    for key, versions in sorted(groups.items()):
        all_baseline = versions.get("baseline", [])
        all_candidate = versions.get("candidate", [])
        baseline = [r for r in all_baseline if r["status"] == "ok"]
        candidate = [r for r in all_candidate if r["status"] == "ok"]
        base_seconds = [median_seconds(record) for record in baseline]
        candidate_seconds = [median_seconds(record) for record in candidate]
        base_memory = [
            record.get("peak_memory_bytes") or record.get("process_peak_rss_bytes")
            for record in baseline
        ]
        candidate_memory = [
            record.get("peak_memory_bytes") or record.get("process_peak_rss_bytes")
            for record in candidate
        ]
        complete = bool(base_seconds and candidate_seconds)
        base_median = statistics.median(base_seconds) if base_seconds else None
        candidate_median = (
            statistics.median(candidate_seconds) if candidate_seconds else None
        )
        rows.append(
            {
                "protocol": key[0],
                "modality": key[1],
                "dataset_id": key[2],
                "device": key[3],
                "batch_size": key[4],
                "polymer_residues": next(
                    (record["polymer_residues"] for record in [*baseline, *candidate]),
                    None,
                ),
                "baseline_replicates": len(base_seconds),
                "candidate_replicates": len(candidate_seconds),
                "baseline_failures": len(all_baseline) - len(baseline),
                "candidate_failures": len(all_candidate) - len(candidate),
                "baseline_seconds_per_structure": base_median,
                "candidate_seconds_per_structure": candidate_median,
                "candidate_speedup": (
                    base_median / candidate_median if complete else None
                ),
                "baseline_peak_memory_bytes": (
                    statistics.median(base_memory) if base_memory else None
                ),
                "candidate_peak_memory_bytes": (
                    statistics.median(candidate_memory) if candidate_memory else None
                ),
                "candidate_memory_ratio": (
                    statistics.median(candidate_memory) / statistics.median(base_memory)
                    if base_memory and candidate_memory
                    else None
                ),
            }
        )

    if not rows:
        parser.error(f"no JSON benchmark records found under {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "candidate_ab_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    report = [
        "# Combined performance candidate A/B",
        "",
        "Speedup is baseline 0.1.55 time divided by candidate time; values above 1 are faster.",
        "Peak memory ratio is candidate divided by baseline; values below 1 use less memory.",
        "",
        "| Protocol | Modality | Residues | Device | Batch | Speedup | Memory ratio | Successful A/B | Failed A/B |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        speedup = row["candidate_speedup"]
        memory = row["candidate_memory_ratio"]
        report.append(
            "| {protocol} | {modality} | {polymer_residues} | {device} | {batch_size} | {speedup} | {memory} | {baseline_replicates}/{candidate_replicates} | {baseline_failures}/{candidate_failures} |".format(
                **row,
                speedup=f"{speedup:.3f}x" if speedup is not None else "pending",
                memory=f"{memory:.3f}x" if memory is not None else "pending",
            )
        )
    (args.output / "candidate_ab_report.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
