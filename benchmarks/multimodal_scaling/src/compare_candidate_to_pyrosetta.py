"""Project candidate performance against PyRosetta using paired TMol A/B data."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-version", default="0.1.55")
    args = parser.parse_args()

    candidate_rows = read_csv(args.candidate_summary)
    reference_rows = read_csv(args.reference_summary)
    reference: dict[tuple, dict[str, dict[str, str]]] = defaultdict(dict)
    pyro: dict[tuple, dict[str, str]] = {}
    for row in reference_rows:
        if row["status"] != "ok":
            continue
        dataset_key = (row["protocol"], row["modality"], row["dataset_id"])
        if row["engine"] == "pyrosetta" and row["device"] == "cpu":
            pyro[dataset_key] = row
        elif row["engine"] == "tmol" and row["engine_version"] == args.baseline_version:
            key = (*dataset_key, row["device"], int(row["batch_size"]))
            reference[key]["tmol"] = row

    rows: list[dict] = []
    for candidate in candidate_rows:
        speedup = number(candidate.get("candidate_speedup"))
        if speedup is None:
            continue
        dataset_key = (
            candidate["protocol"],
            candidate["modality"],
            candidate["dataset_id"],
        )
        key = (
            *dataset_key,
            candidate["device"],
            int(candidate["batch_size"]),
        )
        tmol_row = reference.get(key, {}).get("tmol")
        pyro_row = pyro.get(dataset_key)
        if tmol_row is None or pyro_row is None:
            continue
        tmol_seconds = number(tmol_row["seconds_per_structure"])
        pyro_seconds = number(pyro_row["seconds_per_structure"])
        if tmol_seconds is None or pyro_seconds is None:
            continue
        reference_ratio = pyro_seconds / tmol_seconds
        projected_ratio = reference_ratio * speedup
        rows.append(
            {
                "protocol": candidate["protocol"],
                "modality": candidate["modality"],
                "dataset_id": candidate["dataset_id"],
                "polymer_residues": candidate["polymer_residues"],
                "device": candidate["device"],
                "batch_size": candidate["batch_size"],
                "reference_pyrosetta_over_tmol": reference_ratio,
                "candidate_over_baseline_speedup": speedup,
                "projected_pyrosetta_over_candidate": projected_ratio,
                "candidate_faster_than_pyrosetta": projected_ratio > 1,
            }
        )

    if not rows:
        parser.error("no complete candidate/reference/PyRosetta triplets found")
    args.output.mkdir(parents=True, exist_ok=True)
    detail_path = args.output / "candidate_vs_pyrosetta.csv"
    with detail_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["protocol"],
                row["modality"],
                row["device"],
                row["batch_size"],
            )
        ].append(row["projected_pyrosetta_over_candidate"])
    report = [
        "# Candidate versus PyRosetta",
        "",
        "The projected ratio is `(PyRosetta / TMol 0.1.55)` from the broad "
        "reference matrix multiplied by `(TMol 0.1.55 / candidate)` from the "
        "same-node A–B–B–A run. Values above 1 mean the candidate is faster. "
        "This ratio-of-ratios avoids treating timing from two different jobs "
        "as a direct paired comparison.",
        "",
        "| Protocol | Modality | Device | Batch | Median PyRosetta/candidate | Candidate wins | Points |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for key, values in sorted(grouped.items()):
        report.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | "
            f"{statistics.median(values):.3f}x | "
            f"{sum(value > 1 for value in values)}/{len(values)} | {len(values)} |"
        )
    report.extend(
        [
            "",
            "## Per-input results",
            "",
            "| Protocol | Modality | Input | Residues | Device | Batch | Reference PyRosetta/TMol | Candidate A/B speedup | Projected PyRosetta/candidate |",
            "|---|---|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        rows,
        key=lambda item: (
            item["protocol"],
            item["modality"],
            item["device"],
            int(item["batch_size"]),
            int(item["polymer_residues"]),
        ),
    ):
        report.append(
            f"| {row['protocol']} | {row['modality']} | {row['dataset_id']} | "
            f"{row['polymer_residues']} | {row['device']} | {row['batch_size']} | "
            f"{row['reference_pyrosetta_over_tmol']:.3f}x | "
            f"{row['candidate_over_baseline_speedup']:.3f}x | "
            f"{row['projected_pyrosetta_over_candidate']:.3f}x |"
        )
    (args.output / "candidate_vs_pyrosetta.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
