"""Summarize where latest TMol CPU remains behind PyRosetta by pose length."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

from common import ROOT


LENGTH_LABELS = ("<100", "100–299", "300–599", "600–999", "≥1000")


def length_bin(residues: int) -> str:
    if residues < 100:
        return LENGTH_LABELS[0]
    if residues < 300:
        return LENGTH_LABELS[1]
    if residues < 600:
        return LENGTH_LABELS[2]
    if residues < 1000:
        return LENGTH_LABELS[3]
    return LENGTH_LABELS[4]


def quartiles(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    q25, _, q75 = statistics.quantiles(values, n=4, method="inclusive")
    return q25, q75


def main() -> None:
    summary_dir = ROOT / "results" / "summary"
    with (summary_dir / "timing_summary.csv").open(newline="") as handle:
        timing = list(csv.DictReader(handle))
    keys = ["protocol", "modality", "dataset_id"]
    valid = [
        row
        for row in timing
        if row["status"] == "ok" and row["selected_for_plot"].lower() == "true"
    ]
    pyro = {
        tuple(row[key] for key in keys): row
        for row in valid
        if row["engine"] == "pyrosetta" and row["device"] == "cpu"
    }
    tmol = {
        tuple(row[key] for key in keys): row
        for row in valid
        if row["engine"] == "tmol"
        and row["engine_version"] == "0.1.55"
        and row["device"] == "cpu"
        and int(row["batch_size"]) == 1
    }
    paired = []
    for key in sorted(pyro.keys() & tmol.keys()):
        pyro_seconds = float(pyro[key]["seconds_per_structure"])
        tmol_seconds = float(tmol[key]["seconds_per_structure"])
        residues = int(float(pyro[key]["polymer_residues"]))
        ratio = pyro_seconds / tmol_seconds
        paired.append(
            {
                "protocol": key[0],
                "modality": key[1],
                "dataset_id": key[2],
                "polymer_residues": residues,
                "length_bin": length_bin(residues),
                "pyrosetta_seconds": pyro_seconds,
                "tmol_seconds": tmol_seconds,
                "pyrosetta_over_tmol": ratio,
                "tmol_faster": ratio > 1,
            }
        )
    paired.sort(
        key=lambda row: (
            row["protocol"],
            row["modality"],
            row["polymer_residues"],
        )
    )
    with (summary_dir / "cpu_gap_inputs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=paired[0].keys())
        writer.writeheader()
        writer.writerows(paired)

    rows = []
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in paired:
        groups[(row["protocol"], row["modality"], row["length_bin"])].append(row)
    for (protocol, modality, bin_label), group in sorted(
        groups.items(),
        key=lambda item: (item[0][0], item[0][1], LENGTH_LABELS.index(item[0][2])),
    ):
        ratios = [row["pyrosetta_over_tmol"] for row in group]
        q25, q75 = quartiles(ratios)
        rows.append(
            {
                "protocol": protocol,
                "modality": modality,
                "length_bin": bin_label,
                "n": len(group),
                "median_pyrosetta_over_tmol": statistics.median(ratios),
                "q25_pyrosetta_over_tmol": q25,
                "q75_pyrosetta_over_tmol": q75,
                "tmol_wins": sum(row["tmol_faster"] for row in group),
                "tmol_win_fraction": sum(row["tmol_faster"] for row in group)
                / len(group),
            }
        )
    with (summary_dir / "cpu_gap_by_length.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    report = [
        "# Latest TMol CPU gaps versus PyRosetta",
        "",
        "Ratios are PyRosetta time divided by TMol 0.1.55 time for the same "
        "input and protocol. Values above 1 mean TMol is faster; values below "
        "1 identify a remaining TMol latency gap.",
        "",
        "| Protocol | Modality | Residues | Median ratio | IQR | TMol wins |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['protocol']} | {row['modality']} | {row['length_bin']} | "
            f"{row['median_pyrosetta_over_tmol']:.3f}x | "
            f"{row['q25_pyrosetta_over_tmol']:.3f}–"
            f"{row['q75_pyrosetta_over_tmol']:.3f}x | "
            f"{row['tmol_wins']}/{row['n']} |"
        )
    (summary_dir / "cpu_gap_by_length.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
