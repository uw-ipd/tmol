"""Report candidate CPU gaps to PyRosetta across the frozen broad matrix."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_cpu_gaps import LENGTH_LABELS, length_bin, quartiles


KEYS = ("protocol", "modality", "dataset_id")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def median_seconds(result: dict) -> float | None:
    values = result.get("seconds_per_structure_samples", [])
    return statistics.median(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path)
    args = parser.parse_args()

    reference_summary = args.reference_summary or (
        args.candidate_root / "metadata" / "frozen_reference_timing_summary.csv"
    )

    provenance_path = (
        args.candidate_root / "metadata" / "broad_candidate_cpu_provenance.json"
    )
    provenance = json.loads(provenance_path.read_text())
    candidate_commit = provenance["candidate"]["source"]["revision"]

    all_reference_rows = read_csv(reference_summary)
    reference_rows = [
        row
        for row in all_reference_rows
        if row["status"] == "ok" and row["selected_for_plot"].lower() == "true"
    ]
    pyro = {
        tuple(row[key] for key in KEYS): row
        for row in reference_rows
        if row["engine"] == "pyrosetta" and row["device"] == "cpu"
    }
    baseline = {
        tuple(row[key] for key in KEYS): row
        for row in reference_rows
        if row["engine"] == "tmol"
        and row["engine_version"] == "0.1.55"
        and row["device"] == "cpu"
        and int(row["batch_size"]) == 1
    }
    baseline_records = {
        tuple(row[key] for key in KEYS): row
        for row in all_reference_rows
        if row["engine"] == "tmol"
        and row["engine_version"] == "0.1.55"
        and row["device"] == "cpu"
        and int(row["batch_size"]) == 1
    }

    expected_keys = set()
    expected_counts = {}
    for protocol in ("score_gradient", "fastrelax"):
        table = (
            args.candidate_root
            / "metadata"
            / f"tasks-broad-candidate-{protocol}-cpu.tsv"
        )
        if not table.exists():
            continue
        with table.open(newline="") as handle:
            tasks = list(csv.DictReader(handle, delimiter="\t"))
        expected_counts[protocol] = len(tasks)
        expected_keys.update(
            (row["protocol"], row["modality"], row["dataset_id"]) for row in tasks
        )

    detail = []
    failures = []
    observed_keys = []
    foreign_records = []
    candidate_records = {}
    for path in sorted((args.candidate_root / "raw").glob("*.json")):
        result = json.loads(path.read_text())
        if result.get("engine_commit") != candidate_commit:
            foreign_records.append(str(path))
            continue
        key = tuple(result.get(field) for field in KEYS)
        observed_keys.append(key)
        candidate_records[key] = result
        if result.get("status") != "ok":
            failures.append({"source_file": str(path), **result})
            continue
        seconds = median_seconds(result)
        if seconds is None or key not in pyro or key not in baseline:
            continue
        pyro_seconds = float(pyro[key]["seconds_per_structure"])
        baseline_seconds = float(baseline[key]["seconds_per_structure"])
        residues = int(float(result["polymer_residues"]))
        detail.append(
            {
                "protocol": key[0],
                "modality": key[1],
                "dataset_id": key[2],
                "polymer_residues": residues,
                "length_bin": length_bin(residues),
                "pyrosetta_seconds": pyro_seconds,
                "baseline_tmol_seconds": baseline_seconds,
                "candidate_tmol_seconds": seconds,
                "candidate_over_baseline_speedup": baseline_seconds / seconds,
                "pyrosetta_over_candidate": pyro_seconds / seconds,
                "candidate_faster_than_pyrosetta": pyro_seconds > seconds,
                "candidate_process_peak_rss_bytes": result.get(
                    "process_peak_rss_bytes"
                ),
                "source_file": str(path),
            }
        )

    output = args.candidate_root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    if detail:
        detail.sort(
            key=lambda row: (row["protocol"], row["modality"], row["polymer_residues"])
        )
        with (output / "candidate_cpu_gap_inputs.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=detail[0].keys())
            writer.writeheader()
            writer.writerows(detail)

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in detail:
        grouped[(row["protocol"], row["modality"], row["length_bin"])].append(row)
    rows = []
    for key, group in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1], LENGTH_LABELS.index(item[0][2])),
    ):
        ratios = [row["pyrosetta_over_candidate"] for row in group]
        speedups = [row["candidate_over_baseline_speedup"] for row in group]
        q25, q75 = quartiles(ratios)
        rows.append(
            {
                "protocol": key[0],
                "modality": key[1],
                "length_bin": key[2],
                "n": len(group),
                "median_pyrosetta_over_candidate": statistics.median(ratios),
                "q25_pyrosetta_over_candidate": q25,
                "q75_pyrosetta_over_candidate": q75,
                "candidate_wins": sum(
                    row["candidate_faster_than_pyrosetta"] for row in group
                ),
                "median_candidate_over_baseline_speedup": statistics.median(speedups),
            }
        )
    if rows:
        with (output / "candidate_cpu_gap_by_length.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    observed_set = set(observed_keys)
    duplicate_keys = len(observed_keys) - len(observed_set)
    missing_keys = sorted(expected_keys - observed_set)
    unexpected_keys = sorted(observed_set - expected_keys)
    candidate_only_failures = []
    baseline_only_failures = []
    shared_failures = []
    for key in sorted(expected_keys):
        candidate_record = candidate_records.get(key)
        baseline_record = baseline_records.get(key)
        candidate_status = (
            candidate_record.get("status", "missing")
            if candidate_record is not None
            else "missing"
        )
        baseline_status = (
            baseline_record.get("status", "missing")
            if baseline_record is not None
            else "missing"
        )
        candidate_failed = candidate_status != "ok"
        baseline_failed = baseline_status != "ok"
        comparison = {
            "protocol": key[0],
            "modality": key[1],
            "dataset_id": key[2],
            "candidate_status": candidate_status,
            "baseline_status": baseline_status,
        }
        if candidate_failed and baseline_failed:
            shared_failures.append(comparison)
        elif candidate_failed:
            candidate_only_failures.append(comparison)
        elif baseline_failed:
            baseline_only_failures.append(comparison)

    failure_comparison = {
        "reference": "frozen tmol 0.1.55 CPU B1",
        "candidate_only": candidate_only_failures,
        "baseline_only": baseline_only_failures,
        "shared": shared_failures,
    }
    actual = {
        protocol: sum(row["protocol"] == protocol for row in detail)
        for protocol in expected_counts
    }
    audit = {
        "candidate_commit": candidate_commit,
        "expected_successful_or_failed_records": expected_counts,
        "successful_paired_records": actual,
        "failed_records": len(failures),
        "failure_comparison": failure_comparison,
        "duplicate_measurement_keys": duplicate_keys,
        "missing_measurement_keys": missing_keys,
        "unexpected_measurement_keys": unexpected_keys,
        "foreign_commit_records": foreign_records,
        "complete": not missing_keys
        and not unexpected_keys
        and duplicate_keys == 0
        and not foreign_records,
        "all_successful": not failures
        and not missing_keys
        and not unexpected_keys
        and duplicate_keys == 0
        and not foreign_records,
    }
    (output / "candidate_cpu_gap_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )

    report = [
        "# Broad candidate CPU gaps versus PyRosetta",
        "",
        "Ratios are PyRosetta time divided by candidate TMol time for the same frozen input and protocol. Values above 1 mean the candidate is faster. Candidate and reference measurements are from separate scheduler allocations; use the matched A–B report for causal speedup claims.",
        "",
        f"Coverage: {sum(actual.values())}/{sum(expected_counts.values())} successful paired records; {len(failures)} explicit failures.",
        "Failure delta versus frozen TMol 0.1.55: "
        f"{len(candidate_only_failures)} candidate-only, "
        f"{len(baseline_only_failures)} baseline-only, and "
        f"{len(shared_failures)} shared failures.",
        "",
        "| Protocol | Modality | Residues | PyRosetta/candidate median (IQR) | Candidate wins | Candidate/0.1.55 median |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['protocol']} | {row['modality']} | {row['length_bin']} | "
            f"{row['median_pyrosetta_over_candidate']:.3f}x "
            f"({row['q25_pyrosetta_over_candidate']:.3f}–{row['q75_pyrosetta_over_candidate']:.3f}x) | "
            f"{row['candidate_wins']}/{row['n']} | "
            f"{row['median_candidate_over_baseline_speedup']:.3f}x |"
        )
    (output / "candidate_cpu_gap_by_length.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
