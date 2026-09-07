from __future__ import annotations

import json
import math

import pandas as pd
from scipy.stats import pearsonr, spearmanr

from common import ROOT, read_manifest, write_rows

RECORD_KEY = [
    "engine",
    "engine_version",
    "protocol",
    "device",
    "batch_size",
    "modality",
    "dataset_id",
    "requested_cuda_execution",
]
LEGACY_FASTRELAX_DATASETS = {
    "protein": ("5uoi", "5yzf", "5m4a", "1o7j"),
    "protein_ligand": ("hsp90", "p38", "src", "ace"),
    "protein_nucleic": ("5exh", "1ysa", "3ndh", "6q1h"),
}


def performance_rows(timing: pd.DataFrame) -> pd.DataFrame:
    return timing[(timing.status == "ok") & timing.selected_for_plot].copy()


def speedups(timing: pd.DataFrame) -> list[dict]:
    selected = performance_rows(timing)
    keys = ["modality", "dataset_id", "protocol"]
    pyrosetta = selected[selected.engine == "pyrosetta"][
        [*keys, "seconds_per_structure"]
    ].rename(columns={"seconds_per_structure": "reference_seconds"})
    rows: list[dict] = []
    for _, row in (
        selected[selected.engine == "tmol"]
        .merge(pyrosetta, on=keys, how="inner")
        .iterrows()
    ):
        rows.append(
            {
                **{key: row[key] for key in keys},
                "comparison": "pyrosetta_over_tmol",
                "tmol_version": row.engine_version,
                "series": row.series,
                "speedup": row.reference_seconds / row.seconds_per_structure,
            }
        )

    tmol = selected[selected.engine == "tmol"].copy()
    latest = tmol[tmol.engine_version == "0.1.55"]
    historical = tmol[tmol.engine_version.isin(["0.1.46", "0.1.47"])]
    pair_keys = [*keys, "series"]
    paired = latest.merge(
        historical,
        on=pair_keys,
        how="inner",
        suffixes=("_latest", "_historical"),
    )
    for _, row in paired.iterrows():
        rows.append(
            {
                **{key: row[key] for key in keys},
                "comparison": "historical_over_latest",
                "tmol_version": row.engine_version_latest,
                "historical_version": row.engine_version_historical,
                "series": row.series,
                "speedup": (
                    row.seconds_per_structure_historical
                    / row.seconds_per_structure_latest
                ),
            }
        )
    return rows


def agreement_statistics(agreement: pd.DataFrame) -> list[dict]:
    rows = []
    for (modality, version), group in agreement.groupby(
        ["modality", "tmol_version"], sort=True
    ):
        paired = group[["pyrosetta_reu", "tmol_score"]].dropna()
        can_correlate = (
            len(paired) >= 2
            and paired.pyrosetta_reu.nunique() >= 2
            and paired.tmol_score.nunique() >= 2
        )
        rows.append(
            {
                "modality": modality,
                "tmol_version": version,
                "n": len(paired),
                "pearson_r": (
                    pearsonr(paired.pyrosetta_reu, paired.tmol_score).statistic
                    if can_correlate
                    else math.nan
                ),
                "spearman_rho": (
                    spearmanr(paired.pyrosetta_reu, paired.tmol_score).statistic
                    if can_correlate
                    else math.nan
                ),
                "mean_absolute_score_difference": (
                    paired.pyrosetta_reu - paired.tmol_score
                )
                .abs()
                .mean(),
            }
        )
    return rows


def key(values) -> tuple:
    normalized = []
    for field, value in zip(RECORD_KEY, values, strict=True):
        if pd.isna(value):
            value = ""
        elif field == "batch_size":
            value = int(value)
        else:
            value = str(value)
        normalized.append(value)
    return tuple(normalized)


def expected_measurements() -> set[tuple]:
    datasets = [row for row in read_manifest() if row["status"] == "ok"]
    expected = set()

    def add(engine, version, protocol, device, batch, modality, dataset, execution):
        expected.add(
            key(
                (engine, version, protocol, device, batch, modality, dataset, execution)
            )
        )

    for row in datasets:
        for protocol in ("score_gradient",):
            add(
                "pyrosetta",
                "2024.39",
                protocol,
                "cpu",
                1,
                row["modality"],
                row["dataset_id"],
                "",
            )
            for version in ("0.1.55", row["historical_version"]):
                add(
                    "tmol",
                    version,
                    protocol,
                    "cpu",
                    1,
                    row["modality"],
                    row["dataset_id"],
                    "eager",
                )
                for batch in (1, 10, 100, 1000):
                    add(
                        "tmol",
                        version,
                        protocol,
                        "cuda",
                        batch,
                        row["modality"],
                        row["dataset_id"],
                        "eager",
                    )
                    if version == "0.1.55":
                        add(
                            "tmol",
                            version,
                            protocol,
                            "cuda",
                            batch,
                            row["modality"],
                            row["dataset_id"],
                            "graph",
                        )

    fastrelax_rows = [
        row
        for row in datasets
        if row.get("fastrelax", "").lower() in {"yes", "true", "1"}
        or (
            not row.get("fastrelax")
            and row["dataset_id"] in LEGACY_FASTRELAX_DATASETS.get(row["modality"], ())
        )
    ]
    for row in fastrelax_rows:
        modality = row["modality"]
        dataset_id = row["dataset_id"]
        add("pyrosetta", "2024.39", "fastrelax", "cpu", 1, modality, dataset_id, "")
        for version in ("0.1.55", row["historical_version"]):
            add("tmol", version, "fastrelax", "cpu", 1, modality, dataset_id, "auto")
            for batch in (1, 10, 100, 1000):
                add(
                    "tmol",
                    version,
                    "fastrelax",
                    "cuda",
                    batch,
                    modality,
                    dataset_id,
                    "auto",
                )
    return expected


def integrity_counts(timing: pd.DataFrame) -> tuple[int, int]:
    duplicate_records = int(timing.duplicated(RECORD_KEY, keep=False).sum())
    invalid_records = 0
    for row in timing[timing.status == "ok"].itertuples():
        if (
            not math.isfinite(row.seconds_per_structure)
            or row.seconds_per_structure <= 0
        ):
            invalid_records += 1
            continue
        if not math.isfinite(row.validation_score_mean):
            invalid_records += 1
            continue
        if row.protocol == "score_gradient" and not math.isfinite(
            row.validation_gradient_norm
        ):
            invalid_records += 1
    return duplicate_records, invalid_records


def numerical_consistency(timing: pd.DataFrame) -> list[str]:
    """Check that batching/device choice does not change validated numerics."""
    scoring = timing[
        (timing.status == "ok") & timing.protocol.isin(["score", "score_gradient"])
    ]
    issues = []
    keys = ["engine", "engine_version", "dataset_id"]
    for key, group in scoring.groupby(keys, sort=True):
        scores = group.validation_score_mean.dropna()
        if len(scores) > 1:
            # Different FP32 device/batch reductions may differ by a few
            # hundredths of a score unit for large, strongly cancelling poses.
            tolerance = max(5e-2, 1e-4 * max(1.0, abs(scores.median())))
            if scores.max() - scores.min() > tolerance:
                issues.append(f"{key}: validation score differs across execution modes")

        gradients = group[group.protocol == "score_gradient"].copy()
        if len(gradients) > 1:
            # A repeated batch's total gradient norm grows as sqrt(batch size).
            normalized = gradients.validation_gradient_norm / gradients.batch_size.pow(
                0.5
            )
            tolerance = max(1e-2, 1e-4 * max(1.0, abs(normalized.median())))
            if normalized.max() - normalized.min() > tolerance:
                issues.append(
                    f"{key}: per-structure gradient norm differs across modes"
                )
    return issues


def main() -> None:
    summary_dir = ROOT / "results/summary"
    timing = pd.read_csv(summary_dir / "timing_summary.csv")
    speedup_rows = speedups(timing)
    write_rows(summary_dir / "speedups.csv", speedup_rows)
    agreement_rows = []
    total_agreement_rows = []
    for source_name, output_name in (
        ("energy_agreement.csv", "energy_agreement_statistics.csv"),
        ("total_energy_agreement.csv", "total_energy_agreement_statistics.csv"),
    ):
        source = summary_dir / source_name
        if source.exists():
            statistics = agreement_statistics(pd.read_csv(source))
            write_rows(
                summary_dir / output_name,
                statistics,
            )
            if source_name == "energy_agreement.csv":
                agreement_rows = statistics
            else:
                total_agreement_rows = statistics

    expected = expected_measurements()
    actual = {key(row) for row in timing[RECORD_KEY].itertuples(index=False, name=None)}
    missing_measurements = sorted(expected - actual)
    unexpected_measurements = sorted(actual - expected)
    expected_scoring = sum(item[2] != "fastrelax" for item in expected)
    expected_fastrelax = sum(item[2] == "fastrelax" for item in expected)
    actual_scoring = int(timing.protocol.isin(["score", "score_gradient"]).sum())
    actual_fastrelax = int((timing.protocol == "fastrelax").sum())
    failures = timing[timing.status != "ok"]
    duplicate_records, invalid_records = integrity_counts(timing)
    consistency_issues = numerical_consistency(timing)
    report = [
        "# Benchmark audit",
        "",
        f"- Scoring records: {actual_scoring}/{expected_scoring} expected.",
        f"- FastRelax records: {actual_fastrelax}/{expected_fastrelax} expected.",
        f"- Successful records: {(timing.status == 'ok').sum()}.",
        f"- Explicit failures: {len(failures)}.",
        f"- Duplicate measurement keys: {duplicate_records}.",
        f"- Missing expected measurement keys: {len(missing_measurements)}.",
        f"- Unexpected measurement keys: {len(unexpected_measurements)}.",
        f"- Successful records with invalid numerical checks: {invalid_records}.",
        f"- Cross-mode numerical consistency issues: {len(consistency_issues)}.",
        "",
        "## Median speedups",
        "",
    ]
    speedup_frame = pd.DataFrame(speedup_rows)
    for keys, group in speedup_frame.groupby(
        ["comparison", "modality", "protocol", "series", "tmol_version"],
        sort=True,
    ):
        comparison, modality, protocol, series, tmol_version = keys
        version_label = f"tmol {tmol_version}"
        historical_versions = group.get("historical_version")
        if historical_versions is not None and historical_versions.notna().any():
            versions = "/".join(sorted(historical_versions.dropna().unique()))
            version_label = f"tmol {tmol_version} over {versions}"
        report.append(
            f"- {comparison}; {version_label}; {modality}; {protocol}; {series}: "
            f"{group.speedup.median():.3g}× (n={len(group)})."
        )
    report.extend(["", "## Total-energy agreement", ""])
    for row in total_agreement_rows:
        report.append(
            f"- {row['modality']}; tmol {row['tmol_version']}: "
            f"Pearson r={row['pearson_r']:.4f}, "
            f"Spearman ρ={row['spearman_rho']:.4f}, n={row['n']}."
        )
    report.extend(["", "## Matched-term energy agreement", ""])
    for row in agreement_rows:
        report.append(
            f"- {row['modality']}; tmol {row['tmol_version']}: "
            f"Pearson r={row['pearson_r']:.4f}, "
            f"Spearman ρ={row['spearman_rho']:.4f}, n={row['n']}."
        )
    if len(failures):
        report.extend(["", "## Failed or unsupported points", ""])
        for _, row in failures.iterrows():
            error = str(row.error)
            if len(error) > 180:
                error = error[:177] + "..."
            report.append(
                f"- {row.dataset_id}; {row.protocol}; {row.engine} "
                f"{row.engine_version}; {row.device}; B{row.batch_size}; "
                f"{row.failure_class}: {error}"
            )
    if consistency_issues:
        report.extend(["", "## Numerical consistency issues", ""])
        report.extend(f"- {issue}" for issue in consistency_issues)
    (summary_dir / "benchmark_report.md").write_text("\n".join(report) + "\n")

    audit = {
        "expected_scoring_records": expected_scoring,
        "actual_scoring_records": actual_scoring,
        "expected_fastrelax_records": expected_fastrelax,
        "actual_fastrelax_records": actual_fastrelax,
        "successful_records": int((timing.status == "ok").sum()),
        "failed_records": len(failures),
        "duplicate_measurement_keys": duplicate_records,
        "missing_measurement_keys": missing_measurements,
        "unexpected_measurement_keys": unexpected_measurements,
        "invalid_successful_records": invalid_records,
        "numerical_consistency_issues": consistency_issues,
        "complete": actual_scoring == expected_scoring
        and actual_fastrelax == expected_fastrelax
        and duplicate_records == 0
        and not missing_measurements
        and not unexpected_measurements
        and invalid_records == 0
        and not consistency_issues,
    }
    (summary_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
