from __future__ import annotations

import json
import math
import statistics
from common import ROOT, write_rows

TERM_GROUPS = {
    "fa_atr": (("fa_ljatr",), ("fa_atr", "fa_intra_atr_xover4")),
    "fa_rep": (("fa_ljrep",), ("fa_rep", "fa_intra_rep_xover4")),
    "fa_sol": (("fa_lk",), ("fa_sol", "fa_intra_sol_xover4")),
    "fa_elec": (("fa_elec",), ("fa_elec", "fa_intra_elec")),
    "hbond": (("hbond",), ("hbond_sr_bb", "hbond_lr_bb", "hbond_bb_sc", "hbond_sc")),
    "lk_ball": (("lk_ball",), ("lk_ball",)),
    "lk_ball_iso": (("lk_ball_iso",), ("lk_ball_iso",)),
    "lk_bridge": (("lk_bridge",), ("lk_ball_bridge",)),
    "lk_bridge_uncpl": (("lk_bridge_uncpl",), ("lk_ball_bridge_uncpl",)),
    "dunbrack_rot": (("dunbrack_rot",), ("fa_dun_rot",)),
    "dunbrack_rotdev": (("dunbrack_rotdev",), ("fa_dun_dev",)),
    "dunbrack_semirot": (("dunbrack_semirot",), ("fa_dun_semi",)),
    "cart_bonded": (
        ("cart_lengths", "cart_angles", "cart_torsions", "cart_impropers"),
        ("cart_bonded",),
    ),
    "hxl_tors": (("cart_hxltorsions",), ("hxl_tors",)),
    "omega": (("omega",), ("omega",)),
    "rama": (("rama",), ("rama_prepro", "p_aa_pp")),
    "ref": (("ref",), ("ref",)),
    "disulfide": (("disulfide",), ("dslf_fa13",)),
    "gen_bonded": (("gen_torsions",), ("gen_bonded",)),
}


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def label(result: dict) -> str:
    if result["engine"] == "pyrosetta":
        return "PyRosetta CPU · B1"
    if result["device"] == "cpu":
        return "tmol CPU · B1"
    return f"tmol GPU · B{result['batch_size']}"


def failure_class(result: dict) -> str | None:
    if result.get("status") == "ok":
        return None
    error = result.get("error", "")
    if "out of memory" in error.lower():
        return "out_of_memory"
    if "negative dimension" in error:
        return "integer_index_overflow"
    if "status 124" in error:
        return "timeout"
    return "execution_error"


def main() -> None:
    results = []
    for path in sorted((ROOT / "results/raw").glob("*.json")):
        if path.name.startswith("pilot-"):
            continue
        result = json.loads(path.read_text())
        result["source_file"] = str(path)
        results.append(result)

    raw_rows = []
    summary_rows = []
    for result in results:
        common = {
            key: result.get(key)
            for key in (
                "engine",
                "engine_version",
                "engine_commit",
                "protocol",
                "device",
                "batch_size",
                "modality",
                "dataset_id",
                "residues",
                "polymer_residues",
                "atoms",
                "status",
                "error",
                "setup_seconds",
                "peak_memory_bytes",
                "process_peak_rss_bytes",
                "requested_cuda_execution",
                "score_function",
                "atom_type_set",
                "parameter_file_sha256",
                "ligand_reference_score_sha256",
                "ligand_reference_matched_terms",
                "ligand_reference_terms_within_tolerance",
                "ligand_reference_max_abs_error_reu",
                "initial_score_mean",
                "validation_score_mean",
                "validation_gradient_norm",
                "source_file",
            )
        }
        common["series"] = label(result)
        common["failure_class"] = failure_class(result)
        common["peak_working_memory_bytes"] = (
            result.get("peak_memory_bytes")
            if result.get("device") == "cuda"
            else result.get("process_peak_rss_bytes")
        )
        samples = result.get("seconds_per_structure_samples", [])
        if samples:
            for sample_index, seconds in enumerate(samples):
                raw_rows.append(
                    {
                        **common,
                        "sample": sample_index,
                        "seconds_per_structure": seconds,
                        "structures_per_second": 1.0 / seconds,
                    }
                )
            median = statistics.median(samples)
            summary_rows.append(
                {
                    **common,
                    "n_samples": len(samples),
                    "seconds_per_structure": median,
                    "seconds_q25": quantile(samples, 0.25),
                    "seconds_q75": quantile(samples, 0.75),
                    "structures_per_second": 1.0 / median,
                    "throughput_q25": 1.0 / quantile(samples, 0.75),
                    "throughput_q75": 1.0 / quantile(samples, 0.25),
                }
            )
        else:
            summary_rows.append({**common, "n_samples": 0})

    successful_by_key = {}
    for index, row in enumerate(summary_rows):
        if row["status"] != "ok" or not row.get("seconds_per_structure"):
            continue
        key = tuple(
            row.get(field)
            for field in (
                "engine",
                "engine_version",
                "protocol",
                "device",
                "batch_size",
                "modality",
                "dataset_id",
            )
        )
        if (
            key not in successful_by_key
            or row["seconds_per_structure"]
            < summary_rows[successful_by_key[key]]["seconds_per_structure"]
        ):
            successful_by_key[key] = index
    selected_indices = set(successful_by_key.values())
    for index, row in enumerate(summary_rows):
        row["selected_for_plot"] = index in selected_indices

    if raw_rows:
        write_rows(ROOT / "results/summary/timing_samples.csv", raw_rows)
    if summary_rows:
        write_rows(ROOT / "results/summary/timing_summary.csv", summary_rows)

    # Pair score-only, batch-one CPU term reports for each tmol version with the
    # single PyRosetta reference. Aggregations reflect how Rosetta splits terms
    # that tmol evaluates together.
    pyrosetta = {
        (r["dataset_id"], r["protocol"]): r
        for r in results
        if r["engine"] == "pyrosetta" and r["status"] == "ok"
    }
    agreement_rows = []
    total_agreement_rows = []
    for result in results:
        if not (
            result["engine"] == "tmol"
            and result["protocol"] == "score"
            and result["device"] == "cpu"
            and result["batch_size"] == 1
            and result["status"] == "ok"
        ):
            continue
        reference = pyrosetta.get((result["dataset_id"], "score"))
        if reference is None:
            continue
        if (
            result.get("validation_score_mean") is not None
            and reference.get("validation_score_mean") is not None
        ):
            total_agreement_rows.append(
                {
                    "tmol_version": result["engine_version"],
                    "tmol_commit": result["engine_commit"],
                    "modality": result["modality"],
                    "dataset_id": result["dataset_id"],
                    "polymer_residues": result["polymer_residues"],
                    "atoms": result["atoms"],
                    "pyrosetta_score_function": reference.get("score_function"),
                    "pyrosetta_reu": reference["validation_score_mean"],
                    "tmol_score": result["validation_score_mean"],
                }
            )
        tmol_terms = result.get("score_terms", {})
        rosetta_terms = reference.get("score_terms", {})
        for group_name, (tmol_group, rosetta_group) in TERM_GROUPS.items():
            tmol_present = [term for term in tmol_group if term in tmol_terms]
            if not tmol_present:
                continue
            present = [term for term in rosetta_group if term in rosetta_terms]
            if not present:
                continue
            agreement_rows.append(
                {
                    "tmol_version": result["engine_version"],
                    "tmol_commit": result["engine_commit"],
                    "modality": result["modality"],
                    "dataset_id": result["dataset_id"],
                    "residues": result["residues"],
                    "polymer_residues": result["polymer_residues"],
                    "atoms": result["atoms"],
                    "term": group_name,
                    "tmol_terms": "+".join(tmol_present),
                    "pyrosetta_terms": "+".join(present),
                    "pyrosetta_reu": sum(rosetta_terms[term] for term in present),
                    "tmol_score": sum(tmol_terms[term] for term in tmol_present),
                }
            )
    if agreement_rows:
        write_rows(ROOT / "results/summary/energy_agreement.csv", agreement_rows)
    if total_agreement_rows:
        write_rows(
            ROOT / "results/summary/total_energy_agreement.csv",
            total_agreement_rows,
        )


if __name__ == "__main__":
    main()
