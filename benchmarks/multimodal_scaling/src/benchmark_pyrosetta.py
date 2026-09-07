from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from pathlib import Path

from common import (
    SPEC,
    benchmark_callable,
    data_path,
    machine_metadata,
    process_peak_rss_bytes,
    read_manifest,
    sha256,
)


def select(dataset_id: str, modality: str | None = None) -> dict[str, str]:
    matches = [
        row
        for row in read_manifest()
        if row["dataset_id"] == dataset_id
        and row["status"] == "ok"
        and (modality is None or row["modality"] == modality)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one included dataset named {dataset_id!r}, got {len(matches)}"
        )
    return matches[0]


def initialize(row: dict[str, str], pyrosetta_path: Path):
    import sys

    sys.path.insert(0, str(pyrosetta_path))
    import pyrosetta

    options = [
        "-mute all",
        "-ignore_waters true",
        "-load_PDB_components false",
        "-in:file:obey_ENDMDL true",
        "-multithreading:total_threads 1",
        f"-run:constant_seed -run:jran {SPEC['seed']}",
    ]
    if row["modality"] == "protein_ligand":
        options.append("-corrections::gen_potential true")
        options.append(f"-extra_res_fa {data_path(row['ligand_rosetta_params'])}")
    else:
        options.append("-corrections::beta_nov16 true")
    pyrosetta.init(" ".join(options))
    return pyrosetta


def weighted_terms(pyrosetta, pose, score_function) -> dict[str, float]:
    score_function(pose)
    energies = pose.energies().total_energies()
    weights = score_function.weights()
    terms = {}
    for score_type in score_function.get_nonzero_weighted_scoretypes():
        name = pyrosetta.rosetta.core.scoring.name_from_score_type(score_type)
        terms[name] = float(energies[score_type] * weights[score_type])
    return terms


def validate_ligand_reference(
    structure_path: str, measured: dict[str, float]
) -> tuple[str, int, int, float, dict[str, float]]:
    reference_path = str(Path(structure_path).with_suffix(".sc"))
    lines = [
        line.split()[1:]
        for line in Path(reference_path).read_text().splitlines()
        if line.startswith("SCORE:")
    ]
    if len(lines) < 2:
        raise RuntimeError(f"no score table in {reference_path}")
    reference = dict(zip(lines[0], lines[-1]))
    differences = {
        name: abs(value - float(reference[name]))
        for name, value in measured.items()
        if name in reference
    }
    if len(differences) < 20:
        raise RuntimeError(
            f"only {len(differences)} matched weighted terms in {reference_path}"
        )
    tolerance = 0.002
    within_tolerance = sum(value <= tolerance for value in differences.values())
    required = math.ceil(0.95 * len(differences))
    gen_bonded_error = differences.get("gen_bonded", math.inf)
    if within_tolerance < required or gen_bonded_error > tolerance:
        raise RuntimeError(
            f"only {within_tolerance}/{len(differences)} terms reproduce "
            f"{reference_path} within {tolerance} REU; "
            f"gen_bonded error is {gen_bonded_error:.6g} REU"
        )
    outliers = {name: value for name, value in differences.items() if value > tolerance}
    return (
        reference_path,
        len(differences),
        within_tolerance,
        max(differences.values()),
        outliers,
    )


def cartesian_score_gradient_operation(pyrosetta, pose, score_function):
    rosetta = pyrosetta.rosetta
    # Match Rosetta's CartesianMinimizer setup, while timing only the C++
    # objective and derivative evaluations rather than a Python atom loop.
    score_function(pose)
    move_map = rosetta.core.kinematics.MoveMap()
    move_map.set_bb(True)
    move_map.set_chi(True)
    move_map.set_jump(True)
    min_map = rosetta.core.optimization.CartesianMinimizerMap()
    min_map.setup(pose, move_map)
    score_function.setup_for_minimizing(pose, min_map)
    rosetta.core.optimization.activate_dof_deriv_terms_for_cart_min(
        pose, score_function, min_map
    )
    variables = rosetta.utility.vector1_double(min_map.ndofs())
    min_map.copy_dofs_from_pose(pose, variables)
    derivatives = rosetta.utility.vector1_double(min_map.ndofs())
    objective = rosetta.core.optimization.CartesianMultifunc(
        pose, min_map, score_function
    )

    def operation():
        # CartesianMultifunc holds a C++ reference, not shared ownership.
        # Capturing the map keeps it alive for the complete timed loop.
        _ = min_map
        objective(variables)
        objective.dfunc(variables, derivatives)

    return operation, derivatives, min_map.ndofs()


def benchmark_score(pyrosetta, pose, score_function, gradient: bool):
    setup_start = time.perf_counter()
    derivatives = None
    n_dofs = 0
    if gradient:
        operation, derivatives, n_dofs = cartesian_score_gradient_operation(
            pyrosetta, pose, score_function
        )
    else:

        def operation():
            # ScoreFunction caches energies for an unchanged Pose. Drop only
            # those values so every timed call performs a complete rescore.
            pose.energies().clear_energies()
            score_function(pose)

    setup_seconds = time.perf_counter() - setup_start
    timings, iterations = benchmark_callable(operation, device="cpu")
    validation_gradient_norm = None
    if derivatives is not None:
        operation()
        validation_gradient_norm = math.sqrt(
            sum(float(derivatives[index]) ** 2 for index in range(1, n_dofs + 1))
        )
        if not math.isfinite(validation_gradient_norm):
            raise RuntimeError("non-finite Cartesian gradient")
    validation_score = float(score_function(pose))
    if not math.isfinite(validation_score):
        raise RuntimeError("non-finite score")
    terms = weighted_terms(pyrosetta, pose, score_function)
    if gradient:
        score_function.finalize_after_minimizing(pose)
    return {
        "seconds_per_batch_samples": timings,
        "seconds_per_structure_samples": timings,
        "iterations_per_sample": iterations,
        "setup_seconds": setup_seconds,
        "score_terms": terms,
        "peak_memory_bytes": None,
        "process_peak_rss_bytes": process_peak_rss_bytes(),
        "validation_score_mean": validation_score,
        "validation_gradient_norm": validation_gradient_norm,
    }


def benchmark_fastrelax(pyrosetta, pose, score_function):
    setup_start = time.perf_counter()
    relax = pyrosetta.rosetta.protocols.relax.FastRelax(
        score_function, SPEC["fastrelax_repeats"]
    )
    script_path = (
        Path(pyrosetta._rosetta_database_from_env())
        / "sampling/relax_scripts/MonomerRelax2019.txt"
    )
    script_lines = pyrosetta.rosetta.std.vector_std_string()
    for line in script_path.read_text().splitlines():
        line = line.replace("%%nrepeats%%", str(SPEC["fastrelax_repeats"]))
        if line.strip():
            script_lines.append(line)
    relax.set_script_from_lines(script_lines)
    relax.cartesian(True)
    setup_seconds = time.perf_counter() - setup_start
    working_pose = pose.clone()
    initial_score = float(score_function(working_pose))
    start = time.perf_counter()
    relax.apply(working_pose)
    elapsed = time.perf_counter() - start
    final_score = float(score_function(working_pose))
    return {
        "seconds_per_batch_samples": [elapsed],
        "seconds_per_structure_samples": [elapsed],
        "iterations_per_sample": 1,
        "setup_seconds": setup_seconds,
        "score_terms": {},
        "peak_memory_bytes": None,
        "process_peak_rss_bytes": process_peak_rss_bytes(),
        "initial_score_mean": initial_score,
        "validation_score_mean": final_score,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--modality", choices=("protein", "protein_ligand", "protein_nucleic")
    )
    parser.add_argument(
        "--protocol", choices=("score", "score_gradient", "fastrelax"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pyrosetta-path",
        type=Path,
        default=Path("/mnt/home/kdidi/projects/pyrosetta-2024.39"),
    )
    args = parser.parse_args()
    os.environ.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    row = select(args.dataset, args.modality)
    score_function_name = (
        "beta_genpot_cart" if row["modality"] == "protein_ligand" else "beta_nov16_cart"
    )
    result = {
        "engine": "pyrosetta",
        "engine_version": "2024.39",
        "engine_commit": "59628fbc5bc09f1221e1642f1f8d157ce49b1410",
        "protocol": args.protocol,
        "device": "cpu",
        "batch_size": 1,
        "modality": row["modality"],
        "dataset_id": row["dataset_id"],
        "residues": int(row["residues"]),
        "polymer_residues": sum(
            int(row[key])
            for key in ("protein_residues", "dna_residues", "rna_residues")
        ),
        "atoms": int(row["atoms"]),
        "status": "ok",
        "error": "",
        "score_function": score_function_name,
        "atom_type_set": (
            "fa_standard_genpot"
            if row["modality"] == "protein_ligand"
            else "fa_standard"
        ),
        "parameter_file_sha256": (
            sha256(data_path(row["ligand_rosetta_params"]))
            if row["modality"] == "protein_ligand"
            else None
        ),
        "machine": machine_metadata(),
    }
    try:
        pyrosetta = initialize(row, args.pyrosetta_path)
        structure_path = data_path(row["structure_path"])
        pose = pyrosetta.pose_from_file(str(structure_path))
        loaded_residues = pose.total_residue()
        if loaded_residues != int(row["residues"]):
            raise RuntimeError(
                f"loaded {loaded_residues} residues; manifest records {row['residues']}"
            )
        score_function = pyrosetta.create_score_function(score_function_name)
        if args.protocol == "fastrelax":
            result.update(benchmark_fastrelax(pyrosetta, pose, score_function))
        else:
            result.update(
                benchmark_score(
                    pyrosetta,
                    pose,
                    score_function,
                    gradient=args.protocol == "score_gradient",
                )
            )
            reference_path = structure_path.with_suffix(".sc")
            if row["modality"] == "protein_ligand" and reference_path.is_file():
                (
                    reference_path,
                    matched_terms,
                    terms_within_tolerance,
                    maximum_error,
                    outliers,
                ) = validate_ligand_reference(
                    str(structure_path), result["score_terms"]
                )
                result["ligand_reference_score_path"] = reference_path
                result["ligand_reference_score_sha256"] = sha256(Path(reference_path))
                result["ligand_reference_matched_terms"] = matched_terms
                result["ligand_reference_terms_within_tolerance"] = (
                    terms_within_tolerance
                )
                result["ligand_reference_max_abs_error_reu"] = maximum_error
                result["ligand_reference_outliers_reu"] = outliers
        result["pyrosetta_version_string"] = pyrosetta.version().splitlines()[-1]
    except Exception as error:
        result.update(
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
        result["process_peak_rss_bytes"] = process_peak_rss_bytes()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
