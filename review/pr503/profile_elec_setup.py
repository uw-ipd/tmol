"""Paired electrostatic setup timings; no native scoring speedup is implied."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import tracemalloc
import types

import torch

from tmol.database import ParameterDatabase
from tmol.io import extended_pose_stack_from_sequences
from tmol.score.elec import ElecEnergyTerm, ElecParamResolver
from tmol.tests.score.elec.test_parameter_identity import isolated_charge_database

BASELINE = "20742270fd3b527a30e89ec7b0f3b0e42abfd5a6"


def load_baseline(path, name):
    source = subprocess.check_output(["git", "show", BASELINE + ":" + path], text=True)
    module = types.ModuleType(name)
    module.__package__ = "tmol.score.elec"
    sys.modules[name] = module
    exec(compile(source, path, "exec"), module.__dict__)
    return module


def main():  # noqa: C901 - benchmark stages share the same prepared pose
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    old_params = load_baseline("tmol/score/elec/_params.py", "_profile_old_elec_params")
    old_term = load_baseline(
        "tmol/score/elec/_elec_energy_term.py", "_profile_old_elec_term"
    )
    old_term.ElecParamResolver = old_params.ElecParamResolver
    database = ParameterDatabase.get_default()
    pose = extended_pose_stack_from_sequences(["ACDEFGHIKLMNPQRSTVWY"], device=device)
    pbt = pose.packed_block_types
    # Check all default chemical types, including combined terminal variants.
    from tmol.chemical import ResidueTypeSet

    restypes = ResidueTypeSet.from_database(database.chemical).residue_types
    resolvers = [
        cls.from_database(database.scoring.elec, device)
        for cls in (old_params.ElecParamResolver, ElecParamResolver)
    ]
    import numpy as np

    for bt in restypes:
        for method in (
            "get_partial_charges_for_block",
            "get_bonded_path_length_mapping_for_block",
        ):
            np.testing.assert_array_equal(*(getattr(r, method)(bt) for r in resolvers))

    terms = [cls(database, device) for cls in (old_term.ElecEnergyTerm, ElecEnergyTerm)]
    for term in terms:
        for bt in pbt.active_block_types:
            term.setup_block_type(bt)
        term.setup_packed_block_types(pbt)
        term.setup_poses(pose)
    for left, right in zip(*(term.get_score_term_attributes(pose) for term in terms)):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        else:
            assert left == right

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def measure(label, functions, iterations):
        seconds, peaks = {}, {}
        for name, function in functions.items():
            function()
            seconds[name] = []
        for repeat in range(7):
            names = list(functions)
            if repeat % 2:
                names.reverse()
            for name in names:
                synchronize()
                start = time.perf_counter()
                for _ in range(iterations):
                    functions[name]()
                synchronize()
                seconds[name].append((time.perf_counter() - start) / iterations)
        for name, function in functions.items():
            gc.collect()
            tracemalloc.start()
            function()
            synchronize()
            _, peaks[name] = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        medians = {name: statistics.median(values) for name, values in seconds.items()}
        row = dict(
            stage=label,
            iterations=iterations,
            seconds=seconds,
            median_seconds=medians,
            python_peak_bytes=peaks,
        )
        print(label, medians, flush=True)
        return row

    rows = [
        measure(
            "warm_score_attributes",
            {
                "baseline": lambda: terms[0].get_score_term_attributes(pose),
                "candidate": lambda: terms[1].get_score_term_attributes(pose),
            },
            200,
        )
    ]
    rows.append(
        measure(
            "term_constructor",
            {
                "baseline": lambda: old_term.ElecEnergyTerm(database, device),
                "candidate": lambda: ElecEnergyTerm(database, device),
            },
            30,
        )
    )

    alternatives = [
        ElecEnergyTerm(isolated_charge_database(database, q), device)
        for q in (0.5, 0.75)
    ]

    def switch(rebuild):
        for term in alternatives:
            if rebuild:
                for owner in (*pbt.active_block_types, pbt):
                    if hasattr(owner, "_elec_parameters"):
                        del owner._elec_parameters
            term.setup_packed_block_types(pbt)

    switch(False)
    expected = pbt._elec_parameters
    switch(True)
    for field in ("charges", "inter", "intra", "all_ligand_typed"):
        torch.testing.assert_close(
            getattr(expected, field),
            getattr(pbt._elec_parameters, field),
            rtol=0,
            atol=0,
        )
    rows.append(
        measure(
            "two_charge_switches",
            {
                "forced_rebuild": lambda: switch(True),
                "reuse_geometry": lambda: switch(False),
            },
            30,
        )
    )
    hashes = {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in (
            "tmol/score/elec/_params.py",
            "tmol/score/elec/_elec_energy_term.py",
        )
    }
    args.output.write_text(
        json.dumps(
            dict(
                baseline=BASELINE,
                device=str(device),
                torch=torch.__version__,
                default_types_checked=len(restypes),
                source_sha256=hashes,
                measurements=rows,
                scope="Warm setup only; traced Python peaks exclude tensor/native allocations. Charge-switch reference forces a correct rebuild using candidate code, not the stale baseline cache.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
