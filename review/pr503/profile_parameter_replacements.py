"""Compare guarded installation with the previous private installer.

Both paths install the same generated parameters. The old path uses its own
original baseline digest, so timings include the new serialization-stable
fingerprint and validation costs. Generation is outside the timed region.
"""

import argparse
import ast
from dataclasses import replace
import gc
import json
from pathlib import Path
import statistics
import subprocess
import time
import tracemalloc

import attr

from tmol.database.scoring._content_hash import content_hash
from tmol.io import atom_array_from_cif
from tmol.ligand import prepare_ligands
from tmol.ligand._local_conjugate_params import (
    generate_conjugate_parameters,
    install_conjugate_parameters,
)
from tmol.ligand._registry import LigandPreparation, inject_ligand_preparations
from tmol.score.elec._params import ElecParamResolver
from tmol.tests.data import data_path
from tmol.tests.pack.test_conjugated_group_packing import FIXTURES


def old_installer(ref):
    source = subprocess.check_output(
        ["git", "show", f"{ref}:tmol/ligand/_local_conjugate_params.py"], text=True
    )
    names = {
        "_baseline_charges",
        "_local_identity",
        "_connection_key",
        "install_conjugate_parameters",
    }
    tree = ast.Module(
        body=[
            n
            for n in ast.parse(source).body
            if isinstance(n, ast.FunctionDef) and n.name in names
        ],
        type_ignores=[],
    )
    scope = {
        "attr": attr,
        "content_hash": content_hash,
        "ElecParamResolver": ElecParamResolver,
    }
    exec(compile(tree, f"{ref}:installer", "exec"), scope)
    return scope


def measure(call, repeats):
    for _ in range(3):
        call()
    times = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        result = call()
        times.append(1000 * (time.perf_counter() - start))
        del result
    gc.collect()
    tracemalloc.start()
    result = call()
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del result
    return {
        "median_ms": statistics.median(times),
        "samples_ms": times,
        "python_retained_bytes": retained,
        "python_peak_bytes": peak,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="165d1fe70")
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old = old_installer(args.baseline)
    cases = {}
    for fixture, filename in sorted(FIXTURES.items()):
        array = atom_array_from_cif(data_path("covalent_fixtures", filename + ".cif"))
        db, _ = prepare_ligands(array, seed=20250828)
        result = generate_conjugate_parameters(array, db)
        residues = {r.name: r for r in db.chemical.residues}
        charge_index = {
            (p.res, p.atom): p.charge for p in db.scoring.elec.atom_charge_parameters
        }
        old_rows = []
        for row in result.residues:
            rt = residues[row.residue_type.name]
            bonded = db.scoring.cartbonded.residue_params.get(
                rt.name, db.scoring.cartbonded.residue_params.get(rt.base_name)
            )
            digest = old["_local_identity"](
                rt, old["_baseline_charges"](charge_index, rt), bonded
            )
            old_rows.append(replace(row, baseline_sha256=digest))
        old_result = replace(result, residues=tuple(old_rows))
        preps = [
            LigandPreparation(
                residue_type=r.residue_type,
                partial_charges=r.partial_charges,
                cartbonded_params=r.cartbonded_params,
                baseline_sha256=r.baseline_sha256,
                connection_params=result.connections if i == 0 else (),
            )
            for i, r in enumerate(result.residues)
        ]
        calls = {
            "previous_private": lambda: old["install_conjugate_parameters"](
                db, old_result
            ),
            "current_private": lambda: install_conjugate_parameters(db, result),
            "current_ordinary": lambda: inject_ligand_preparations(db, preps),
        }
        expected = calls["previous_private"]()
        for call in calls.values():
            actual = call()
            assert actual.chemical == expected.chemical
            assert actual.scoring.elec == expected.scoring.elec
            assert actual.scoring.cartbonded == expected.scoring.cartbonded
        cases[fixture] = {
            name: measure(call, args.repeats) for name, call in calls.items()
        }
        installed = calls["current_ordinary"]()
        assert inject_ligand_preparations(installed, preps) is installed
        cases[fixture]["ordinary_repeat"] = measure(
            lambda: inject_ligand_preparations(installed, preps), args.repeats
        )
    args.output.write_text(
        json.dumps(
            {"baseline": args.baseline, "repeats": args.repeats, "cases": cases},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
