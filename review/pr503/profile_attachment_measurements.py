"""Measured cross-residue bond collection on identical repeated inputs."""

import argparse
import ast
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time
import tracemalloc

import biotite.structure as struc
import numpy as np

from tmol.io import atom_array_from_cif
from tmol.ligand._preparation import _bond_lengths_by_site
from tmol.tests.data import data_path
from tmol.tests.pack.test_conjugated_group_packing import FIXTURES

BASELINE = "daef9b803"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--copies", nargs="+", type=int, default=[1, 8, 32])
    args = parser.parse_args()
    source = subprocess.check_output(
        ["git", "show", BASELINE + ":tmol/ligand/_preparation.py"], text=True
    )
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "_bond_lengths_by_site"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "baseline.py", "exec"),
        namespace,
    )
    baseline = namespace["_bond_lengths_by_site"]
    rows = []
    for fixture, stem in FIXTURES.items():
        original = atom_array_from_cif(data_path("covalent_fixtures", stem + ".cif"))
        for copies in args.copies:
            arrays = []
            for index in range(copies):
                array = original.copy()
                array.res_id += 10000 * index
                arrays.append(array)
            array = struc.concatenate(arrays)
            del arrays
            expected = baseline(array)

            def check(result):
                assert result.keys() == expected.keys()
                np.testing.assert_allclose(
                    [result[key] for key in expected],
                    list(expected.values()),
                    rtol=0,
                    atol=2e-7,
                )

            functions = {"baseline": baseline, "candidate": _bond_lengths_by_site}
            seconds = {name: [] for name in functions}
            for fn in functions.values():
                check(fn(array))
            for repeat in range(7):
                names = list(functions)
                if repeat % 2:
                    names.reverse()
                for name in names:
                    start = time.perf_counter()
                    result = functions[name](array)
                    seconds[name].append(time.perf_counter() - start)
                    check(result)
            peaks = {}
            for name, fn in functions.items():
                gc.collect()
                tracemalloc.start()
                result = fn(array)
                _, peaks[name] = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                check(result)
            medians = {
                name: statistics.median(values) for name, values in seconds.items()
            }
            row = dict(
                fixture=fixture,
                copies=copies,
                atoms=len(array),
                seconds=seconds,
                median_seconds=medians,
                speedup=medians["baseline"] / medians["candidate"],
                python_peak_bytes=peaks,
                maximum_length_difference=max(
                    abs(result[key] - expected[key]) for key in expected
                ),
            )
            rows.append(row)
            print(fixture, copies, medians, peaks, flush=True)
    args.output.write_text(
        json.dumps(
            dict(
                baseline=BASELINE,
                source_sha256=hashlib.sha256(
                    Path("tmol/ligand/_preparation.py").read_bytes()
                ).hexdigest(),
                scope="Cross-residue measurement collection only; seven alternating warm pairs with identical keys and lengths within 2e-7 angstrom. Batched float32 norms can differ by one ULP. Not full preparation timing. Traced Python/NumPy peaks exclude native process allocations.",
                rows=rows,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
