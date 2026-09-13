"""Compare inter-residue bond extraction against a pinned AtomWorks revision."""

import argparse
import ast
import json
import statistics
import subprocess
import time
import tracemalloc
from pathlib import Path

import biotite.structure as struc
import numpy as np
from atomworks.io.utils.leaving_atoms import _get_inter_residue_bonds

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--atomworks", type=Path, required=True)
parser.add_argument("--baseline", default="774056c7")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
source = subprocess.check_output(
    ["git", "show", args.baseline + ":src/atomworks/io/utils/leaving_atoms.py"],
    cwd=args.atomworks,
    text=True,
)
function = next(
    n
    for n in ast.parse(source).body
    if isinstance(n, ast.FunctionDef) and n.name == "_get_inter_residue_bonds"
)
namespace = {"np": np, "struc": struc}
exec(
    compile(ast.Module(body=[function], type_ignores=[]), "baseline", "exec"), namespace
)
rows = []
for count in (1000, 10000, 100000):
    atoms = struc.AtomArray(count)
    atoms.res_name[:] = "ALA"
    atoms.res_id[:] = np.arange(count) // 10
    atoms.chain_id[:] = "A"
    atoms.set_annotation("transformation_id", np.full(count, "assembly_1"))
    atoms.bonds = struc.BondList(
        count,
        np.column_stack(
            (np.arange(count - 1), np.arange(1, count), np.ones(count - 1, int))
        ),
    )
    np.testing.assert_array_equal(
        namespace["_get_inter_residue_bonds"](atoms), _get_inter_residue_bonds(atoms)
    )
    row = {"atoms": count}
    for label, function in (
        ("baseline", namespace["_get_inter_residue_bonds"]),
        ("candidate", _get_inter_residue_bonds),
    ):
        elapsed = []
        for _ in range(9):
            start = time.perf_counter()
            function(atoms)
            elapsed.append(time.perf_counter() - start)
        tracemalloc.start()
        function(atoms)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        row[label] = {
            "median_seconds": statistics.median(elapsed),
            "python_traced_peak_bytes": peak,
        }
    rows.append(row)
report = {
    "scope": "Synthetic contiguous residues; identical bond indices; nine warm repetitions. tracemalloc excludes native allocations.",
    "baseline": args.baseline,
    "rows": rows,
}
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(report)
