"""Benchmark shared RDKit conversion against a pinned AtomWorks converter.

Run with current AtomWorks on PYTHONPATH. Synthetic cases assert identical
SMILES and coordinates before reporting warm timing and Python-traced memory.
"""

import argparse
import importlib.util
import json
import statistics
import subprocess
import tempfile
import time
import tracemalloc
from pathlib import Path
import biotite.structure as struc
import numpy as np
from rdkit import Chem
from atomworks.io.tools import rdkit as current

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--atomworks", type=Path, required=True)
parser.add_argument("--baseline", default="59afb1e2")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = args.atomworks
scratch = tempfile.TemporaryDirectory()
path = Path(scratch.name) / "baseline_rdkit.py"
path.write_bytes(
    subprocess.check_output(
        ["git", "show", args.baseline + ":src/atomworks/io/tools/rdkit.py"], cwd=root
    )
)
spec = importlib.util.spec_from_file_location("baseline_rdkit_conversion", path)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
rows = []
for count, annotations in [(20, 0), (200, 0), (3000, 32)]:
    arr = struc.AtomArray(count)
    arr.element[:] = "C"
    arr.atom_name[:] = "C"
    arr.coord = np.column_stack(
        (np.arange(count) * 1.2, np.arange(count) % 2 * 0.8, np.zeros(count))
    )
    arr.bonds = struc.BondList(
        count,
        np.column_stack(
            (np.arange(count - 1), np.arange(1, count), np.ones(count - 1, dtype=int))
        ),
    )
    for i in range(annotations):
        arr.set_annotation(f"unused_{i}", np.full(count, "unused_" + "x" * 120))
    products = []
    timings = {}
    peaks = {}
    for name, module in [("dev_59afb1e2", old), ("shared_774056c7", current)]:
        fn = lambda: module.atom_array_to_rdkit(arr, annotations_to_keep=[])
        products.append(fn())
        elapsed = []
        for _ in range(9):
            start = time.perf_counter()
            fn()
            elapsed.append(time.perf_counter() - start)
        timings[name] = statistics.median(elapsed)
        tracemalloc.start()
        fn()
        peaks[name] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    assert Chem.MolToSmiles(products[0]) == Chem.MolToSmiles(products[1])
    np.testing.assert_array_equal(
        products[0].GetConformer().GetPositions(),
        products[1].GetConformer().GetPositions(),
    )
    rows.append(
        dict(
            atoms=count,
            unused_annotations=annotations,
            seconds=timings,
            python_traced_peak_bytes=peaks,
        )
    )
report = {
    "scope": "Isolated AtomArray-to-RDKit conversion; synthetic saturated carbon chains, finite coordinates, nine warm repetitions, identical chemistry and coordinates. tracemalloc excludes native allocations. Baseline converter loaded from dev59afb1e2 with the same current runtime dependencies.",
    "rows": rows,
}
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(report)

scratch.cleanup()
