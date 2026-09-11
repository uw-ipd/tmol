"""Compare bond-preserving CIF insertion against an unmodified PR checkout.

Run with a Python environment containing numpy and biotite:
    python review/pr503/benchmark_cif_completion.py /path/to/pr-checkout
"""

import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

import biotite.structure as struc
import numpy as np


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(baseline):
    root = Path(__file__).resolve().parents[2]
    old = load(Path(baseline) / "tmol/io/_cif.py", "baseline_cif")
    new = load(root / "tmol/io/_cif.py", "candidate_cif")
    template = new._component_array(
        "ZZZ",
        ["C1", "C2", "C3"],
        ["C"] * 3,
        [("C1", "C2", 1), ("C2", "C3", 1)],
    )
    results = []
    for count in (100, 500, 2000):
        pieces = []
        for i in range(count):
            piece = template[:2].copy()
            piece.res_id[:] = i + 1
            piece.coord[:] = [[3 * i, 0, 0], [3 * i + 1, 0, 0]]
            pieces.append(piece)
        atoms = struc.concatenate(pieces)
        starts = struc.get_residue_starts(atoms)
        additions = {int(i): (["C3"], template) for i in starts}
        outputs, timings = [], {}
        for name, module in (("baseline", old), ("candidate", new)):
            output = module._inserted(atoms, starts, additions)
            outputs.append(output)
            elapsed = []
            for _ in range(5):
                begin = time.perf_counter()
                module._inserted(atoms, starts, additions)
                elapsed.append(time.perf_counter() - begin)
            timings[name] = statistics.median(elapsed)
        np.testing.assert_array_equal(outputs[0].coord, outputs[1].coord)
        for field in outputs[0].get_annotation_categories():
            np.testing.assert_array_equal(
                outputs[0].get_annotation(field), outputs[1].get_annotation(field)
            )
        assert {tuple(row) for row in outputs[0].bonds.as_array()} == {
            tuple(row) for row in outputs[1].bonds.as_array()
        }
        results.append(
            dict(
                residues=count,
                seconds=timings,
                speedup=timings["baseline"] / timings["candidate"],
            )
        )
    return results


if __name__ == "__main__":
    print(json.dumps(run(sys.argv[1]), indent=2))
