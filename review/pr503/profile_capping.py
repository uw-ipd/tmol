"""Compare cap construction on identical source residues and profiles."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time
import types

import numpy as np

from tmol.ligand._polymer_profile import cap_residue, profile_for_atom_array
from tmol.tests.ligand.test_nonstandard_backbones import _residue, _connection_atoms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--baseline", default="b723e88c746c728183c66ba9d00091d0bad658cb"
    )
    args = parser.parse_args()
    old = types.ModuleType("tmol.ligand.cap_review_baseline")
    old.__package__ = "tmol.ligand"
    exec(
        compile(
            subprocess.check_output(
                ["git", "show", f"{args.baseline}:tmol/ligand/_polymer_profile.py"],
                text=True,
            ),
            "baseline_polymer_profile.py",
            "exec",
        ),
        old.__dict__,
    )
    results = []
    for code in ("HYP", "MLE", "B3K", "FGA"):
        source = _residue(code).copy()
        source.set_annotation(
            "source_chemistry", np.array([f"tag_{i}" for i in range(len(source))])
        )
        profile = profile_for_atom_array(source, _connection_atoms(code))
        baseline, old_caps = old.cap_residue(source, profile)
        candidate, new_caps = cap_residue(source, profile)
        assert old_caps == new_caps
        np.testing.assert_array_equal(baseline.coord, candidate.coord)
        np.testing.assert_array_equal(
            baseline.bonds.as_array(), candidate.bonds.as_array()
        )
        for name in baseline.get_annotation_categories():
            np.testing.assert_array_equal(
                baseline.get_annotation(name), candidate.get_annotation(name)
            )
        functions = {
            "baseline": lambda: old.cap_residue(source, profile),
            "candidate_coordinates": lambda: cap_residue(source, profile),
            "candidate_topology": lambda: cap_residue(
                source, profile, include_coordinates=False
            ),
        }
        times = {name: [] for name in functions}
        for fn in functions.values():
            fn()
        for pair in range(7):
            for name in (list(functions) if pair % 2 == 0 else list(functions)[::-1]):
                start = time.perf_counter()
                for _ in range(100):
                    functions[name]()
                times[name].append((time.perf_counter() - start) / 100)
        results.append(
            {
                "residue": code,
                "seconds": times,
                "baseline_missing_annotations": sorted(
                    set(source.get_annotation_categories())
                    - set(baseline.get_annotation_categories())
                ),
                "candidate_missing_annotations": sorted(
                    set(source.get_annotation_categories())
                    - set(candidate.get_annotation_categories())
                ),
                "coordinate_speedup": statistics.median(times["baseline"])
                / statistics.median(times["candidate_coordinates"]),
                "topology_speedup": statistics.median(times["baseline"])
                / statistics.median(times["candidate_topology"]),
            }
        )
    args.output.write_text(
        json.dumps(
            {
                "baseline": args.baseline,
                "rows": results,
                "scope": "Warm cap construction only, seven alternating-order sets of 100 calls; coordinate/bond parity on common annotations checked first. Not full preparation timing.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
