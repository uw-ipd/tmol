"""Paired generator stage timings and Python allocation peaks.

The reference disables within-call reuse and materializes all capped models.
It uses identical chemistry and record generation; this measures an internal
optimization of attachment record generation, not an end-to-end speedup
over upstream preparation. Historical MMFF measurements do not describe
the current generated-geometry default.
"""

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time
import tracemalloc
from unittest.mock import patch

import biotite.structure as struc

from tmol.io import atom_array_from_cif
from tmol.ligand import prepare_ligands
from tmol.ligand import _connection_params as generator
from tmol.tests.data import data_path
from tmol.tests.pack.test_conjugated_group_packing import FIXTURES


@contextmanager
def implementation(reference):
    if reference:
        iterate = generator.iter_capped_conjugate_models
        with (
            patch.object(generator, "_model_identity", lambda _: object()),
            patch.object(
                generator,
                "iter_capped_conjugate_models",
                lambda *args: tuple(iterate(*args)),
            ),
        ):
            yield
    else:
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copies", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    rows = []
    for fixture, stem in FIXTURES.items():
        original = atom_array_from_cif(data_path("covalent_fixtures", stem + ".cif"))
        database, _ = prepare_ligands(original, seed=20250828)
        for count in args.copies:
            copies = []
            for i in range(count):
                copy = original.copy()
                copy.res_id += 10000 * i
                copies.append(copy)
            array = struc.concatenate(copies)
            expected = generator.generate_conjugate_connection_params(array, database)
            times, peaks = {}, {}
            for reference in (True, False):
                name = "reference" if reference else "candidate"
                times[name] = []
                with implementation(reference):
                    assert (
                        generator.generate_conjugate_connection_params(array, database)
                        == expected
                    )
            for repeat in range(args.repeats):
                for reference in ((True, False) if repeat % 2 == 0 else (False, True)):
                    name = "reference" if reference else "candidate"
                    with implementation(reference):
                        start = time.perf_counter()
                        result = generator.generate_conjugate_connection_params(
                            array, database
                        )
                        times[name].append(time.perf_counter() - start)
                        assert result == expected
            for reference in (True, False):
                name = "reference" if reference else "candidate"
                with implementation(reference):
                    gc.collect()
                    tracemalloc.start()
                    result = generator.generate_conjugate_connection_params(
                        array, database
                    )
                    _, peaks[name] = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    assert result == expected
            row = dict(
                fixture=fixture,
                copies=count,
                atoms=len(array),
                records=len(expected),
                seconds=times,
                python_peak_bytes=peaks,
                speedup=statistics.median(times["reference"])
                / statistics.median(times["candidate"]),
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    sources = [
        Path(generator.__file__),
        Path(__file__).resolve(),
        Path(generator.iter_capped_conjugate_models.__code__.co_filename).resolve(),
    ]
    args.output.write_text(
        json.dumps(
            dict(
                rows=rows,
                source_sha256={
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
                },
                notes=[
                    "Reference disables reuse and materializes models; chemistry and records are identical.",
                    "Not an upstream or end-to-end preparation speed comparison.",
                    "Warm CPU wall time; Python allocation peaks measured separately, not native RSS.",
                ],
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
