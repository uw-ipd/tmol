"""Compare lookup metadata construction, excluding native table preparation."""

import argparse
import gc
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import subprocess
import time
import tracemalloc

import attr
import pandas

from tmol.database import ParameterDatabase
from tmol.score.dunbrack import DunbrackParamResolver

BASELINE = "e5b67c122"
SOURCE = "tmol/score/dunbrack/_params.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = subprocess.check_output(["git", "show", f"{BASELINE}:{SOURCE}"], text=True)
    namespace = dict(vars(importlib.import_module("tmol.score.dunbrack._params")))
    exec(compile(source, "baseline_dun_lookup", "exec"), namespace)
    previous = namespace["DunbrackParamResolver"]
    default = ParameterDatabase.get_default().scoring.dun
    results = []
    for extra_rows in (0, 10000):
        database = attr.evolve(
            default,
            dun_lookup=(
                *default.dun_lookup,
                *(
                    attr.evolve(
                        default.dun_lookup[i % len(default.dun_lookup)],
                        residue_name=f"benchmark_alias_{i}",
                    )
                    for i in range(extra_rows)
                ),
            ),
        )
        names = [
            x.table_name
            for x in (*database.rotameric_libraries, *database.semi_rotameric_libraries)
        ]

        def before():
            return (
                previous._create_all_table_indices(names, database.dun_lookup),
                previous._create_rotameric_indices(database),
                previous._create_semirotameric_indices(database),
            )

        def after():
            return DunbrackParamResolver._create_table_indices(
                names, database.dun_lookup, len(database.rotameric_libraries)
            )

        for a, b in zip(before(), after()):
            pandas.testing.assert_frame_equal(a, b)
        functions = {"before": before, "after": after}
        times = {side: [] for side in functions}
        calls = 100 if extra_rows == 0 else 10
        for trial in range(7):
            for side in (list(functions) if trial % 2 == 0 else list(functions)[::-1]):
                start = time.perf_counter()
                for _ in range(calls):
                    functions[side]()
                times[side].append((time.perf_counter() - start) * 1000 / calls)
        memory = {}
        for side, function in functions.items():
            gc.collect()
            tracemalloc.start()
            outputs = function()
            retained, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory[side] = {"retained_bytes": retained, "peak_bytes": peak}
            del outputs
        medians = {side: statistics.median(v) for side, v in times.items()}
        results.append(
            {
                "lookup_rows": len(database.dun_lookup),
                "additional_synthetic_aliases": extra_rows,
                "frames_exact": True,
                "calls_per_round": calls,
                "milliseconds": times,
                "median_milliseconds": medians,
                "speedup": medians["before"] / medians["after"],
                "traced_python_memory": memory,
            }
        )
    report = {
        "baseline_commit": subprocess.check_output(
            ["git", "rev-parse", BASELINE], text=True
        ).strip(),
        "source_sha256": {
            "before": hashlib.sha256(source.encode()).hexdigest(),
            "after": hashlib.sha256(Path(SOURCE).read_bytes()).hexdigest(),
        },
        "pandas": pandas.__version__,
        "results": results,
        "limits": "Lookup metadata only; excludes imports, fixture construction, spline fitting, device transfers and scoring. Seven alternating warm rounds. Synthetic aliases stress metadata scaling without adding statistical libraries. Tracemalloc covers traced Python allocations, not total process/native/CUDA memory.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
