"""Compare warm params export and retained Python allocations on shared inputs."""

import argparse
import gc
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
import tracemalloc
import types

from tmol.io import atom_array_from_cif
from tmol.ligand import prepare_ligands, load_params_file
from tmol.ligand import _params_io
from tmol.tests.data import data_path
from tmol.tests.pack.test_conjugated_group_packing import FIXTURES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", default="e2107de71f5f5335cce6f39507856923f64ea77d"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old = types.ModuleType("params_writer_baseline")
    exec(
        compile(
            subprocess.check_output(
                ["git", "show", f"{args.baseline}:tmol/ligand/_params_io.py"], text=True
            ),
            "baseline_params_io.py",
            "exec",
        ),
        old.__dict__,
    )
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "parameters.tmol"
        for name, stem in FIXTURES.items():
            prepare_ligands(
                atom_array_from_cif(data_path("covalent_fixtures", stem + ".cif")),
                seed=20250828,
                params_output=str(path),
            )
            preps = load_params_file(path)
            modules = {"baseline": old, "candidate": _params_io}
            timings = {key: [] for key in modules}
            for module in modules.values():
                module.write_params_file(preps, path, format="tmol")
            for pair in range(7):
                for key in (list(modules) if pair % 2 == 0 else list(modules)[::-1]):
                    start = time.perf_counter()
                    modules[key].write_params_file(preps, path, format="tmol")
                    timings[key].append(time.perf_counter() - start)
            memory = {}
            for key, module in modules.items():
                gc.collect()
                count_before = len(module._CompactDumper.yaml_representers)
                tracemalloc.start()
                start = tracemalloc.get_traced_memory()[0]
                for _ in range(20):
                    module.write_params_file(preps, path, format="tmol")
                gc.collect()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                memory[key] = {
                    "writes": 20,
                    "retained_python_bytes": current - start,
                    "peak_python_bytes": peak - start,
                    "added_yaml_representers": len(
                        module._CompactDumper.yaml_representers
                    )
                    - count_before,
                }
            rows.append(
                {
                    "fixture": name,
                    "seconds": timings,
                    "memory": memory,
                    "median_speedup": statistics.median(timings["baseline"])
                    / statistics.median(timings["candidate"]),
                }
            )
    args.output.write_text(
        json.dumps(
            {
                "baseline": args.baseline,
                "rows": rows,
                "scope": "Same prepared records; warm YAML export only. Tracemalloc excludes native allocations and RSS.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
