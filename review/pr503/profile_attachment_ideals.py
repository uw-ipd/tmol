"""Compare attachment generation against the former whole-conformer targets.

Both implementations share the current preparation/conformer code, unchanged
by the target-source fix. Only attachment record generation is timed. Equilibrium
values intentionally change; this does not measure end-to-end speed or validate
an independently fitted force field.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time
import types
from unittest.mock import patch

import attr

from tmol.io import atom_array_from_cif
from tmol.ligand import _connection_params, prepare_ligands
from tmol.tests.data import data_path
from tmol.tests.pack.test_conjugated_group_packing import FIXTURES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="f849c6900")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = {}
    modules = {}
    for name in ("_conjugate_geometry", "_connection_params"):
        filename = f"tmol/ligand/{name}.py"
        sources[filename] = subprocess.check_output(
            ["git", "show", f"{args.baseline}:{filename}"], text=True
        )
        module = types.ModuleType(f"tmol.ligand.{name}")
        with patch.dict("sys.modules", modules):
            exec(
                compile(sources[filename], f"{args.baseline}:{filename}", "exec"),
                module.__dict__,
            )
        modules[module.__name__] = module
    old = modules["tmol.ligand._connection_params"]
    cases = {}
    for name, stem in FIXTURES.items():
        array = atom_array_from_cif(data_path("covalent_fixtures", stem + ".cif"))
        database, _ = prepare_ligands(array, seed=20250828)
        cases[name] = {}
        for label, module in (
            ("whole_conformer", old),
            ("generator_ideals", _connection_params),
        ):
            samples, records = [], []
            for _ in range(3):
                start = time.perf_counter()
                records.append(
                    module.generate_conjugate_connection_params(
                        array, database, seed=20250828
                    )
                )
                samples.append(time.perf_counter() - start)
            assert records[0] == records[1] == records[2]
            cases[name][label] = {
                "seconds": samples,
                "median_seconds": statistics.median(samples),
                "records": [attr.asdict(r) for r in records[0]],
            }
        print(
            name, {k: v["median_seconds"] for k, v in cases[name].items()}, flush=True
        )
    args.output.write_text(
        json.dumps(
            {
                "baseline": args.baseline,
                "baseline_source_sha256": {
                    k: hashlib.sha256(v.encode()).hexdigest()
                    for k, v in sources.items()
                },
                "current_source_sha256": hashlib.sha256(
                    Path(_connection_params.__file__).read_bytes()
                ).hexdigest(),
                "cases": cases,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
