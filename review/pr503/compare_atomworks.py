"""Compare parser workloads in separate processes (AtomWorks patches Biotite).

Run once per backend in the same environment. These parsers have different
semantics: timings are accompanied by output inventories, not a speedup claim.
"""

import argparse
import cProfile
import hashlib
import importlib.metadata
import subprocess
import json
from pathlib import Path
import statistics
import time
import traceback

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("tmol", "atomworks"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    data = root / "tmol/tests/data"
    paths = [data / "cif/1UBQ.cif"]
    paths += sorted((data / "ncaa_fixtures").glob("*.cif"))
    paths += sorted((data / "covalent_fixtures").glob("*.cif"))
    paths += [data / "cif/cyclic_peptide_1jbl.cif"]
    if args.backend == "tmol":
        from tmol.io import atom_array_from_cif

        read = atom_array_from_cif
    else:
        from atomworks.io.parser import parse
        from atomworks.io.config import ParseConfig

        config = ParseConfig(
            model=1,
            build_assembly=None,
            remove_ccds=(),
            remove_waters=False,
            fix_arginines=False,
            fix_ligands_at_symmetry_centers=False,
            long_bond_policy="keep",
            struct_conn_distance_policy="keep",
            add_bond_types_from_struct_conn=("covale", "disulf"),
            hydrogen_policy="keep",
            ccd_mirror_path=None,
            add_id_and_entity_annotations=False,
        )

        def read(path):
            array = parse(path, config=config)["asym_unit"]
            return array[0] if array.coord.ndim == 3 else array

    metadata = {
        "backend": args.backend,
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("biotite", "numpy", "rdkit", "atomworks")
        },
        "repeats": args.repeats,
        "timing_instrumented": False,
    }
    if args.backend == "atomworks":
        import atomworks

        package_root = Path(atomworks.__file__).resolve().parents[2]
        metadata["source"] = str(package_root)
        metadata["commit"] = subprocess.check_output(
            ["git", "-C", str(package_root), "rev-parse", "HEAD"], text=True
        ).strip()
        metadata["diff_sha256"] = hashlib.sha256(
            subprocess.check_output(["git", "-C", str(package_root), "diff", "HEAD"])
        ).hexdigest()
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path in paths:
        row = {"fixture": str(path.relative_to(data))}
        prof = cProfile.Profile()
        try:
            durations = []
            for i in range(args.repeats + 1):
                start = time.perf_counter()
                array = read(path)
                durations.append(time.perf_counter() - start)
            if args.profile:
                prof.runcall(read, path)
                prof.dump_stats(
                    str(args.output.parent / f"{args.backend}-{path.stem}.pstats")
                )
            resolved_heavy = (~np.isnan(array.coord).any(axis=1)) & (
                ~np.isin(array.element, ["H", "D"])
            )
            # Match inventories independently of author/label numbering and order.
            observed = sorted(
                (str(r), str(n), str(e), *np.round(x.astype(float), 3))
                for r, n, e, x in zip(
                    array.res_name[resolved_heavy],
                    array.atom_name[resolved_heavy],
                    array.element[resolved_heavy],
                    array.coord[resolved_heavy],
                )
            )
            keys = [
                tuple(str(x) for x in record)
                for record in zip(
                    array.chain_id,
                    array.res_id,
                    array.ins_code,
                    array.res_name,
                    array.atom_name,
                )
            ]
            atom_inventory = sorted(
                (k, str(e), tuple(str(x) for x in np.round(c, 3)))
                for k, e, c in zip(keys, array.element, array.coord)
            )
            bond_inventory = sorted(
                (tuple(sorted((keys[i], keys[j]))), int(o))
                for i, j, o in array.bonds.as_array()
            )
            row["atom_inventory"] = hashlib.sha256(
                repr(atom_inventory).encode()
            ).hexdigest()
            row["bond_inventory"] = hashlib.sha256(
                repr(bond_inventory).encode()
            ).hexdigest()
            row.update(
                status="passed",
                cold_s=durations[0],
                warm_s=durations[1:],
                median_s=statistics.median(durations[1:]),
                atoms=len(array),
                unresolved=int(np.isnan(array.coord).any(axis=1).sum()),
                resolved_heavy=len(observed),
                resolved_inventory=hashlib.sha256(repr(observed).encode()).hexdigest(),
                bonds=array.bonds.get_bond_count() if array.bonds is not None else None,
                residues={
                    str(k): int(v)
                    for k, v in zip(*np.unique(array.res_name, return_counts=True))
                },
            )
        except Exception:
            row.update(status="failed", traceback=traceback.format_exc())
        rows.append(row)
        args.output.write_text(json.dumps({**metadata, "results": rows}, indent=2))
        print(path.stem, row["status"], row.get("median_s"), flush=True)


if __name__ == "__main__":
    main()
