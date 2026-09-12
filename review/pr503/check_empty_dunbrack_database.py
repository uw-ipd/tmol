"""Verify default resolver equality and storage for absent library families."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess

import attr
import pandas
import torch

from tmol.database import ParameterDatabase
from tmol.score.dunbrack import DunbrackParamResolver

BASELINE = "3709b04f0"
SOURCE = "tmol/score/dunbrack/_params.py"


def storage_bytes(resolver):
    storages = {}
    for name in ("scoring_db", "scoring_db_aux", "sampling_db"):
        view = getattr(resolver, name)
        for field in attr.fields(type(view)):
            tensor = getattr(view, field.name)
            storage = tensor.untyped_storage()
            storages[str(tensor.device), storage.data_ptr()] = storage.nbytes()
    return sum(storages.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    module = importlib.import_module("tmol.score.dunbrack._params")
    source = subprocess.check_output(
        ["git", "show", f"{args.baseline}:{SOURCE}"], text=True
    )
    namespace = dict(vars(module))
    exec(compile(source, "previous_dunbrack_resolver", "exec"), namespace)
    database = ParameterDatabase.get_default().scoring.dun
    before = namespace["DunbrackParamResolver"].from_database(database, device)
    after = DunbrackParamResolver.from_database(database, device)
    checked = 0
    for name in ("scoring_db", "scoring_db_aux", "sampling_db"):
        left, right = getattr(before, name), getattr(after, name)
        for field in attr.fields(type(left)):
            torch.testing.assert_close(
                getattr(left, field.name), getattr(right, field.name), rtol=0, atol=0
            )
            checked += 1
    for name in (
        "all_table_indices",
        "rotameric_table_indices",
        "semirotameric_table_indices",
    ):
        pandas.testing.assert_frame_equal(getattr(before, name), getattr(after, name))
    # Aligned default grids need no separate source-frame storage.
    assert (
        after.sampling_db.rotameric_bb_source_start
        is after.sampling_db.rotameric_bb_start
    )
    default_sizes = {"before": storage_bytes(before), "after": storage_bytes(after)}
    assert default_sizes["after"] == default_sizes["before"]
    sizes = {"complete_default": storage_bytes(after)}
    for label, residue in (
        ("rotameric_LEU_only", "LEU"),
        ("semirotameric_PHE_only", "PHE"),
        ("no_libraries", None),
    ):
        names = {
            row.dun_table_name
            for row in database.dun_lookup
            if row.residue_name == residue
        }
        subset = attr.evolve(
            database,
            dun_lookup=tuple(
                row for row in database.dun_lookup if row.dun_table_name in names
            ),
            rotameric_libraries=tuple(
                lib for lib in database.rotameric_libraries if lib.table_name in names
            ),
            semi_rotameric_libraries=tuple(
                lib
                for lib in database.semi_rotameric_libraries
                if lib.table_name in names
            ),
        )
        resolver = DunbrackParamResolver.from_database(subset, device)
        sizes[label] = storage_bytes(resolver)
    result = {
        "baseline_commit": subprocess.check_output(
            ["git", "rev-parse", args.baseline], text=True
        ).strip(),
        "baseline_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "candidate_source_sha256": hashlib.sha256(
            Path(SOURCE).read_bytes()
        ).hexdigest(),
        "device": str(device),
        "torch": torch.__version__,
        "default_tensor_fields_exact": checked,
        "default_lookup_dataframes_exact": 3,
        "default_source_grid_reuses_target_storage": True,
        "default_storage_bytes": default_sizes,
        "unique_derived_tensor_storage_bytes": sizes,
        "limits": "Derived resolver tensor storage only, deduplicated across views. Excludes source library tensors, parameter database, Python metadata, allocator overhead and temporary peaks. Smaller databases deliberately omit unrelated scoring/sampling references; no automatic pruning or timing improvement is claimed.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
