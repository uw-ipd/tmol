"""Compare selective mirrored-library generation with the preceding function."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import attr
import torch

from tmol.database import ParameterDatabase, l_base_name
from tmol.database.scoring import _mirrored_dunbrack as mirroring

BASELINE = "b538a2e60"
SOURCE = "tmol/database/scoring/_mirrored_dunbrack.py"


def tensor_storages(value, output=None):
    output = {} if output is None else output
    if isinstance(value, torch.Tensor):
        storage = value.untyped_storage()
        output[str(value.device), storage.data_ptr()] = storage.nbytes()
    elif attr.has(type(value)):
        for field in attr.fields(type(value)):
            tensor_storages(getattr(value, field.name), output)
    elif isinstance(value, (tuple, list)):
        for item in value:
            tensor_storages(item, output)
    return output


def compare_values(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif attr.has(type(left)):
        for field in attr.fields(type(left)):
            compare_values(getattr(left, field.name), getattr(right, field.name))
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            compare_values(a, b)
    else:
        assert left == right


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = subprocess.check_output(["git", "show", f"{BASELINE}:{SOURCE}"], text=True)
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "with_mirrored_libraries"
    )
    namespace = dict(vars(mirroring))
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]), "previous_generation", "exec"
        ),
        namespace,
    )
    functions = {
        "before": namespace["with_mirrored_libraries"],
        "after": mirroring.with_mirrored_libraries,
    }
    database = ParameterDatabase.get_default()
    all_libraries = database.scoring.dun
    rot = tuple(
        lib
        for lib in all_libraries.rotameric_libraries
        if not lib.rotameric_data.backbone_is_mirrored
    )
    semi = tuple(
        lib
        for lib in all_libraries.semi_rotameric_libraries
        if not lib.rotameric_data.backbone_is_mirrored
    )
    names = {lib.table_name for lib in (*rot, *semi)}
    original = attr.evolve(
        all_libraries,
        rotameric_libraries=rot,
        semi_rotameric_libraries=semi,
        dun_lookup=tuple(
            row for row in all_libraries.dun_lookup if row.dun_table_name in names
        ),
    )
    full_request = {
        l_base_name(residue): residue.name
        for residue in database.chemical.residues
        if residue.name == residue.base_name
        and residue.properties.polymer.sidechain_chirality == "d"
    }
    compare_values(
        functions["before"](original, full_request),
        functions["after"](original, full_request),
    )
    source_storages = tensor_storages(original)
    cases = {}
    for label, requested in (
        ("all_default_D_types", full_request),
        ("DARG_only", {"ARG": "DARG"}),
        ("DSER_only", {"SER": "DSER"}),
        ("no_D_types", {}),
    ):
        sizes, timings = {}, {name: [] for name in functions}
        # Record owned tensor storage separately from timed calls. Existing
        # input storage is shared and must not be counted as an allocation.
        for name, function in functions.items():
            result = function(original, requested)
            storage = tensor_storages(result)
            sizes[name] = sum(
                size for key, size in storage.items() if key not in source_storages
            )
            if name == "after":
                old = functions["before"](original, requested)
                old_tables = {
                    lib.table_name: lib
                    for lib in (*old.rotameric_libraries, *old.semi_rotameric_libraries)
                }
                for lib in (
                    *result.rotameric_libraries,
                    *result.semi_rotameric_libraries,
                ):
                    compare_values(lib, old_tables[lib.table_name])
                del old, old_tables
            del result
        for round_index in range(7):
            for name in list(functions)[:: -1 if round_index % 2 else 1]:
                samples = []
                for _ in range(3):
                    start = time.perf_counter()
                    result = functions[name](original, requested)
                    samples.append(time.perf_counter() - start)
                    del result
                timings[name].append(statistics.median(samples))
        medians = {name: statistics.median(values) for name, values in timings.items()}
        cases[label] = {
            "new_tensor_storage_bytes": sizes,
            "round_median_seconds": timings,
            "median_seconds": medians,
            "before_over_after": medians["before"] / medians["after"],
        }
    output = {
        "baseline_commit": subprocess.check_output(
            ["git", "rev-parse", BASELINE], text=True
        ).strip(),
        "baseline_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "candidate_source_sha256": hashlib.sha256(
            Path(SOURCE).read_bytes()
        ).hexdigest(),
        "torch": torch.__version__,
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "all_default_values_exact": True,
        "requested_generated_table_values_exact": True,
        "cases": cases,
        "limits": "Library generation only, not spline-resolver construction or packing/scoring. Seven alternating warm rounds of three uninstrumented samples. Input/default database construction excluded. New tensor-storage bytes exclude shared input storage, Python metadata, temporary peak allocations and allocator/process overhead. The no-D case is an early return, not a representative whole-application speedup.",
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
