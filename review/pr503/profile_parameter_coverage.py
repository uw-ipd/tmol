"""Compare host type setup and LJLK tables with the pinned parent revision."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time
import tracemalloc

import attr
import torch

from tmol.chemical import ResidueTypeSet
from tmol.database import ParameterDatabase
from tmol.pose import PackedBlockTypes
from tmol.score import AtomTypeDependentTerm
from tmol.score.ljlk import LJLKParamResolver

BASELINE = "9ceaeeb1f422292f58be65a5f8503bb49ce241f9"


def previous(relative, name, hashes):
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:{relative}"], text=True
    )
    hashes[relative] = hashlib.sha256(source.encode()).hexdigest()
    module = relative.removesuffix(".py").replace("/", ".")
    namespace = {"__name__": module, "__package__": module.rsplit(".", 1)[0]}
    exec(compile(source, f"baseline:{relative}", "exec"), namespace)
    return namespace[name]


def measure(functions, device, repeats=9, calls=5):
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    times = {label: [] for label in functions}
    for repeat in range(repeats + 1):
        for label in (list(functions) if repeat % 2 else list(functions)[::-1]):
            sync()
            start = time.perf_counter()
            for _ in range(calls):
                result = functions[label]()
            sync()
            if repeat:
                times[label].append((time.perf_counter() - start) / calls)
            del result
    allocations = {}
    for label, function in functions.items():
        tracemalloc.start()
        result = function()
        retained, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        allocations[label] = dict(
            python_retained_bytes=retained, python_peak_bytes=peak
        )
        del result
    return dict(
        seconds=times,
        medians={k: statistics.median(v) for k, v in times.items()},
        allocations=allocations,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    hashes = {}
    old_term = previous(
        "tmol/score/_atom_type_dependent_term.py", "AtomTypeDependentTerm", hashes
    )
    old_resolver = previous("tmol/score/ljlk/_params.py", "LJLKParamResolver", hashes)
    database = ParameterDatabase.get_default()
    restypes = ResidueTypeSet.from_database(database.chemical)
    terms = {
        "before": old_term(database, device),
        "after": AtomTypeDependentTerm(database, device),
    }
    output = dict(
        baseline=BASELINE,
        source_sha256=hashes,
        device=str(device),
        torch=torch.__version__,
        packed={},
    )
    for label, blocks in [
        ("ALA", [next(bt for bt in restypes.residue_types if bt.name == "ALA")]),
        ("default_catalog", restypes.residue_types),
    ]:
        template = PackedBlockTypes.from_restype_list(
            database.chemical, restypes, blocks, device
        )

        def prepare(term):
            packed = copy.copy(template)
            term.setup_packed_block_types(packed)
            return packed

        functions = {k: lambda term=term: prepare(term) for k, term in terms.items()}
        before, after = (f() for f in functions.values())
        for field in (
            "atom_types",
            "n_heavy_atoms",
            "heavy_atom_inds",
            "atom_unique_ids",
            "atom_wildcard_ids",
            "atom_cross_ids",
        ):
            torch.testing.assert_close(
                getattr(before, field), getattr(after, field), rtol=0, atol=0
            )
        assert before.atom_unique_id_index == after.atom_unique_id_index
        output["packed"][label] = dict(
            n_types=len(blocks),
            real_atoms=sum(len(bt.atoms) for bt in blocks),
            exact_annotations=True,
            **measure(functions, device),
        )
    functions = {
        k: lambda cls=cls: cls.from_database(
            database.chemical, database.scoring.ljlk, device
        )
        for k, cls in [("before", old_resolver), ("after", LJLKParamResolver)]
    }
    before, after = (f() for f in functions.values())
    fields_checked = []
    for group in ("global_params", "type_params"):
        for field in attr.fields(type(getattr(before, group))):
            torch.testing.assert_close(
                getattr(getattr(before, group), field.name),
                getattr(getattr(after, group), field.name),
                rtol=0,
                atol=0,
                equal_nan=True,
            )
            fields_checked.append(f"{group}.{field.name}")
    assert before.atom_type_index.equals(after.atom_type_index)
    output["resolver"] = dict(
        exact_tensor_fields=fields_checked, **measure(functions, device)
    )
    output["scope"] = (
        "Nine alternating warm rounds of five calls; packed setup reuses warm block annotations and fresh shallow packed copies. Resolver includes chemical lookup and device transfers. Python allocation tracing excludes native/RSS/GPU allocations. No whole-workflow speedup claim."
    )
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
