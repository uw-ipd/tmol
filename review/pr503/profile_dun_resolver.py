"""Check exact resolver tables, released-owner lifetime and warm cache lookup."""

import argparse
import gc
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import subprocess
import time
import weakref

import attr
import pandas
import torch

from tmol.database import ParameterDatabase
from tmol.score.dunbrack import DunbrackParamResolver

BASELINE = "be24bc72da1e5a62c7d0b3571feafb86f1472b1c"
SOURCE = "tmol/score/dunbrack/_params.py"


def tensor_storage_bytes(resolver):
    storages = {}
    for name in ("scoring_db", "scoring_db_aux", "sampling_db"):
        group = getattr(resolver, name)
        for field in attr.fields(type(group)):
            value = getattr(group, field.name)
            storage = value.untyped_storage()
            storages[str(value.device), storage.data_ptr()] = storage.nbytes()
    return sum(storages.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        device = torch.device("cuda", torch.cuda.current_device())
    source_sha = hashlib.sha256(Path(SOURCE).read_bytes()).hexdigest()
    old_source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:{SOURCE}"], text=True
    )
    module = importlib.import_module("tmol.score.dunbrack._params")
    namespace = dict(vars(module))
    exec(compile(old_source, "baseline_dun_resolver", "exec"), namespace)
    old_cls = namespace["DunbrackParamResolver"]
    classes = {"baseline": old_cls, "candidate": DunbrackParamResolver}
    database = ParameterDatabase.get_default().scoring.dun
    old = old_cls.from_database(database, device)
    new = DunbrackParamResolver.from_database(database, device)
    tensor_fields = 0
    for name in ("scoring_db", "scoring_db_aux", "sampling_db"):
        for field in attr.fields(type(getattr(new, name))):
            torch.testing.assert_close(
                getattr(getattr(old, name), field.name),
                getattr(getattr(new, name), field.name),
                rtol=0,
                atol=0,
            )
            tensor_fields += 1
    for name in (
        "all_table_indices",
        "rotameric_table_indices",
        "semirotameric_table_indices",
    ):
        pandas.testing.assert_frame_equal(getattr(old, name), getattr(new, name))
    assert tensor_storage_bytes(old) == tensor_storage_bytes(new)
    bytes_per_resolver = tensor_storage_bytes(new)

    def released_owner(cls):
        private = attr.evolve(
            database,
            dun_lookup=(
                *database.dun_lookup,
                attr.evolve(database.dun_lookup[0], residue_name="profile_lifetime"),
            ),
        )
        resolver = cls.from_database(private, device)
        refs = weakref.ref(private), weakref.ref(resolver)
        return refs

    lifetimes = {}
    for name, cls in classes.items():
        owner, resolver = released_owner(cls)
        gc.collect()
        lifetimes[name] = dict(
            database_retained=owner() is not None,
            resolver_retained=resolver() is not None,
            retained_derived_tensor_bytes=(
                tensor_storage_bytes(resolver()) if resolver() is not None else 0
            ),
        )
    assert lifetimes["baseline"]["database_retained"]
    assert not lifetimes["candidate"]["database_retained"]
    assert not lifetimes["candidate"]["resolver_retained"]
    timings = {name: [] for name in classes}
    for round_index in range(5):
        for name in list(classes)[:: -1 if round_index % 2 else 1]:
            cls = classes[name]
            start = time.perf_counter()
            for _ in range(100):
                cls.from_database(database, device)
            timings[name].append((time.perf_counter() - start) / 100)
    assert hashlib.sha256(Path(SOURCE).read_bytes()).hexdigest() == source_sha
    result = dict(
        baseline=BASELINE,
        baseline_source_sha256=hashlib.sha256(old_source.encode()).hexdigest(),
        candidate_source_sha256=source_sha,
        device=str(device),
        torch=torch.__version__,
        exactly_equal_tensor_fields=tensor_fields,
        exactly_equal_dataframes=3,
        unique_tensor_storage_bytes_per_resolver=bytes_per_resolver,
        released_private_owner=lifetimes,
        warm_lookup_round_seconds=timings,
        warm_lookup_median_seconds={
            name: statistics.median(values) for name, values in timings.items()
        },
        limits="Derived tensor storage only, deduplicated by device/storage pointer. Excludes source database, Python/native overhead, allocator reservations, active consumers and process RSS. Four-entry LRU limits cached resolvers, not bytes for arbitrarily large libraries. Lookup timings exclude all construction/scoring/sampling and perform no device work.",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
