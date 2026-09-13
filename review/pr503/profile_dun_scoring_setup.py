"""Compare exact Dunbrack scoring annotations and cold/repeated setup costs."""

import argparse
import copy
import dataclasses
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import numpy
import torch

from tmol.database import ParameterDatabase
from tmol.io import default_packed_block_types
from tmol.score.dunbrack import DunbrackEnergyTerm

BASELINE = "24df324dc8f6299086453546078f7e9af0cada44"
SOURCE = "tmol/score/dunbrack/_dunbrack_energy_term.py"


def fresh_pbt(template):
    pbt = copy.copy(template)
    pbt.active_block_types = [copy.copy(bt) for bt in template.active_block_types]
    for owner in [pbt, *pbt.active_block_types]:
        for name in tuple(vars(owner)):
            if name.startswith(("dunbrack_", "_dunbrack_")):
                delattr(owner, name)
    return pbt


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
    namespace = dict(
        vars(importlib.import_module("tmol.score.dunbrack._dunbrack_energy_term"))
    )
    exec(compile(old_source, "baseline_dun_scoring", "exec"), namespace)
    classes = {
        "baseline": namespace["DunbrackEnergyTerm"],
        "candidate": DunbrackEnergyTerm,
    }
    database = ParameterDatabase.get_default()
    template = default_packed_block_types(device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def setup(term, pbt):
        for bt in pbt.active_block_types:
            term.setup_block_type(bt)
        term.setup_packed_block_types(pbt)

    prepared = {}
    for name, cls in classes.items():
        term = cls(database, device)
        pbt = fresh_pbt(template)
        setup(term, pbt)
        prepared[name] = term, pbt
    old, new = (prepared[name][1] for name in classes)
    compared = 0
    for old_bt, new_bt in zip(old.active_block_types, new.active_block_types):
        for field in dataclasses.fields(new_bt.dunbrack_attrs):
            numpy.testing.assert_array_equal(
                getattr(old_bt.dunbrack_attrs, field.name),
                getattr(new_bt.dunbrack_attrs, field.name),
            )
            compared += 1
    for first, second in zip(
        old.dunbrack_packed_block_data, new.dunbrack_packed_block_data
    ):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    storage = {
        name: sum(t.numel() * t.element_size() for t in pbt.dunbrack_packed_block_data)
        for name, (_, pbt) in prepared.items()
    }
    results = {}
    for mode in ("cold", "warm"):
        timings = {name: [] for name in classes}
        for round_index in range(5):
            for name in list(classes)[:: -1 if round_index % 2 else 1]:
                measured = []
                for _ in range(5):
                    if mode == "cold":
                        pbt = fresh_pbt(template)
                    else:
                        term, pbt = prepared[name]
                    sync()
                    start = time.perf_counter()
                    if mode == "cold":
                        term = classes[name](database, device)
                    setup(term, pbt)
                    sync()
                    measured.append(time.perf_counter() - start)
                timings[name].append(statistics.median(measured))
        medians = {name: statistics.median(values) for name, values in timings.items()}
        results[mode] = dict(
            round_medians_seconds=timings,
            median_seconds=medians,
            baseline_over_candidate=medians["baseline"] / medians["candidate"],
        )
        print(mode, medians, flush=True)
    assert hashlib.sha256(Path(SOURCE).read_bytes()).hexdigest() == source_sha
    result = dict(
        baseline=BASELINE,
        baseline_source_sha256=hashlib.sha256(old_source.encode()).hexdigest(),
        candidate_source_sha256=source_sha,
        device=str(device),
        torch=torch.__version__,
        n_types=template.n_types,
        exactly_equal_rt_fields=compared,
        exactly_equal_packed_fields=len(new.dunbrack_packed_block_data),
        packed_tensor_bytes=storage,
        results=results,
        limits="Five alternating warm-process rounds of five samples. Cold includes term construction and first RT/PBT setup; warm repeats RT/PBT setup on the same term/types. Database/resolver construction, object copies and scoring excluded. Packed tensor bytes exclude host metadata, temporary arrays, allocator overhead and process RSS. No kernel or whole-application speed claim.",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
