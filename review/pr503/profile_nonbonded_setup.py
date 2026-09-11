"""Alternating-process setup comparison with exact annotation inventories.

Constructors and new BT/PBT annotation are timed separately. Input construction,
imports, native compilation and validation copies are outside the timed regions.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time


def worker(args):  # noqa: C901 - timing and validation share the same local arrays
    if args.variant == "baseline":
        from check_nonbonded_baseline import BaselineImports, MODULES

        MODULES.add("tmol.score.hbond._params")
        sys.meta_path.insert(0, BaselineImports())

    import attr
    import numpy as np
    import torch
    from tmol.database import ParameterDatabase
    from tmol.io import extended_pose_stack_from_sequences
    from tmol.score.hbond import HBondEnergyTerm
    from tmol.score.lk_ball import LKBallEnergyTerm
    from tmol.score.ljlk import LJLKEnergyTerm
    from tmol.tests.score.test_nonbonded_parameter_identity import (
        fresh_annotations,
        setup,
    )

    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    database = ParameterDatabase.get_default()
    template = extended_pose_stack_from_sequences(["AA"], device=device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def arrays(value, prefix=""):
        if isinstance(value, torch.Tensor):
            yield prefix, value
        elif isinstance(value, np.ndarray) and value.dtype != object:
            yield prefix, value
        elif attr.has(type(value)):
            for field in attr.fields(type(value)):
                yield from arrays(getattr(value, field.name), prefix + "/" + field.name)
        elif isinstance(value, dict):
            for name, child in value.items():
                yield from arrays(child, prefix + "/" + str(name))
        elif isinstance(value, (tuple, list)):
            for i, child in enumerate(value):
                yield from arrays(child, prefix + "/" + str(i))

    def inventory(pbt):
        fields = (
            "atom_types",
            "heavy_atom_inds",
            "n_heavy_atoms",
            "atom_unique_ids",
            "atom_wildcard_ids",
            "atom_cross_ids",
            "ljlk_heavy_atoms_in_tile",
            "ljlk_n_heavy_atoms_in_tile",
            "ljlk_bond_separation",
            "ljlk_all_atoms_ligand_typed",
            "hbbt_params",
            "hbpbt_params",
            "lk_ball_params",
        )
        result = {}
        for prefix, owner in [
            ("packed", pbt),
            *[(f"block/{i}", bt) for i, bt in enumerate(pbt.active_block_types)],
        ]:
            for field in fields:
                if not hasattr(owner, field):
                    continue
                for key, array in arrays(getattr(owner, field), prefix + "/" + field):
                    data = (
                        array.detach().cpu().numpy()
                        if isinstance(array, torch.Tensor)
                        else array
                    )
                    result[key] = [
                        str(data.dtype),
                        list(data.shape),
                        hashlib.sha256(data.tobytes()).hexdigest(),
                    ]
        return result

    def storage(value):
        # Merge memory intervals so tensor/NumPy views count once.
        intervals = {}
        for _, array in arrays(value):
            if isinstance(array, torch.Tensor):
                buf = array.untyped_storage()
                dev, address, size = str(array.device), buf.data_ptr(), buf.nbytes()
            else:
                dev, address, size = (
                    "cpu",
                    array.__array_interface__["data"][0],
                    array.nbytes,
                )
            if size:
                intervals.setdefault(dev, []).append((address, address + size))
        result = {}
        for dev, segments in intervals.items():
            total, last = 0, -1
            for start, stop in sorted(segments):
                total += max(0, stop - max(start, last))
                last = max(last, stop)
            result[dev] = total
        return result

    rows = {}
    for cls in (LJLKEnergyTerm, HBondEnergyTerm, LKBallEnergyTerm):
        constructor, annotation, warm = [], [], []
        fingerprint = None
        for repeat in range(args.samples + 1):
            pose = fresh_annotations(template)
            sync()
            start = time.perf_counter()
            term = cls(database, device)
            sync()
            constructed = time.perf_counter()
            setup(term, pose)
            sync()
            annotated = time.perf_counter()
            if repeat:
                constructor.append(constructed - start)
                annotation.append(annotated - constructed)
            current = inventory(pose.packed_block_types)
            assert fingerprint is None or fingerprint == current
            fingerprint = current
            sync()
            start = time.perf_counter()
            for _ in range(50):
                term.setup_packed_block_types(pose.packed_block_types)
            sync()
            elapsed = (time.perf_counter() - start) / 50
            if repeat:
                warm.append(elapsed)
        rows[cls.__name__] = dict(
            constructor_seconds=constructor,
            construction_and_annotation_seconds=[
                a + b for a, b in zip(constructor, annotation)
            ],
            annotation_seconds=annotation,
            warm_packed_seconds=warm,
            annotations=fingerprint,
            term_array_storage_bytes=storage(vars(term)),
        )
    args.output.write_text(
        json.dumps(
            dict(
                variant=args.variant,
                device=str(device),
                torch=torch.__version__,
                numpy=np.__version__,
                n_types=template.packed_block_types.n_types,
                n_atoms=int(template.packed_block_types.n_atoms.sum()),
                rows=rows,
            ),
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--variant", choices=("baseline", "candidate"))
    args = parser.parse_args()
    if args.variant:
        worker(args)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    directory = args.output.with_suffix("")
    directory.mkdir(exist_ok=True)
    runs = []
    for repeat in range(args.rounds):
        order = (
            ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
        )
        pair = {}
        for variant in order:
            output = directory / f"{repeat}-{variant}.json"
            with (directory / f"{repeat}-{variant}.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        "--device",
                        args.device,
                        "--output",
                        str(output),
                        "--variant",
                        variant,
                        "--samples",
                        str(args.samples),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            pair[variant] = json.loads(output.read_text())
        for term in pair["baseline"]["rows"]:
            assert (
                pair["baseline"]["rows"][term]["annotations"]
                == pair["candidate"]["rows"][term]["annotations"]
            ), term
        runs.append(pair)
        print(f"round {repeat + 1}: all annotation arrays match exactly", flush=True)
    summary = {}
    for term in runs[0]["baseline"]["rows"]:
        summary[term] = {}
        for metric in (
            "constructor_seconds",
            "construction_and_annotation_seconds",
            "annotation_seconds",
            "warm_packed_seconds",
        ):
            values = {
                variant: [
                    statistics.median(pair[variant]["rows"][term][metric])
                    for pair in runs
                ]
                for variant in ("baseline", "candidate")
            }
            medians = {
                variant: statistics.median(data) for variant, data in values.items()
            }
            summary[term][metric] = dict(
                round_medians=values,
                medians=medians,
                baseline_over_candidate=medians["baseline"] / medians["candidate"],
            )
    # The detailed array inventories remain in per-worker JSON files.
    for pair in runs:
        for variant in pair.values():
            for row in variant["rows"].values():
                row.pop("annotations")
    args.output.write_text(
        json.dumps(
            dict(
                baseline="841d594d5",
                exact_annotation_equality=True,
                summary=summary,
                runs=runs,
                limits="Setup only. Warm-hit timing includes new identity checks; term array storage excludes Python objects, inputs, outputs and allocator overhead.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
