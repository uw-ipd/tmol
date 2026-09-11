"""Constraint ablation and paired cold/warm group-topology measurements.

The ablation removes only the new axis restrictions; it is not an upstream
checkout benchmark. Cold/warm comparisons use identical valid conformers.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import attr
import numpy
import torch

from tmol.io import atom_array_from_cif, pose_stack_from_biotite
from tmol.pack import SetPackerTask
from tmol.pack.rotamer import build_rotamers
from tmol.pack.rotamer import _conjugated_groups as group_module
from tmol.pack.rotamer._conjugated_chi_sampler import _fold_group_conformers
from tmol.tests.data import data_path
from tmol.tests.pack.test_conjugated_group_packing import FIXTURES, _task
from tmol.tests.pack.rotamer.test_group_constraints import (
    crosslinked_lysines,
    pose_bonds,
)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clear_shape_caches(pose):
    for name in ("conjugated_sampling_topology_cache", "conjugated_kinforest_cache"):
        cache = getattr(pose.packed_block_types, name, None)
        if cache is not None:
            cache.clear()


def bond_error(pose, group, coordinates):
    indices = []
    for block in group.blocks:
        bt = pose.packed_block_types.active_block_types[
            int(pose.block_type_ind[0, block])
        ]
        start = int(pose.block_coord_offset[0, block])
        indices.extend(range(start, start + bt.n_atoms))
    mapping = {pose_atom: group_atom for group_atom, pose_atom in enumerate(indices)}
    pairs = sorted((a, b) for a, b in pose_bonds(pose) if a in mapping or b in mapping)
    actual = []
    for endpoint in (0, 1):
        actual.append(
            torch.stack(
                [
                    (
                        coordinates[:, mapping[edge[endpoint]]]
                        if edge[endpoint] in mapping
                        else pose.coords[0, edge[endpoint]].expand(len(coordinates), -1)
                    )
                    for edge in pairs
                ],
                dim=1,
            ).double()
        )
    pairs = torch.tensor(pairs, device=pose.device)
    source = pose.coords[0].double()
    expected = (source[pairs[:, 0]] - source[pairs[:, 1]]).norm(dim=-1)
    return float(((actual[0] - actual[1]).norm(dim=-1) - expected).abs().max())


def measure_case(name, device):
    if name in FIXTURES:
        array = atom_array_from_cif(
            data_path("covalent_fixtures", FIXTURES[name] + ".cif")
        )
    else:
        array = crosslinked_lysines(name)
    pose, context = pose_stack_from_biotite(
        array,
        device,
        prepare_ligands=True,
        no_optH=True,
        ligand_seed=503,
        return_context=True,
    )
    task, sampler = _task(pose, context.parameter_database, device)
    task = SetPackerTask.from_packer_task(task)
    pose, _rotamers = build_rotamers(pose, task, context.parameter_database.chemical)
    groups = group_module.find_conjugated_groups(pose)

    def evaluate():
        library = sampler.anchor_library_chi(pose, task, groups)
        enumerated = sampler.group_conformers(pose, library)
        coordinates = []
        for group, columns, conformers in enumerated:
            _kf, dofs, _offsets = sampler.group_coords(pose, group, columns, conformers)
            kf, _ = sampler._group_kinforest(pose, group)
            coordinates.append(_fold_group_conformers(kf, dofs))
        return enumerated, coordinates

    reference, reference_coords = evaluate()
    timings = {"cold_shapes": [], "cached_shapes": []}
    for pair in range(8):
        order = list(timings)
        if pair % 2:
            order.reverse()
        for label in order:
            if label == "cold_shapes":
                clear_shape_caches(pose)
            synchronize(device)
            started = time.perf_counter()
            enumerated, coords = evaluate()
            synchronize(device)
            elapsed = time.perf_counter() - started
            if pair:
                timings[label].append(elapsed)
            for old, new, a, b in zip(
                reference, enumerated, reference_coords, coords, strict=True
            ):
                assert old[:2] == new[:2]
                numpy.testing.assert_array_equal(old[2], new[2])
                torch.testing.assert_close(a, b, atol=1e-5, rtol=0)
    row = {
        "fixture": name,
        "timings": {
            k: {"seconds": v, "median_seconds": statistics.median(v)}
            for k, v in timings.items()
        },
        "groups": [
            {
                "blocks": len(group.blocks),
                "internal_links": len(group.links),
                "external_links": len(group.external_links),
                "columns": len(columns),
                "conformers": len(conformers),
                "coordinate_bytes": xyz.numel() * xyz.element_size(),
                "max_bond_error": bond_error(pose, group, xyz),
            }
            for (group, columns, conformers), xyz in zip(
                reference, reference_coords, strict=True
            )
        ],
    }
    original = group_module._group_sampling_topology

    def without_constraints(group, pose):
        topology = original(group, pose)
        return attr.evolve(
            topology,
            movable_axes={
                frozenset((int(child), int(parent))): int(child)
                for child, parent in enumerate(topology.kinforest.preds)
                if parent >= 0
            },
        )

    try:
        group_module._group_sampling_topology = without_constraints
        clear_shape_caches(pose)
        enumerated, coordinates = evaluate()
        row["unconstrained_axis_ablation"] = [
            {
                "conformers": len(conformers),
                "max_bond_error": bond_error(pose, group, xyz),
            }
            for (group, _columns, conformers), xyz in zip(
                enumerated, coordinates, strict=True
            )
        ]
    finally:
        group_module._group_sampling_topology = original
        clear_shape_caches(pose)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda:0" if args.device == "cuda" else args.device)
    results = {
        "device": str(device),
        "torch": torch.__version__,
        "seed": 503,
        "method": "Seven alternating-order warm pairs with topology/tree cache reuse versus explicit eviction; same conformers. Separate constraint ablation.",
        "rows": [],
    }
    for name in [*sorted(FIXTURES), "free", "cycle", "external"]:
        row = measure_case(name, device)
        results["rows"].append(row)
        print(json.dumps(row), flush=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
