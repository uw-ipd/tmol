"""Report L/D packing coverage; a diagnostic report is not a passing test."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import torch

from tmol.database import ParameterDatabase
from tmol.pack import PackerPalette, PackerTask, SetPackerTask
from tmol.pack.rotamer import IncludeCurrentSampler, FixedAAChiSampler, build_rotamers
from tmol.pack.rotamer.dunbrack import create_dunbrack_sampler_from_database
from tmol.tests.score.test_mirror_image_scoring import _pose, MIRROR_PAIR


def records(database, device, side):
    pose = _pose(f"{MIRROR_PAIR}_{side}", database, device)
    task = PackerTask(pose, PackerPalette())
    task.restrict_to_repacking()
    for sampler in (
        IncludeCurrentSampler(),
        FixedAAChiSampler(),
        create_dunbrack_sampler_from_database(database, device),
    ):
        task.add_conformer_sampler(sampler)
    pose, rotamers = build_rotamers(
        pose, SetPackerTask.from_packer_task(task), database.chemical
    )
    coords = rotamers.coords.detach().cpu().numpy()
    groups = defaultdict(list)
    for index, (block, ti, offset) in enumerate(
        zip(
            rotamers.block_ind_for_rot.cpu().tolist(),
            rotamers.block_type_ind_for_rot.cpu().tolist(),
            rotamers.coord_offset_for_rot.cpu().tolist(),
        )
    ):
        rt = pose.packed_block_types.active_block_types[ti]
        names = tuple(sorted(rt.atom_to_idx))
        atom_types = {a.name: a.atom_type for a in rt.atoms}
        elements = {a.name: a.element for a in database.chemical.atom_types}
        heavy = numpy.array([elements[atom_types[n]] != "H" for n in names])
        xyz = coords[offset + numpy.array([rt.atom_to_idx[n] for n in names])]
        groups[block, names].append((index, rt.name, xyz, heavy))
    return groups, len(rotamers.block_ind_for_rot)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if args.device == "cuda"
        else torch.device("cpu")
    )
    database = ParameterDatabase.get_default().with_symmetric_gly()
    left, nl = records(database, device, "l")
    right, nr = records(database, device, "d")
    assert (
        left.keys() == right.keys()
    ), "Prepared atom-name/type group inventories differ"
    report = []
    for key in left:
        a, b = left[key], right[key]
        xyz_a = numpy.array([v[2] for v in a])
        xyz_b = numpy.array([v[2] for v in b])
        heavy = a[0][3]
        # Match whole conformers one-to-one on named heavy atoms. Rectangular
        # groups retain their unmatched count; nearest-neighbor reuse could
        # hide missing or duplicate conformers.
        cost = cdist(
            xyz_a[:, heavy].reshape(len(a), -1), -xyz_b[:, heavy].reshape(len(b), -1)
        )
        rows, columns = linear_sum_assignment(cost)
        errors = numpy.abs(xyz_a[rows] + xyz_b[columns])
        heavy_errors = errors[:, heavy].max(axis=(1, 2))
        worst = int(heavy_errors.argmax())
        report.append(
            dict(
                block=key[0],
                l_type=a[0][1],
                d_type=b[0][1],
                atom_names=key[1],
                l_count=len(a),
                d_count=len(b),
                matched=len(rows),
                max_heavy_error_angstrom=float(heavy_errors.max()),
                matched_heavy_errors_over_1e_3=int((heavy_errors > 1e-3).sum()),
                max_named_atom_error_angstrom=float(errors.max()),
                worst_heavy_rotamer_pair=[
                    a[int(rows[worst])][0],
                    b[int(columns[worst])][0],
                ],
            )
        )
    paths = (
        "tmol/pack/rotamer/dunbrack/dispatch.impl.hh",
        "tmol/pack/rotamer/dunbrack/_dunbrack_chi_sampler.py",
        "tmol/pack/rotamer/_build_rotamers.py",
        "tmol/pack/rotamer/_chi_sampler.py",
        "tmol/database/scoring/_mirrored_dunbrack.py",
        "review/pr503/check_mirror_packing.py",
    )
    output = dict(
        fixture=MIRROR_PAIR,
        device=str(device),
        torch=torch.__version__,
        source_sha256={
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths
        },
        l_conformers=nl,
        d_conformers=nr,
        groups=report,
        passes_count_and_named_heavy_geometry_gate=all(
            r["l_count"] == r["d_count"] and r["matched_heavy_errors_over_1e_3"] == 0
            for r in report
        ),
        limits="Diagnostic only; does not claim mirror-image packing is validated. Exact input reflection, symmetrized glycine tables, no hydrogen optimization, default repacking palette and 0.98 library coverage. One-to-one matching within residue/atom-name groups on heavy-atom coordinates, including rectangular groups; unmatched counts remain failures. Named hydrogen discrepancies can reflect equivalent-atom permutations and are not independently classified here. No energy-table/annealing equivalence is implied.",
    )
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in output.items() if k not in ("groups", "source_sha256")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
