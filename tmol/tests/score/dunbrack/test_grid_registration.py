"""Declared grid starts must agree with interpolation and reflected grid points."""

import attr
import pytest
import torch

from tmol.database.scoring._mirrored_dunbrack import mirror_semi_rotameric_library
from tmol.io import pose_stack_from_pdb
from tmol.score.dunbrack import DunbrackEnergyTerm
from tmol.tests.score.dunbrack.test_parameter_identity import setup, render
from tmol.tests.score.test_mirror_image_scoring import _pose, MIRROR_PAIR


def evaluate(module, pose):
    coords = pose.coords.detach().double().requires_grad_(True)
    scores = module(coords)
    # Keep the derivative tolerance independent of the number of block pairs.
    # Nonuniform weights still exercise each component and output position.
    weights = torch.linspace(
        0.5, 1.5, scores.numel(), device=pose.device, dtype=scores.dtype
    ).reshape(scores.shape)
    gradient = torch.autograd.grad((scores * weights).sum(), coords)[0]
    assert torch.isfinite(scores).all() and torch.isfinite(gradient).all()
    return scores.detach(), gradient


def phe_library(database):
    name = next(
        row.dun_table_name for row in database.dun_lookup if row.residue_name == "PHE"
    )
    return next(
        lib for lib in database.semi_rotameric_libraries if lib.table_name == name
    )


def replace_libraries(database, replacements):
    dun = database.scoring.dun
    return attr.evolve(
        database,
        scoring=attr.evolve(
            database.scoring,
            dun=attr.evolve(
                dun,
                semi_rotameric_libraries=tuple(
                    replacements.get(lib.table_name, lib)
                    for lib in dun.semi_rotameric_libraries
                ),
            ),
        ),
    )


def shifted_origin(library):
    data = library.rotameric_data
    # A circular reindexing plus matching origin shift is the SAME function.
    shifts = (1, -2)
    source = attr.evolve(
        data,
        backbone_dihedral_start=data.backbone_dihedral_start
        + torch.tensor(shifts) * data.backbone_dihedral_step,
        rotamer_probabilities=torch.roll(data.rotamer_probabilities, (-1, 2), (1, 2)),
        rotamer_means=torch.roll(data.rotamer_means, (-1, 2), (1, 2)),
        rotamer_stdvs=torch.roll(data.rotamer_stdvs, (-1, 2), (1, 2)),
        prob_sorted_rot_inds=torch.roll(data.prob_sorted_rot_inds, (-1, 2), (0, 1)),
    )
    return attr.evolve(
        library,
        rotameric_data=source,
        nonrotameric_chi_probabilities=torch.roll(
            library.nonrotameric_chi_probabilities, (-1, 2), (1, 2)
        ),
    )


@pytest.mark.parametrize("block_pairs", [False, True])
def test_semirotameric_grid_reindexing_preserves_energies_and_gradients(
    default_database, ubq_pdb, torch_device, block_pairs
):
    source = phe_library(default_database.scoring.dun)
    shifted = shifted_origin(source)
    private = replace_libraries(default_database, {source.table_name: shifted})
    pose = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=12)
    results = []
    for database in (default_database, private):
        term = DunbrackEnergyTerm(database, torch_device)
        setup(term, pose)
        results.append(evaluate(render(term, pose, block_pairs), pose))
    for before, after in zip(*results):
        torch.testing.assert_close(before, after, atol=2e-4, rtol=2e-5)


def non_aligned_grid(library):
    data = library.rotameric_data
    return attr.evolve(
        library,
        rotameric_data=attr.evolve(
            data,
            backbone_dihedral_start=data.backbone_dihedral_start
            + torch.tensor([1.25, -2.0]),
            backbone_dihedral_step=data.backbone_dihedral_step * torch.tensor([2, 3]),
            rotamer_probabilities=data.rotamer_probabilities[:, ::2, ::3],
            rotamer_means=data.rotamer_means[:, ::2, ::3],
            rotamer_stdvs=data.rotamer_stdvs[:, ::2, ::3],
            prob_sorted_rot_inds=data.prob_sorted_rot_inds[::2, ::3],
        ),
        non_rot_chi_start=library.non_rot_chi_start + 1.25,
        non_rot_chi_step=library.non_rot_chi_step * 2,
        nonrotameric_chi_probabilities=library.nonrotameric_chi_probabilities[
            :, ::2, ::3, ::2
        ],
    )


def test_reflected_grid_points_have_exact_reflected_coordinates(default_database):
    source = non_aligned_grid(phe_library(default_database.scoring.dun))
    mirrored = mirror_semi_rotameric_library(source)
    starts = [*source.rotameric_data.backbone_dihedral_start, source.non_rot_chi_start]
    targets = [
        *mirrored.rotameric_data.backbone_dihedral_start,
        mirrored.non_rot_chi_start,
    ]
    steps = [*source.rotameric_data.backbone_dihedral_step, source.non_rot_chi_step]
    table = source.nonrotameric_chi_probabilities
    for dim, (start, target, step) in enumerate(zip(starts, targets, steps), 1):
        indices = (
            -(float(start) + float(target)) / float(step)
            - torch.arange(table.shape[dim])
        ).remainder(table.shape[dim])
        torch.testing.assert_close(indices, indices.round(), rtol=0, atol=1e-6)
        table = table.index_select(dim, indices.round().long())
    torch.testing.assert_close(
        table, mirrored.nonrotameric_chi_probabilities, rtol=0, atol=0
    )


@pytest.mark.parametrize("block_pairs", [False, True])
def test_non_aligned_grid_mirrors_score_and_coordinate_gradients(
    default_database, torch_device, block_pairs
):
    source = non_aligned_grid(phe_library(default_database.scoring.dun))
    mirrored = mirror_semi_rotameric_library(source)
    database = replace_libraries(
        default_database.with_symmetric_gly(),
        {source.table_name: source, mirrored.table_name: mirrored},
    )
    poses = [
        _pose(f"{MIRROR_PAIR}_{side}", database, torch_device) for side in ("l", "d")
    ]
    for block in range(poses[0].max_n_blocks):
        names = [
            tuple(
                a.name
                for a in pose.packed_block_types.active_block_types[
                    int(pose.block_type_ind[0, block])
                ].atoms
            )
            for pose in poses
        ]
        assert names[0] == names[1]
    torch.testing.assert_close(poses[0].coords, -poses[1].coords, rtol=0, atol=0)
    term = DunbrackEnergyTerm(database, torch_device)
    results = []
    for pose in poses:
        setup(term, pose)
        results.append(evaluate(render(term, pose, block_pairs), pose))
    torch.testing.assert_close(results[0][0], results[1][0], atol=2e-4, rtol=2e-5)
    torch.testing.assert_close(results[0][1], -results[1][1], atol=2e-4, rtol=2e-5)
