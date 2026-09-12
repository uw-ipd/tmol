"""Reusing chemical annotations must preserve each rendered nonbonded scorer."""

import copy

import attr
import pytest
import torch

from tmol.pose import PackedBlockTypes
from tmol.score.hbond import HBondEnergyTerm
from tmol.score.lk_ball import LKBallEnergyTerm
from tmol.score.ljlk import LJLKEnergyTerm
from tmol.tests.score.common import pose_stack_from_pdb_and_resnums


def fresh_annotations(pose):
    old = pose.packed_block_types
    blocks = [copy.copy(bt) for bt in old.active_block_types]
    for bt in blocks:
        for name in list(vars(bt)):
            if name.startswith(
                (
                    "atom_types",
                    "heavy_atom_inds",
                    "atom_unique",
                    "atom_wildcard",
                    "atom_cross",
                    "ljlk_",
                    "lk_ball_",
                    "hbbt_",
                    "_atom_type_",
                    "_hbond_",
                    "_ljlk_",
                    "_lk_ball_",
                )
            ):
                delattr(bt, name)
    pbt = PackedBlockTypes.from_restype_list(
        old.chem_db, old.restype_set, blocks, pose.device
    )
    return attr.evolve(pose, packed_block_types=pbt)


def setup(term, pose):
    for bt in pose.packed_block_types.active_block_types:
        term.setup_block_type(bt)
    term.setup_packed_block_types(pose.packed_block_types)
    term.setup_poses(pose)


def database_change(database, kind):
    if kind == "hbond_family_order":
        hb = database.scoring.hbond
        return attr.evolve(
            database,
            scoring=attr.evolve(
                database.scoring,
                hbond=attr.evolve(
                    hb,
                    donor_type_params=tuple(reversed(hb.donor_type_params)),
                    acceptor_type_params=tuple(reversed(hb.acceptor_type_params)),
                    pair_parameters=tuple(reversed(hb.pair_parameters)),
                ),
            ),
        )
    if kind == "type_order":
        return attr.evolve(
            database,
            chemical=attr.evolve(
                database.chemical,
                atom_types=tuple(reversed(database.chemical.atom_types)),
            ),
        )
    if kind == "ownership":
        return attr.evolve(
            database,
            scoring=attr.evolve(
                database.scoring,
                genbonded=attr.evolve(
                    database.scoring.genbonded, rosetta_typed=frozenset()
                ),
            ),
        )
    if kind == "donors":
        hb = database.scoring.hbond
        hb = attr.evolve(
            hb,
            donor_atom_types=(),
            donor_type_mapper=hb.donor_type_mapper.iloc[:0].copy(),
        )
        return attr.evolve(database, scoring=attr.evolve(database.scoring, hbond=hb))
    if kind == "solvation":
        ljlk = database.scoring.ljlk
        ljlk = attr.evolve(
            ljlk,
            atom_type_parameters=tuple(
                attr.evolve(row, lk_dgfree=1.5 * row.lk_dgfree)
                for row in ljlk.atom_type_parameters
            ),
        )
        return attr.evolve(database, scoring=attr.evolve(database.scoring, ljlk=ljlk))
    raise AssertionError(kind)


@pytest.mark.parametrize(
    "term_class,change",
    [
        (LJLKEnergyTerm, "type_order"),
        (LJLKEnergyTerm, "ownership"),
        (LKBallEnergyTerm, "type_order"),
        (LKBallEnergyTerm, "donors"),
        (LKBallEnergyTerm, "solvation"),
        (HBondEnergyTerm, "donors"),
        (HBondEnergyTerm, "hbond_family_order"),
    ],
)
@pytest.mark.parametrize("changed_first", [False, True])
@pytest.mark.parametrize("block_pairs", [False, True])
def test_reused_pose_retains_each_database_energy_and_gradient(
    ubq_pdb,
    default_database,
    torch_device,
    term_class,
    change,
    changed_first,
    block_pairs,
):
    pose = fresh_annotations(
        pose_stack_from_pdb_and_resnums(ubq_pdb, torch_device, [(0, 16)])
    )
    databases = [default_database, database_change(default_database, change)]
    render = (
        "render_block_pair_scoring_module"
        if block_pairs
        else "render_whole_pose_scoring_module"
    )
    coords = pose.coords.double().clone()
    coords += 0.025 * torch.sin(
        torch.arange(coords.numel(), device=torch_device)
    ).reshape_as(coords)
    coords.requires_grad_(True)
    weights = (
        0.3
        + torch.arange(pose.max_n_blocks**2, device=torch_device).reshape(
            pose.max_n_blocks, pose.max_n_blocks
        )
        / 100
    )

    def evaluate(module):
        values = module(coords)
        scalar = (values * weights).sum() if block_pairs else values.sum()
        return values, torch.autograd.grad(scalar, coords, retain_graph=True)[0]

    expected = []
    for database in databases:
        clean = fresh_annotations(pose)
        term = term_class(database, torch_device)
        setup(term, clean)
        expected.append(evaluate(getattr(term, render)(clean)))
    assert float(expected[0][0].detach().abs().sum()) > 0.01
    if change in ("type_order", "hbond_family_order"):
        for a, b in zip(expected[0], expected[1]):
            torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-6)
    elif change == "donors" and term_class is HBondEnergyTerm:
        for value in expected[1]:
            assert torch.count_nonzero(value) == 0
    else:
        assert not torch.allclose(expected[0][0], expected[1][0])

    order = (1, 0) if changed_first else (0, 1)
    modules, terms = {}, {}
    for state in order:
        term = terms[state] = term_class(databases[state], torch_device)
        setup(term, pose)
        modules[state] = getattr(term, render)(pose)
    # A new render must restore its own annotations without disturbing old modules.
    again = getattr(terms[order[0]], render)(pose)
    for state, module in [*modules.items(), (order[0], again)]:
        for actual, reference in zip(evaluate(module), expected[state]):
            torch.testing.assert_close(actual, reference, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize(
    "term_class,change",
    [
        (LJLKEnergyTerm, "type_order"),
        (LJLKEnergyTerm, "ownership"),
        (LKBallEnergyTerm, "donors"),
        (LKBallEnergyTerm, "solvation"),
        (HBondEnergyTerm, "donors"),
        (HBondEnergyTerm, "hbond_family_order"),
    ],
)
@pytest.mark.parametrize("changed_first", [False, True])
def test_reused_rotamer_pose_retains_each_database(
    ubq_pdb,
    default_database,
    torch_device,
    dun_sampler,
    term_class,
    change,
    changed_first,
):
    from tmol.pack import PackerPalette, PackerTask, SetPackerTask
    from tmol.pack.rotamer import build_rotamers
    from tmol.pose import PoseStackBuilder

    pose = PoseStackBuilder.from_poses(
        [
            pose_stack_from_pdb_and_resnums(ubq_pdb, torch_device, [(0, count)])
            for count in (4, 8)
        ],
        torch_device,
    )
    task = PackerTask(pose, PackerPalette())
    task.restrict_to_repacking()
    task.set_chi_sample_budget(128, 64)
    task.add_conformer_sampler(dun_sampler)
    pose, rotamers = build_rotamers(
        pose, SetPackerTask.from_packer_task(task), default_database.chemical
    )
    pose = fresh_annotations(pose)
    databases = [default_database, database_change(default_database, change)]
    coords = rotamers.coords.double().clone().requires_grad_(True)

    def evaluate(module):
        values, indices = module(coords)
        weights = torch.linspace(
            0.3, 1.7, values.numel(), device=torch_device
        ).reshape_as(values)
        gradient = torch.autograd.grad(
            (values * weights).sum(), coords, retain_graph=True
        )[0]
        return values, indices, gradient

    expected = []
    for db in databases:
        clean = fresh_annotations(pose)
        term = term_class(db, torch_device)
        setup(term, clean)
        expected.append(evaluate(term.render_rotamer_scoring_module(clean, rotamers)))
    assert float(expected[0][0].detach().abs().sum()) > 0.01
    if change in ("type_order", "hbond_family_order"):
        for a, b in zip(expected[0], expected[1]):
            torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-6)
    else:
        assert not torch.allclose(expected[0][0], expected[1][0])
    order = (1, 0) if changed_first else (0, 1)
    modules, terms = {}, {}
    for state in order:
        terms[state] = term_class(databases[state], torch_device)
        setup(terms[state], pose)
        modules[state] = terms[state].render_rotamer_scoring_module(pose, rotamers)
    again = terms[order[0]].render_rotamer_scoring_module(pose, rotamers)
    for state, module in [*modules.items(), (order[0], again)]:
        for actual, reference in zip(evaluate(module), expected[state]):
            torch.testing.assert_close(actual, reference, rtol=2e-6, atol=2e-6)
