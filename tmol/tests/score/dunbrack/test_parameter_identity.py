"""Shared chemical types must not couple independent Dunbrack scoring terms."""

import copy

import attr
import pytest
import torch

from tmol.io import pose_stack_from_pdb
from tmol.pose import PackedBlockTypes, PoseStackBuilder
from tmol.score.dunbrack import DunbrackEnergyTerm


def fresh_pose(pose):
    old = pose.packed_block_types
    blocks = [copy.copy(bt) for bt in old.active_block_types]
    for bt in blocks:
        for name in tuple(vars(bt)):
            if name.startswith(("dunbrack_", "_dunbrack_")):
                delattr(bt, name)
    pbt = PackedBlockTypes.from_restype_list(
        old.chem_db, old.restype_set, blocks, pose.device
    )
    return attr.evolve(pose, packed_block_types=pbt)


def changed_database(database, change):
    dun = database.scoring.dun
    leu = next(
        row.dun_table_name for row in dun.dun_lookup if row.residue_name == "LEU"
    )
    lookup = tuple(
        (
            attr.evolve(row, dun_table_name=leu)
            if row.residue_name == "ILE" and change == "remap"
            else row
        )
        for row in dun.dun_lookup
        if change != "remove" or row.residue_name != "ILE"
    )
    return attr.evolve(
        database,
        scoring=attr.evolve(database.scoring, dun=attr.evolve(dun, dun_lookup=lookup)),
    )


def setup(term, pose):
    for bt in pose.packed_block_types.active_block_types:
        term.setup_block_type(bt)
    term.setup_packed_block_types(pose.packed_block_types)
    term.setup_poses(pose)


def render(term, pose, block_pairs):
    return (
        term.render_block_pair_scoring_module
        if block_pairs
        else term.render_whole_pose_scoring_module
    )(pose)


def evaluate(module, pose):
    coords = pose.coords.detach().double().requires_grad_(True)
    scores = module(coords)
    assert bool(torch.isfinite(scores).all())
    weights = torch.arange(1, scores.numel() + 1, device=pose.device).reshape(
        scores.shape
    )
    gradient = torch.autograd.grad((scores * weights).sum(), coords)[0]
    assert bool(torch.isfinite(gradient).all())
    return scores.detach(), gradient


@pytest.mark.parametrize("change", ["remap", "remove"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("block_pairs", [False, True])
def test_shared_types_preserve_each_rendered_scorer(
    change, reverse, block_pairs, ubq_pdb, default_database, torch_device
):
    pose = fresh_pose(
        PoseStackBuilder.from_poses(
            [
                pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=end)
                for end in (8, 5)
            ],
            device=torch_device,
        )
    )
    databases = [default_database, changed_database(default_database, change)]
    if reverse:
        databases.reverse()
    expected = []
    for database in databases:
        fresh = fresh_pose(pose)
        term = DunbrackEnergyTerm(database, torch_device)
        setup(term, fresh)
        expected.append(evaluate(render(term, fresh, block_pairs), fresh))
    assert not torch.allclose(expected[0][0], expected[1][0])
    assert not torch.allclose(expected[0][1], expected[1][1])
    terms = [DunbrackEnergyTerm(database, torch_device) for database in databases]
    modules = []
    for index in (0, 1, 0, 1):
        setup(terms[index], pose)
        modules.append((index, render(terms[index], pose, block_pairs)))
    # Every previously rendered module must still retain its own configuration.
    for index, module in modules:
        for actual, wanted in zip(evaluate(module, pose), expected[index]):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=1e-10)
    # Rendering again must refresh the calling term's own packed data.
    for index in (0, 1):
        module = render(terms[index], pose, block_pairs)
        for actual, wanted in zip(evaluate(module, pose), expected[index]):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=1e-10)


def test_packed_setup_refreshes_shared_block_annotations(
    ubq_pdb, default_database, torch_device
):
    pose = fresh_pose(pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=8))
    first = DunbrackEnergyTerm(default_database, torch_device)
    setup(first, pose)
    second = DunbrackEnergyTerm(
        changed_database(default_database, "remove"), torch_device
    )
    second.setup_packed_block_types(pose.packed_block_types)
    for bt in pose.packed_block_types.active_block_types:
        if bt.base_name == "ILE":
            assert bt.dunbrack_attrs.rotamer_table_set == -1


def test_scoring_setup_does_not_read_per_residue_device_scalars(
    ubq_pdb, default_database, torch_device, monkeypatch
):
    pose = fresh_pose(pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=8))
    term = DunbrackEnergyTerm(default_database, torch_device)

    def scalar_read(*args, **kwargs):
        raise AssertionError("Scoring setup must use its host metadata")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", scalar_read)
        setup(term, pose)


def test_rendered_scorer_survives_parameter_owner_expiry(
    ubq_pdb, default_database, torch_device
):
    import gc
    import weakref

    pose = fresh_pose(pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=8))
    database = changed_database(default_database, "remap")
    term = DunbrackEnergyTerm(database, torch_device)
    setup(term, pose)
    module = render(term, pose, False)
    expected = evaluate(module, pose)
    owner_ref = weakref.ref(database.scoring.dun)
    resolver_ref = weakref.ref(term.global_params)
    del term, database
    gc.collect()
    assert owner_ref() is None
    assert resolver_ref() is None
    for actual, wanted in zip(evaluate(module, pose), expected):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)


def test_duplicate_lookup_names_are_rejected(default_database, torch_device):
    dun = default_database.scoring.dun
    duplicate = attr.evolve(dun, dun_lookup=(*dun.dun_lookup, dun.dun_lookup[0]))
    database = attr.evolve(
        default_database, scoring=attr.evolve(default_database.scoring, dun=duplicate)
    )
    with pytest.raises(ValueError, match="lookup names must be unique"):
        DunbrackEnergyTerm(database, torch_device)
