"""An empty declared HBond family yields zero native scores and gradients."""

import attr
import pytest
import torch

from tmol.io import pose_stack_from_pdb
from tmol.score.hbond import HBondEnergyTerm


@pytest.mark.parametrize("empty", ["donor", "acceptor", "both"])
def test_empty_family_scores_zero(default_database, ubq_pdb, torch_device, empty):
    hb = default_database.scoring.hbond
    fields = dict(pair_parameters=(), polynomial_parameters=())
    for kind in (["donor", "acceptor"] if empty == "both" else [empty]):
        fields[kind + "_type_params"] = ()
        fields[kind + "_atom_types"] = ()
        fields[kind + "_type_mapper"] = (
            getattr(hb, kind + "_type_mapper").iloc[:0].copy()
        )
    database = attr.evolve(
        default_database,
        scoring=attr.evolve(default_database.scoring, hbond=attr.evolve(hb, **fields)),
    )
    pose = pose_stack_from_pdb(ubq_pdb, torch_device)
    term = HBondEnergyTerm(database, torch_device)
    term.setup_packed_block_types(pose.packed_block_types)
    term.setup_poses(pose)
    coords = pose.coords.detach().requires_grad_(True)
    energy = term.render_whole_pose_scoring_module(pose)(coords)
    gradient = torch.autograd.grad(energy.sum(), coords)[0]
    assert torch.count_nonzero(energy) == 0
    assert torch.count_nonzero(gradient) == 0
