"""CPU pose totals must retain small block-pair contributions."""

import pytest
import torch

from tmol.io import pose_stack_from_cif
from tmol.score.ljlk import LJLKEnergyTerm
from tmol.tests.data import data_path


@pytest.mark.parametrize("requires_grad", [False, True])
def test_cpu_pose_total_matches_precise_sum_of_block_pairs(requires_grad):
    device = torch.device("cpu")
    pose, context = pose_stack_from_cif(
        data_path("covalent_fixtures") / "nglycan_tree_1ax2.cif",
        device,
        prepare_ligands=True,
        ligand_seed=20250828,
        no_optH=True,
        return_context=True,
    )
    term = LJLKEnergyTerm(context.parameter_database, device)
    for bt in pose.packed_block_types.active_block_types:
        term.setup_block_type(bt)
    term.setup_packed_block_types(pose.packed_block_types)
    term.setup_poses(pose)
    whole = term.render_whole_pose_scoring_module(pose)
    pairs = term.render_block_pair_scoring_module(pose)
    coords = pose.coords.detach().requires_grad_(requires_grad)
    expected = pairs(coords).double().sum(dim=(-1, -2))
    # Allow the final cast to float32, not one rounding per block pair.
    torch.testing.assert_close(
        whole(coords),
        expected,
        rtol=torch.finfo(torch.float32).eps,
        atol=0,
        check_dtype=False,
    )
