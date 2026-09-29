import torch

from tmol.chemical import MAX_SIG_BOND_SEPARATION
from tmol.score import BondDependentTerm


def test_create_pose_bond_separation_two_ubq(
    ubq_40_60_pose_stack, default_database, torch_device
):
    bdt = BondDependentTerm(param_db=default_database, device=torch_device)
    bdt.setup_poses(ubq_40_60_pose_stack)

    # PoseStack should already have this data
    assert hasattr(ubq_40_60_pose_stack, "inter_block_bondsep")
    max_n_conn = ubq_40_60_pose_stack.packed_block_types.max_n_conn
    assert ubq_40_60_pose_stack.inter_block_bondsep.shape == (
        2,
        60,
        60,
        max_n_conn,
        max_n_conn,
    )
    assert ubq_40_60_pose_stack.inter_block_bondsep.device == torch_device

    # the scoring kernels read each block pair's minimum separation from the
    # near-block table, and the cap for every pair it does not list
    near_blocks = ubq_40_60_pose_stack.inter_block_bondsep.near_blocks
    assert near_blocks.shape[:2] == (2, 60)
    assert near_blocks.device == torch_device
    dense_min = torch.amin(
        ubq_40_60_pose_stack.inter_block_bondsep.to_dense(), dim=(3, 4)
    ).to(torch.int32)
    table_min = torch.full_like(dense_min, MAX_SIG_BOND_SEPARATION)
    pose, block1, slot = torch.nonzero(near_blocks[..., 0] >= 0, as_tuple=True)
    table_min[pose, block1, near_blocks[pose, block1, slot, 0].long()] = near_blocks[
        pose, block1, slot, 1
    ]
    torch.testing.assert_close(table_min, dense_min)
