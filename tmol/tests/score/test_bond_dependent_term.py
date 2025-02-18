from tmol.score.bond_dependent_term import BondDependentTerm


def test_create_pose_bond_separation_two_ubq(
    ubq_40_60_pose_stack, default_database, torch_device
):
    bdt = BondDependentTerm(param_db=default_database, device=torch_device)
    bdt.setup_poses(ubq_40_60_pose_stack)

    # PoseStack should already have this data
    assert hasattr(ubq_40_60_pose_stack, "inter_block_bondsep")
    assert ubq_40_60_pose_stack.inter_block_bondsep.shape == (2, 60, 60, 3, 3)
    assert ubq_40_60_pose_stack.inter_block_bondsep.device == torch_device

    assert hasattr(ubq_40_60_pose_stack, "min_block_bondsep")

    assert ubq_40_60_pose_stack.min_block_bondsep.shape == (2, 60, 60)
    assert ubq_40_60_pose_stack.min_block_bondsep.device == torch_device
