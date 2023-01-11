from tmol.score.bond_dependent_term import BondDependentTerm


def test_create_pose_bond_separation_two_ubq(
    two_ubq_poses, default_database, torch_device
):
    bdt = BondDependentTerm(param_db=default_database, device=torch_device)
    bdt.setup_poses(two_ubq_poses)

    # PoseStack should already have this data
    assert hasattr(two_ubq_poses, "inter_block_bondsep")
    assert two_ubq_poses.inter_block_bondsep.shape == (2, 60, 60, 2, 2)
    assert two_ubq_poses.inter_block_bondsep.device == torch_device

    assert hasattr(two_ubq_poses, "min_block_bondsep")

    assert two_ubq_poses.min_block_bondsep.shape == (2, 60, 60)
    assert two_ubq_poses.min_block_bondsep.device == torch_device
