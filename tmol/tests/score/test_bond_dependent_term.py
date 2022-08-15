from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.score.bond_dependent_term import BondDependentTerm

# from tmol.tests.pose.test_pose_stack import two_ubq_poses


def test_create_bond_separation(ubq_res, default_database, torch_device):
    rt_dict = {}
    for res in ubq_res:
        if id(res.residue_type) not in rt_dict:
            rt_dict[id(res.residue_type)] = res.residue_type
    rt_list = [rt for addr, rt in rt_dict.items()]
    pbt = PackedBlockTypes.from_restype_list(rt_list, torch_device)

    bdt = BondDependentTerm(param_db=default_database, device=torch_device)
    bdt.setup_packed_block_types(pbt)

    assert hasattr(pbt, "bond_separation")
    assert hasattr(pbt, "max_n_interblock_bonds")
    assert hasattr(pbt, "n_interblock_bonds")
    assert hasattr(pbt, "atoms_for_interblock_bonds")

    assert pbt.max_n_interblock_bonds == 2
    assert pbt.n_interblock_bonds.device == torch_device
    assert pbt.n_interblock_bonds.shape == (pbt.n_types,)
    assert pbt.atoms_for_interblock_bonds.device == torch_device
    assert pbt.atoms_for_interblock_bonds.shape == (
        pbt.n_types,
        pbt.max_n_interblock_bonds,
    )


def test_create_pose_bond_separation_two_ubq(
    two_ubq_poses, default_database, torch_device
):
    # two_ubq = two_ubq_poses(ubq_res, torch_device)
    bdt = BondDependentTerm(param_db=default_database, device=torch_device)
    bdt.setup_poses(two_ubq_poses)

    # PoseStack should already have this data
    assert hasattr(two_ubq_poses, "inter_block_bondsep")
    assert two_ubq_poses.inter_block_bondsep.shape == (2, 60, 60, 2, 2)
    assert two_ubq_poses.inter_block_bondsep.device == torch_device

    assert hasattr(two_ubq_poses, "min_block_bondsep")

    assert two_ubq_poses.min_block_bondsep.shape == (2, 60, 60)
    assert two_ubq_poses.min_block_bondsep.device == torch_device
