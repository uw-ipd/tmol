import torch

from tmol.system.pose import PackedBlockTypes
from tmol.score.BondDependentTerm import BondDependentTerm
from tmol.tests.system.test_pose import two_ubq_poses


def test_create_bond_separation(ubq_res, default_database, torch_device):
    rt_dict = {}
    for res in ubq_res:
        if id(res.residue_type) not in rt_dict:
            rt_dict[id(res.residue_type)] = res.residue_type
    rt_list = [rt for addr, rt in rt_dict.items()]
    pbt = PackedBlockTypes.from_restype_list(rt_list, default_database.chemical)

    bdt = BondDependentTerm(device=torch_device)
    bdt.setup_packed_block_types(pbt)

    assert hasattr(pbt, "bond_separation")
    assert hasattr(pbt, "max_n_interres_bonds")
    assert hasattr(pbt, "n_interres_bonds")
    assert hasattr(pbt, "atoms_for_interres_bonds")

    assert pbt.max_n_interres_bonds == 2
    assert pbt.n_interres_bonds.device == torch_device
    assert pbt.n_interres_bonds.shape == (pbt.n_types,)
    assert pbt.atoms_for_interres_bonds.device == torch_device
    assert pbt.atoms_for_interres_bonds.shape == (pbt.n_types, pbt.max_n_interres_bonds)


def test_create_pose_bond_separation_two_ubq(ubq_res, default_database, torch_device):
    two_ubq = two_ubq_poses(default_database, ubq_res)
    bdt = BondDependentTerm(device=torch_device)
    bdt.setup_poses(two_ubq)

    assert hasattr(two_ubq, "min_block_bondsep")
    assert hasattr(two_ubq, "inter_block_bondsep_t")

    assert two_ubq.min_block_bondsep.shape == (2, 60, 60)
    assert two_ubq.inter_block_bondsep_t.shape == (2, 60, 60, 2, 2)
    assert two_ubq.min_block_bondsep.device == torch_device
    assert two_ubq.inter_block_bondsep_t.device == torch_device
