import torch

from tmol.system.pose import PackedResidueTypes
from tmol.score.BondDependentTerm import BondDependentTerm


def test_create_bond_separation(ubq_res, default_database, torch_device):
    rt_dict = {}
    for res in ubq_res:
        if id(res.residue_type) not in rt_dict:
            rt_dict[id(res.residue_type)] = res.residue_type
    rt_list = [rt for addr, rt in rt_dict.items()]
    prt = PackedResidueTypes.from_restype_list(rt_list, default_database.chemical)

    bdt = BondDependentTerm(device=torch_device)
    bdt.setup_packed_restypes(prt)

    assert hasattr(prt, "bond_separation")
    assert hasattr(prt, "max_n_interres_bonds")
    assert hasattr(prt, "n_interres_bonds")
    assert hasattr(prt, "atoms_for_interres_bonds")

    assert prt.max_n_interres_bonds == 2
    assert prt.n_interres_bonds.device == torch_device
    assert prt.n_interres_bonds.shape == (prt.n_types,)
    assert prt.atoms_for_interres_bonds.device == torch_device
    assert prt.atoms_for_interres_bonds.shape == (prt.n_types, prt.max_n_interres_bonds)
