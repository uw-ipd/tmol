import attr
import torch
import pandas

from tmol.database.chemical import ChemicalDatabase
from tmol.system.pose import PackedBlockTypes, residue_types_from_residues
from tmol.score.AtomTypeDependentTerm import AtomTypeDependentTerm
from tmol.tests.system.test_pose import two_ubq_poses


def test_store_atom_types_in_packed_residue_types(
    ubq_res, default_database, torch_device
):

    rt_list = residue_types_from_residues(ubq_res)
    pbt = PackedBlockTypes.from_restype_list(
        rt_list, default_database.chemical, torch_device
    )

    atdt = AtomTypeDependentTerm.from_database(default_database.chemical, torch_device)
    atdt.setup_packed_block_types(pbt)

    assert hasattr(pbt, "atom_types")
    assert pbt.atom_types.shape == (pbt.n_types, pbt.max_n_atoms)
    assert pbt.atom_types.dtype == torch.int32
    assert pbt.atom_types.device == torch_device

    for i, rt in enumerate(rt_list):
        for j, at in enumerate(rt.atoms):
            # print(
            #     at.atom_type,
            #     "atdt.atom_type_index.get_indexer([at.name])",
            #     atdt.atom_type_index.get_indexer([at.atom_type]),
            # )
            assert (
                atdt.atom_type_index.get_indexer([at.atom_type])
                == pbt.atom_types[i, j].item()
            )
    # print(atdt.atom_type_index)
    # print(pbt.atom_types)
