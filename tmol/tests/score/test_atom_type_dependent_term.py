import torch

from tmol.pose.packed_block_types import PackedBlockTypes, residue_types_from_residues
from tmol.score.atom_type_dependent_term import AtomTypeDependentTerm


def test_setup_block_type(ubq_res, default_database, torch_device):
    rt_list = residue_types_from_residues(ubq_res)
    atdt = AtomTypeDependentTerm(default_database, torch_device)
    for rt in rt_list:
        atdt.setup_block_type(rt)
        assert hasattr(rt, "atom_types")
        assert hasattr(rt, "heavy_atom_inds")


def test_store_atom_types_in_packed_residue_types(
    default_database, fresh_default_restype_set, torch_device
):
    pbt = PackedBlockTypes.from_restype_list(
        default_database.chemical,
        fresh_default_restype_set,
        fresh_default_restype_set.residue_types,
        torch_device,
    )

    atdt = AtomTypeDependentTerm(default_database, torch_device)
    atdt.setup_packed_block_types(pbt)

    assert hasattr(pbt, "atom_types")
    assert pbt.atom_types.shape == (pbt.n_types, pbt.max_n_atoms)
    assert pbt.atom_types.dtype == torch.int32
    assert pbt.atom_types.device == torch_device

    for i, rt in enumerate(fresh_default_restype_set.residue_types):
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


def test_take_heavyatom_inds_in_range():
    heavy_inds = torch.tensor([0, 1, 2, 3, 4, 5, 8, 10, 11, 12, 13], dtype=torch.int64)
    tile_size = 8
    n_tiles = (heavy_inds.shape[0] - 1) // tile_size + 1
    heavy_subset_wi_tile = torch.full((n_tiles * tile_size,), -1, dtype=torch.int64)
    for i in range(n_tiles):
        subset = (heavy_inds >= i * tile_size) & (heavy_inds < (i + 1) * tile_size)
        # print(subset)
        subset_size = torch.sum(subset)
        s = slice(i * tile_size, i * tile_size + subset_size)
        heavy_subset_wi_tile[s] = heavy_inds[subset]
    # print(heavy_subset_wi_tile)
