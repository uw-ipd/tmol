import torch
import attr
import numpy

from tmol.pose import PackedBlockTypes
from tmol.score import AtomTypeDependentTerm


def test_setup_block_type(fresh_default_restype_set, default_database, torch_device):
    rt_list = fresh_default_restype_set.residue_types
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


def test_new_packed_set_uses_current_resolver_for_shared_blocks(
    default_database, fresh_default_restype_set, torch_device
):
    original = AtomTypeDependentTerm(default_database, torch_device)
    restypes = fresh_default_restype_set.residue_types
    for rt in restypes:
        original.setup_block_type(rt)
    reversed_database = attr.evolve(
        default_database,
        chemical=attr.evolve(
            default_database.chemical,
            atom_types=tuple(reversed(default_database.chemical.atom_types)),
        ),
    )
    current = AtomTypeDependentTerm(reversed_database, torch_device)
    pbt = PackedBlockTypes.from_restype_list(
        reversed_database.chemical, fresh_default_restype_set, restypes, torch_device
    )
    current.setup_packed_block_types(pbt)
    elements = {at.name: at.element for at in reversed_database.chemical.atom_types}
    indices = pbt.atom_types.cpu().numpy()
    heavy = pbt.heavy_atom_inds.cpu().numpy()
    counts = pbt.n_heavy_atoms.cpu().numpy()
    for i, rt in enumerate(restypes):
        numpy.testing.assert_array_equal(
            indices[i, : len(rt.atoms)],
            current.atom_type_index.get_indexer([a.atom_type for a in rt.atoms]),
        )
        expected = [
            j for j, atom in enumerate(rt.atoms) if elements[atom.atom_type] != "H"
        ]
        assert counts[i] == len(expected)
        numpy.testing.assert_array_equal(heavy[i, : counts[i]], expected)
        assert numpy.all(heavy[i, counts[i] :] == -1)
