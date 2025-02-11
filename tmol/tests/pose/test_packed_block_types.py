import numpy

from tmol.pose.packed_block_types import PackedBlockTypes


def test_load_packed_residue_types(
    default_database, fresh_default_restype_set, torch_device
):
    pbt = PackedBlockTypes.from_restype_list(
        default_database.chemical,
        fresh_default_restype_set,
        fresh_default_restype_set.residue_types,
        torch_device,
    )
    assert pbt


def test_determine_real_atoms(
    default_database, fresh_default_restype_set, torch_device
):
    pbt = PackedBlockTypes.from_restype_list(
        default_database.chemical,
        fresh_default_restype_set,
        fresh_default_restype_set.residue_types,
        torch_device,
    )
    for i, bt in enumerate(pbt.active_block_types):
        i_nats = bt.n_atoms
        for j in range(pbt.max_n_atoms):
            assert pbt.atom_is_real[i, j] == (j < i_nats)


# def test_packed_residue_type_indexer(default_database, torch_device):
#     # rt_list = residue_types_from_residues(ubq_res)
#     # pbt = PackedBlockTypes.from_restype_list(
#     #     default_database.chemical, rt_list, torch_device
#     # )
#     restype_set = ResidueTypeSet.from_database(default_database.chemical)
#     pbt = PackedBlockTypes.from_restype_list(
#         default_database.chemical, restype_set, restype_set.residue_types, torch_device
#     )
#     inds = pbt.inds_for_res(ubq_res)
#     for i, res in enumerate(ubq_res):
#         assert len(res.residue_type.atoms) == pbt.n_atoms[inds[i]]


def test_packed_residue_type_atoms_downstream_of_conn(
    default_database, fresh_default_restype_set, torch_device
):
    rt_list = fresh_default_restype_set.residue_types
    pbt = PackedBlockTypes.from_restype_list(
        default_database.chemical,
        fresh_default_restype_set,
        fresh_default_restype_set.residue_types,
        torch_device,
    )

    max_n_conn = max(len(rt.connections) for rt in rt_list)

    assert pbt.atom_downstream_of_conn.device == torch_device
    assert pbt.atom_downstream_of_conn.shape == (
        pbt.n_types,
        max_n_conn,
        pbt.max_n_atoms,
    )
    pbt_adoc = pbt.atom_downstream_of_conn.cpu().numpy()

    for i, res in enumerate(rt_list):
        adoc = res.atom_downstream_of_conn
        numpy.testing.assert_equal(
            pbt_adoc[i, : len(res.connections), : len(res.atoms)], adoc
        )


def test_packed_block_types_ordered_torsions(
    default_database, fresh_default_restype_set, torch_device
):
    # rt_list = residue_types_from_residues(ubq_res)
    pbt = PackedBlockTypes.from_restype_list(
        default_database.chemical,
        fresh_default_restype_set,
        fresh_default_restype_set.residue_types,
        torch_device,
    )

    assert pbt.n_torsions.device == torch_device
    assert pbt.torsion_uaids.device == torch_device

    max_n_tor = pbt.torsion_uaids.shape[1]
    for i in range(len(pbt.active_block_types)):
        i_n_tor = pbt.active_block_types[i].ordered_torsions.shape[0]
        numpy.testing.assert_equal(
            pbt.active_block_types[i].ordered_torsions,
            pbt.torsion_uaids[i, :i_n_tor].cpu().numpy(),
        )
        numpy.testing.assert_equal(
            numpy.full((max_n_tor - i_n_tor, 4, 3), -1, dtype=numpy.int32),
            pbt.torsion_uaids[i, i_n_tor:].cpu().numpy(),
        )

        for j in range(max_n_tor):
            assert bool(pbt.torsion_is_real[i, j].cpu().item()) == (j < i_n_tor)


def test_packed_block_types_device(
    fresh_default_restype_set, default_database, torch_device
):
    pbt = PackedBlockTypes.from_restype_list(
        default_database.chemical,
        fresh_default_restype_set,
        fresh_default_restype_set.residue_types,
        torch_device,
    )

    assert pbt.atom_is_real.device == torch_device
    assert pbt.atom_is_hydrogen.device == torch_device
    assert pbt.atom_downstream_of_conn.device == torch_device
    assert pbt.atom_paths_from_conn.device == torch_device
    assert pbt.n_torsions.device == torch_device
    assert pbt.torsion_is_real.device == torch_device
    assert pbt.torsion_uaids.device == torch_device
    assert pbt.n_bonds.device == torch_device
    assert pbt.bond_is_real.device == torch_device
    assert pbt.bond_indices.device == torch_device
    assert pbt.n_conn.device == torch_device
    assert pbt.conn_is_real.device == torch_device
    assert pbt.conn_atom.device == torch_device
    assert pbt.down_conn_inds.device == torch_device
    assert pbt.up_conn_inds.device == torch_device
    assert pbt.device == torch_device
