import numpy

from tmol.pose.packed_block_types import residue_types_from_residues, PackedBlockTypes


def test_load_packed_residue_types(ubq_res, torch_device):
    rt_list = residue_types_from_residues(ubq_res)
    pbt = PackedBlockTypes.from_restype_list(rt_list, torch_device)
    assert pbt


def test_packed_residue_type_indexer(ubq_res, torch_device):
    rt_list = residue_types_from_residues(ubq_res)
    pbt = PackedBlockTypes.from_restype_list(rt_list, torch_device)

    inds = pbt.inds_for_res(ubq_res)
    for i, res in enumerate(ubq_res):
        assert len(res.residue_type.atoms) == pbt.n_atoms[inds[i]]


def test_packed_residue_type_atoms_downstream_of_conn(ubq_res, torch_device):
    rt_list = residue_types_from_residues(ubq_res)
    pbt = PackedBlockTypes.from_restype_list(rt_list, torch_device)

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


def test_packed_block_types_ordered_torsions(ubq_res, torch_device):
    rt_list = residue_types_from_residues(ubq_res)
    pbt = PackedBlockTypes.from_restype_list(rt_list, torch_device)

    assert pbt.n_torsions.device == torch_device
    assert pbt.torsion_uaids.device == torch_device

    max_n_tor = pbt.torsion_uaids.shape[1]
    for i in range(len(rt_list)):
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
