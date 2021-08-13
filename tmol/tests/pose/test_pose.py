import numpy

from tmol.pose.pose_stack import residue_types_from_residues, PackedBlockTypes
from tmol.pose.pose_stack import Pose, Poses


def two_ubq_poses(ubq_res, torch_device):
    p1 = Pose.from_residues_one_chain(ubq_res[:40], torch_device)
    p2 = Pose.from_residues_one_chain(ubq_res[:60], torch_device)
    return Poses.from_poses([p1, p2], torch_device)


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


def test_pose_create_inter_residue_connections(ubq_res, torch_device):
    connections_by_name = Pose.resolve_single_chain_connections(ubq_res[:4])
    inter_residue_connections = Pose.create_inter_residue_connections(
        ubq_res[:4], connections_by_name, torch_device
    )

    assert inter_residue_connections.shape == (4, 2, 2)

    assert inter_residue_connections[0, 0, 0] == -1
    assert inter_residue_connections[0, 0, 1] == -1

    assert inter_residue_connections[0, 1, 0] == 1
    assert inter_residue_connections[0, 1, 1] == 0
    assert inter_residue_connections[1, 0, 0] == 0
    assert inter_residue_connections[1, 0, 1] == 1

    assert inter_residue_connections[1, 1, 0] == 2
    assert inter_residue_connections[1, 1, 1] == 0
    assert inter_residue_connections[2, 0, 0] == 1
    assert inter_residue_connections[2, 0, 1] == 1

    assert inter_residue_connections[2, 1, 0] == 3
    assert inter_residue_connections[2, 1, 1] == 0
    assert inter_residue_connections[3, 0, 0] == 2
    assert inter_residue_connections[3, 0, 1] == 1

    assert inter_residue_connections[3, 1, 0] == -1
    assert inter_residue_connections[3, 1, 1] == -1


def test_pose_resolve_bond_separation(ubq_res, torch_device):
    connections = Pose.resolve_single_chain_connections(ubq_res[1:4])
    bonds = Pose.determine_inter_block_bondsep(ubq_res[1:4], connections, torch_device)
    assert bonds[0, 1, 1, 0] == 1
    assert bonds[1, 2, 1, 0] == 1
    assert bonds[1, 0, 0, 1] == 1
    assert bonds[2, 1, 0, 1] == 1
    assert bonds[0, 2, 1, 0] == 4
    assert bonds[2, 0, 0, 1] == 4


def test_pose_ctor_smoke(ubq_res, torch_device):
    p = Pose.from_residues_one_chain(ubq_res, torch_device)
    assert p


def test_poses_ctor(ubq_res, torch_device):
    p1 = Pose.from_residues_one_chain(ubq_res[:40], torch_device)
    p2 = Pose.from_residues_one_chain(ubq_res[:60], torch_device)
    poses = Poses.from_poses([p1, p2], torch_device)
    assert poses.block_type_ind.shape == (2, 60)
    max_n_atoms = poses.packed_block_types.max_n_atoms
    assert poses.coords.shape == (2, 60, max_n_atoms, 3)
    assert poses.inter_block_bondsep.shape == (2, 60, 60, 2, 2)
