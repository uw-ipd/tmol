from tmol.system.pose import residue_types_from_residues, PackedBlockTypes, Pose, Poses


def two_ubq_poses(default_database, ubq_res):
    p1 = Pose.from_residues_one_chain(ubq_res[:40], default_database.chemical)
    p2 = Pose.from_residues_one_chain(ubq_res[:60], default_database.chemical)
    return Poses.from_poses([p1, p2], default_database.chemical)


def test_load_packed_residue_types(ubq_res, default_database):
    rt_list = residue_types_from_residues(ubq_res)
    pbt = PackedBlockTypes.from_restype_list(rt_list, default_database.chemical)


def test_packed_residue_type_indexer(ubq_res, default_database):
    rt_list = residue_types_from_residues(ubq_res)
    pbt = PackedBlockTypes.from_restype_list(rt_list, default_database.chemical)

    inds = pbt.inds_for_res(ubq_res)
    for i, res in enumerate(ubq_res):
        assert len(res.residue_type.atoms) == pbt.n_atoms[inds[i]]


def test_pose_resolve_bond_separation(ubq_res, default_database):
    bonds = Pose.resolve_single_chain_inter_block_bondsep(ubq_res[1:4])
    assert bonds[0, 1, 1, 0] == 1
    assert bonds[1, 2, 1, 0] == 1
    assert bonds[1, 0, 0, 1] == 1
    assert bonds[2, 1, 0, 1] == 1
    assert bonds[0, 2, 1, 0] == 4
    assert bonds[2, 0, 0, 1] == 4


def test_pose_ctor(ubq_res, default_database):
    p = Pose.from_residues_one_chain(ubq_res, default_database.chemical)


def test_poses_ctor(ubq_res, default_database):
    p1 = Pose.from_residues_one_chain(ubq_res[:40], default_database.chemical)
    p2 = Pose.from_residues_one_chain(ubq_res[:60], default_database.chemical)
    poses = Poses.from_poses([p1, p2], default_database.chemical)
    assert poses.block_inds.shape == (2, 60)
    max_n_atoms = poses.packed_block_types.max_n_atoms
    assert poses.coords.shape == (2, 60, max_n_atoms, 3)
    assert poses.inter_block_bondsep.shape == (2, 60, 60, 2, 2)
