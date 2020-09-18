from tmol.system.pose import PackedBlockTypes, Pose


def test_load_packed_residue_types(ubq_res, default_database):
    rt_dict = {}
    for res in ubq_res:
        if id(res.residue_type) not in rt_dict:
            rt_dict[id(res.residue_type)] = res.residue_type
    rt_list = [rt for addr, rt in rt_dict.items()]
    pbt = PackedBlockTypes.from_restype_list(rt_list, default_database.chemical)


def test_create_pose(ubq_res, default_database):
    bonds = Pose.resolve_single_chain_inter_block_separation(ubq_res[1:4])
    assert bonds[0, 1, 1, 0] == 1
    assert bonds[1, 2, 1, 0] == 1
    assert bonds[1, 0, 0, 1] == 1
    assert bonds[2, 1, 0, 1] == 1
    assert bonds[0, 2, 1, 0] == 4
    assert bonds[2, 0, 0, 1] == 4
