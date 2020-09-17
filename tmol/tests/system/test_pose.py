from tmol.system.pose import PackedResidueTypes


def test_load_packed_residue_types(ubq_res, default_database):
    rt_dict = {}
    for res in ubq_res:
        if id(res.residue_type) not in rt_dict:
            rt_dict[id(res.residue_type)] = res.residue_type
    rt_list = [rt for addr, rt in rt_dict.items()]
    prt = PackedResidueTypes.from_restype_list(rt_list, default_database.chemical)
