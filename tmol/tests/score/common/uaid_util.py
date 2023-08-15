from tmol.utility.cpp_extension import load, modulename, relpaths


# ok, we are going to load the uaid_util.pybind.cc file
def resolve_uaids(
    uaids,
    block_inds,
    pose_inds,
    pose_stack_block_coord_offset,
    pose_stack_block_type,
    pose_stack_inter_block_connections,
    block_type_atom_downstream_of_conn,
):
    compiled = load(modulename(__name__), relpaths(__file__, "uaid_util.pybind.cc"))
    return compiled.resolve_uaids(
        uaids,
        block_inds,
        pose_inds,
        pose_stack_block_coord_offset,
        pose_stack_block_type,
        pose_stack_inter_block_connections,
        block_type_atom_downstream_of_conn,
    )
