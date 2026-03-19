from tmol._load_ext import load_ops

_ops = load_ops(
    __name__,
    __file__,
    [
        "compiled_ops.cpp",
        "compiled.cpu.cpp",
        "compiled.cuda.cu",
    ],
    "tmol_kin",
)

forward_kin_op = _ops.forward_kin_op
forward_only_op = _ops.forward_only_op
get_kfo_indices_for_atoms = _ops.get_kfo_indices_for_atoms
get_kfo_atom_parents = _ops.get_kfo_atom_parents
get_children = _ops.get_children
get_id_and_frame_xyz = _ops.get_id_and_frame_xyz
calculate_ff_edge_delays = _ops.calculate_ff_edge_delays
get_jump_atom_indices = _ops.get_jump_atom_indices
get_block_parent_connectivity_from_toposort = (
    _ops.get_block_parent_connectivity_from_toposort
)
get_kinforest_scans_from_stencils = _ops.get_kinforest_scans_from_stencils
get_kinforest_scans_from_stencils2 = _ops.get_kinforest_scans_from_stencils2
minimizer_map_from_movemap = _ops.minimizer_map_from_movemap
