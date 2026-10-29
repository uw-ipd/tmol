from tmol._load_ext import _load_ops

_ops = _load_ops(
    __name__,
    __file__,
    [
        "compiled_ops.cpp",
        "compiled.cpu.cpp",
        "compiled.cuda.cu",
    ],
    "tmol_kin",
)

_forward_kin_op = _ops.forward_kin_op
_forward_only_op = _ops.forward_only_op
_forward_with_ht_op = _ops.forward_with_ht_op
_kin_deriv_op = _ops.kin_deriv_op
_get_kfo_indices_for_atoms = _ops.get_kfo_indices_for_atoms
_get_kfo_atom_parents = _ops.get_kfo_atom_parents
_get_children_op = _ops.get_children
_get_id_and_frame_xyz = _ops.get_id_and_frame_xyz
_calculate_ff_edge_delays = _ops.calculate_ff_edge_delays
_get_jump_atom_indices = _ops.get_jump_atom_indices
_get_block_parent_connectivity_from_toposort = (
    _ops.get_block_parent_connectivity_from_toposort
)
_get_kinforest_scans_from_stencils = _ops.get_kinforest_scans_from_stencils
_get_kinforest_scans_from_stencils2 = _ops.get_kinforest_scans_from_stencils2
_minimizer_map_from_movemap = _ops.minimizer_map_from_movemap
_inverse_kin = _ops.inverse_kin
