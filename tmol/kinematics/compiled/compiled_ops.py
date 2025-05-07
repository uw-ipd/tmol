from tmol.utility.cpp_extension import relpaths, TorchOpLoader

sources = relpaths(
    __file__, ["compiled_ops.cpp", "compiled.cpu.cpp", "compiled.cuda.cu"]
)

functions = [
    "forward_kin_op",
    "forward_only_op",
    "get_kfo_indices_for_atoms",
    "get_kfo_atom_parents",
    "get_children",
    "get_id_and_frame_xyz",
    "calculate_ff_edge_delays",
    "get_jump_atom_indices",
    "get_block_parent_connectivity_from_toposort",
    "get_kinforest_scans_from_stencils",
    "get_kinforest_scans_from_stencils2",
    "minimizer_map_from_movemap",
    "inverse_kin",
]

loader = TorchOpLoader(__name__, sources, functions)

forward_kin_op = loader.forward_kin_op
forward_only_op = loader.forward_only_op
get_kfo_indices_for_atoms = loader.get_kfo_indices_for_atoms
get_kfo_atom_parents = loader.get_kfo_atom_parents
get_children = loader.get_children
get_id_and_frame_xyz = loader.get_id_and_frame_xyz
calculate_ff_edge_delays = loader.calculate_ff_edge_delays
get_jump_atom_indices = loader.get_jump_atom_indices
get_block_parent_connectivity_from_toposort = (
    loader.get_block_parent_connectivity_from_toposort
)
get_kinforest_scans_from_stencils = loader.get_kinforest_scans_from_stencils
get_kinforest_scans_from_stencils2 = loader.get_kinforest_scans_from_stencils2
minimizer_map_from_movemap = loader.minimizer_map_from_movemap
inverse_kin = loader.inverse_kin
