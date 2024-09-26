import torch
from tmol.utility.cpp_extension import load, relpaths, modulename, cuda_if_available

load(
    modulename(__name__),
    cuda_if_available(
        relpaths(__file__, ["compiled_ops.cpp", "compiled.cpu.cpp", "compiled.cuda.cu"])
    ),
    is_python_module=False,
)

_ops = getattr(torch.ops, modulename(__name__))
forward_kin_op = _ops.forward_kin_op
forward_only_op = _ops.forward_only_op
# fix_jump_nodes_op = _ops.fix_jump_nodes_op
get_kfo_indices_for_atoms = _ops.get_kfo_indices_for_atoms
get_kfo_atom_parents = _ops.get_kfo_atom_parents
get_children = _ops.get_children
get_id_and_frame_xyz = _ops.get_id_and_frame_xyz
calculate_ff_edge_delays = _ops.calculate_ff_edge_delays
