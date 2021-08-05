import torch
from tmol.utility.cpp_extension import load, relpaths, modulename, cuda_if_available

load(
    modulename(__name__),
    cuda_if_available(
        relpaths(
            __file__,
            [
                "annealer.cpu.cpp",
                "annealer.cuda.cu",
                "compiled.ops.cpp",
                "compiled.cpu.cpp",
                "compiled.cuda.cu",
            ],
        )
    ),
    is_python_module=False,
)

_ops = getattr(torch.ops, modulename(__name__))

pick_random_rotamers = _ops.pick_random_rotamers
metropolis_accept_reject = _ops.metropolis_accept_reject
create_sim_annealer = _ops.create_sim_annealer
delete_sim_annealer = _ops.delete_sim_annealer
register_standard_random_rotamer_picker = _ops.register_standard_random_rotamer_picker
register_standard_metropolis_accept_or_rejector = (
    _ops.register_standard_metropolis_accept_or_rejector
)
run_sim_annealing = _ops.run_sim_annealing
