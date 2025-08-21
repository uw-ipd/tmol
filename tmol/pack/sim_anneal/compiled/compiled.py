from tmol.utility.cpp_extension import relpaths, TorchOpLoader

sources = relpaths(
    __file__,
    [
        "annealer.cpu.cpp",
        "annealer.cuda.cu",
        "compiled.ops.cpp",
        "compiled.cpu.cpp",
        "compiled.cuda.cu",
    ],
)

functions = [
    "pick_random_rotamers",
    "metropolis_accept_reject",
    "create_sim_annealer",
    "delete_sim_annealer",
    "register_standard_random_rotamer_picker",
    "register_standard_metropolis_accept_or_rejector",
    "run_sim_annealing",
]

loader = TorchOpLoader(__name__, sources, functions)

pick_random_rotamers = loader.pick_random_rotamers
metropolis_accept_reject = loader.metropolis_accept_reject
create_sim_annealer = loader.create_sim_annealer
delete_sim_annealer = loader.delete_sim_annealer
register_standard_random_rotamer_picker = loader.register_standard_random_rotamer_picker
register_standard_metropolis_accept_or_rejector = (
    loader.register_standard_metropolis_accept_or_rejector
)
run_sim_annealing = loader.run_sim_annealing
