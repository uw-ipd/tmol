from tmol._load_ext import load_ops

_ops = load_ops(
    __name__,
    __file__,
    [
        "compiled.ops.cpp",
        "compiled.cpu.cpp",
        "compiled.cuda.cu",
    ],
    "tmol_pack",
)

pack_anneal = _ops.pack_anneal
validate_energies = _ops.validate_energies
build_interaction_graph = _ops.build_interaction_graph
initialize_interaction_graph_topology = _ops.initialize_interaction_graph_topology
note_interaction_graph_topology = _ops.note_interaction_graph_topology
finalize_interaction_graph_topology = _ops.finalize_interaction_graph_topology
accumulate_interaction_graph_entries = _ops.accumulate_interaction_graph_entries
