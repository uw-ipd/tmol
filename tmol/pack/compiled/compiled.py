import torch

from tmol._load_ext import ensure_compiled_or_jit

if ensure_compiled_or_jit():
    from tmol.utility.cpp_extension import load, relpaths, modulename, cuda_if_available

    load(
        modulename(__name__),
        cuda_if_available(
            relpaths(__file__, ["compiled.ops.cpp", "compiled.cpu.cpp", "compiled.cuda.cu"])
        ),
        is_python_module=False,
    )

    _ops = getattr(torch.ops, modulename(__name__))
else:
    _ops = torch.ops.tmol_pack

pack_anneal = _ops.pack_anneal
validate_energies = _ops.validate_energies
build_interaction_graph = _ops.build_interaction_graph
