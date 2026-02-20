import torch

from tmol._load_ext import ensure_compiled_or_jit

if ensure_compiled_or_jit():
    from tmol.utility.cpp_extension import load, relpaths, modulename, cuda_if_available

    load(
        modulename(__name__),
        cuda_if_available(
            relpaths(
                __file__,
                [
                    "compiled.ops.cpp",
                    "constraint_score.cpu.cpp",
                    "constraint_score.cuda.cu",
                ],
            )
        ),
        is_python_module=False,
    )

# Ops registered under TORCH_LIBRARY(tmol_constraint, ...) in C++
_ops = torch.ops.tmol_constraint

get_torsion_angle = _ops.get_torsion_angle
