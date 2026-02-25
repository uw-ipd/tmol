import torch

from tmol._load_ext import ensure_compiled_or_jit

if ensure_compiled_or_jit():
    from tmol.utility.cpp_extension import load, modulename, relpaths, cuda_if_available

    load(
        modulename(__name__),
        cuda_if_available(
            relpaths(
                __file__,
                [
                    "compiled.ops.cpp",
                    "gen_pose_leaf_atoms.cpu.cpp",
                    "gen_pose_leaf_atoms.cuda.cu",
                    "resolve_his_taut.cpu.cpp",
                    "resolve_his_taut.cuda.cu",
                ],
            )
        ),
        is_python_module=False,
    )

    _ops = getattr(torch.ops, modulename(__name__))
else:
    _ops = torch.ops.tmol_io

gen_pose_leaf_atoms = _ops.gen_pose_leaf_atoms
resolve_his_taut = _ops.resolve_his_taut
