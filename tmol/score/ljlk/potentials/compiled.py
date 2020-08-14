import torch
from tmol.utility.cpp_extension import load, relpaths, modulename, cuda_if_available


load(
    modulename(__name__),
    cuda_if_available(
        relpaths(
            __file__,
            [
                "compiled.ops.cpp",
                "lj.compiled.cpu.cpp",
                "lj.compiled.cuda.cu",
                "lk_isotropic.compiled.cpu.cpp",
                "lk_isotropic.compiled.cuda.cu",
            ],
        )
    ),
    is_python_module=False,
)

_ops = getattr(torch.ops, modulename(__name__))

score_ljlk_lj = _ops.score_ljlk_lj
score_ljlk_lj_triu = _ops.score_ljlk_lj_triu
score_ljlk_lk_isotropic = _ops.score_ljlk_lk_isotropic
score_ljlk_lk_isotropic_triu = _ops.score_ljlk_lk_isotropic_triu
