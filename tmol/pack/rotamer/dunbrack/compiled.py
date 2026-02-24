import torch

from tmol._load_ext import ensure_compiled_or_jit

if ensure_compiled_or_jit():
    from tmol.utility.cpp_extension import cuda_if_available, load, modulename, relpaths

    load(
        modulename(__name__),
        cuda_if_available(relpaths(__file__, ["compiled.ops.cpp", "compiled.cpu.cpp", "compiled.cuda.cu"])),
        is_python_module=False,
    )

# Ops registered under TORCH_LIBRARY(tmol_dun_sampler, ...) in C++
_ops = torch.ops.tmol_dun_sampler

dun_sample_chi = _ops.dun_sample_chi
