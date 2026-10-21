from tmol._load_ext import _load_ops

_ops = _load_ops(
    __name__,
    __file__,
    [
        "compiled.ops.cpp",
        "compiled.cpu.cpp",
        "compiled.cuda.cu",
    ],
    "tmol_dun_sampler",
)

dun_sample_chi = _ops.dun_sample_chi
