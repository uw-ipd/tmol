import torch

from tmol._load_ext import ensure_compiled_or_jit

# custom_op.cpp uses TORCH_EXTENSION_NAME as the ops namespace, so the JIT and
# AOT namespaces differ — load_ops() can't be used with a single ops_name here.
if ensure_compiled_or_jit():
    from tmol.utility.cpp_extension import load, relpaths, modulename

    _name = modulename(__name__)
    load(_name, relpaths(__file__, "custom_op.cpp"), is_python_module=False)
    _ops = getattr(torch.ops, _name)
else:
    import os
    import glob

    _ext_dir = os.path.dirname(__file__)
    _so_files = glob.glob(os.path.join(_ext_dir, "_custom_op*.so"))
    if _so_files:
        torch.ops.load_library(_so_files[0])
    _ops = torch.ops._custom_op

cpow = _ops.cpow
