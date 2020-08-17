import torch
from tmol.utility.cpp_extension import load, relpaths, modulename

load(modulename(__name__), relpaths(__file__, "custom_op.cpp"), is_python_module=False)

cpow = getattr(torch.ops, modulename(__name__)).cpow
