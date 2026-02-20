import os
import glob

import torch

# Load the pre-compiled TORCH_LIBRARY extension.
# This is NOT a Python module (no PyInit_) — it registers ops via TORCH_LIBRARY.
# Use load_library() instead of import.
_ext_dir = os.path.dirname(__file__)
_so_files = glob.glob(os.path.join(_ext_dir, "_custom_op*.so"))
if _so_files:
    torch.ops.load_library(_so_files[0])

cpow = torch.ops._custom_op.cpow
