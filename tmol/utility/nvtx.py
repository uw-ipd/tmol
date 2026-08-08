import contextlib

import torch

try:
    from torch.cuda.nvtx import range_push as _nvtx_push, range_pop as _nvtx_pop
    _NVTX_AVAILABLE = True
except ImportError:
    _NVTX_AVAILABLE = False


@contextlib.contextmanager
def nvtx_range(name):
    """Context manager that annotates a code region with an NVTX range marker.

    Active only when CUDA is available (and thus NVTX is present).
    On CPU-only or MPS builds this is a zero-cost no-op.
    """
    if _NVTX_AVAILABLE and torch.cuda.is_available():
        try:
            _nvtx_push(name)
            yield
        finally:
            _nvtx_pop()
    else:
        yield
