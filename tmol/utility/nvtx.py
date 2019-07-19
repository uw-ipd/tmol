import contextlib

from torch.cuda.nvtx import range_push, range_pop
import torch


@contextlib.contextmanager
def nvtx_range(name):
    if torch.cuda.is_available():
        try:
            range_push(name)
            yield
        finally:
            range_pop()
    else:
        yield
