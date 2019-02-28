import contextlib

from torch.cuda.nvtx import range_push, range_pop


@contextlib.contextmanager
def nvtx_range(name):
    try:
        range_push(name)
        yield
    finally:
        range_pop()
