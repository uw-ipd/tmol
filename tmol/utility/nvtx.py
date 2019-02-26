import contextlib

from torch.cuda.nvtx import range_push, range_pop


@contextlib.contextmanager
def range_ctx(name):
    try:
        range_push(name)
        yield
    finally:
        range_pop()
