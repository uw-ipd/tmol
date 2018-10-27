import functools

from tmol.tests.benchmark import subfixture
from tmol.tests.torch import requires_cuda
import torch


def bsync(f):
    @functools.wraps(f)
    def syncf():
        try:
            return f()
        finally:
            torch.cuda.synchronize()

    return syncf


@requires_cuda
def test_blocked_eval(benchmark, structures_bysize):
    import tmol.score.blocked as blocked

    test_coords = structures_bysize[200].tmol_coords

    bs = 8
    assert test_coords.shape[0] % bs == 0
    nb = test_coords.shape[0] // bs

    atom_interaction_table = blocked.cpu.coord_interaction_table(test_coords, 6.0)
    assert (atom_interaction_table).sum() > 0

    cpu_itable = (
        atom_interaction_table.reshape((nb, bs, nb, bs)).sum(dim=-1).sum(dim=-2)
    )

    assert (cpu_itable > 1).sum() > 0

    cuda_coords = test_coords.cuda()

    @benchmark
    @bsync
    def cuda_itable():
        return blocked.cuda.block_interaction_table(cuda_coords.cuda(), 6.0)

    assert ((cpu_itable > 1).cpu() == (cuda_itable > 1).cpu()).all()
