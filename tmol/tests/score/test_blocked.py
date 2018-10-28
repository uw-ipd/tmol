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

    npairs = int((cpu_itable > 0).sum())
    assert npairs > 0

    cuda_coords = test_coords.cuda()

    @subfixture(benchmark)
    @bsync
    def cuda_itable():
        for _ in range(10):
            r = blocked.cuda.block_interaction_table(cuda_coords, 6.0)
        return r

    @subfixture(benchmark)
    @bsync
    def cuda_ilist():
        for _ in range(10):
            r = blocked.cuda.block_interaction_list(cuda_coords, 6.0)
        return r

    assert ((cpu_itable > 0).cpu() == (cuda_itable > 0).cpu()).all()
    assert int(cuda_ilist[1].sum()) == npairs

    dense_cuda_ilist = torch.sparse_coo_tensor(
        cuda_ilist[0][:npairs].t(), cuda_ilist[0].new_ones((npairs,)), (nb, nb)
    ).to_dense()
    assert ((cpu_itable > 0).cpu() == (dense_cuda_ilist > 0).cpu()).all()
