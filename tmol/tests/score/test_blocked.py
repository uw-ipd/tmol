import functools

import numpy
import torch

from tmol.tests.benchmark import subfixture
from tmol.tests.torch import requires_cuda


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

    test_coords = structures_bysize[500].tmol_coords

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
        return blocked.cuda.block_interaction_table(cuda_coords, 6.0)

    assert (cuda_itable.cpu()[(cpu_itable > 0)] > 0).all()

    @subfixture(benchmark)
    @bsync
    def cuda_ilist():
        return blocked.cuda.block_interaction_list(cuda_coords, 6.0)

    npairs = int(cuda_ilist[1])
    dense_cuda_ilist = torch.sparse_coo_tensor(
        cuda_ilist[0][:npairs].t(), cuda_ilist[0].new_ones((npairs,)), (nb, nb)
    ).to_dense()
    assert ((cuda_itable > 0).cpu() == (dense_cuda_ilist > 0).cpu()).all()


@requires_cuda
def test_aabb_calc(benchmark, structures_bysize):
    import tmol.score.blocked as blocked

    coords = structures_bysize[500].tmol_coords.cuda()
    bs = 8

    assert coords.shape[0] % bs == 0
    nb = coords.shape[0] // bs

    @subfixture(benchmark)
    @bsync
    def naive():
        minbound = coords.reshape(nb, 8, 3).min(dim=1)[0]
        maxbound = coords.reshape(nb, 8, 3).max(dim=1)[0]
        return torch.cat((minbound, maxbound), dim=-1)

    valid = torch.cat(
        (
            torch.tensor(numpy.nanmin(coords.reshape(nb, 8, 3).cpu(), axis=1)),
            torch.tensor(numpy.nanmax(coords.reshape(nb, 8, 3).cpu(), axis=1)),
        ),
        dim=-1,
    ).cuda()

    ncoords = coords.clone()

    @subfixture(benchmark)
    @bsync
    def shuffle():
        return blocked.cuda.calc_block_aabb(ncoords)

    torch.testing.assert_allclose(coords, ncoords, atol=0, rtol=0)
    torch.testing.assert_allclose(valid, shuffle, atol=0, rtol=0)
