import pytest
import torch

from tmol.tests import requires_cuda


def independent_minimum_tables(counts, chunk_size, device):
    """A chain with a unique all-zero minimum and a fully frozen first pose."""
    n_res = len(counts)
    n_rots = sum(counts)
    counts_t = torch.tensor(counts, dtype=torch.int32)
    n_for_res = torch.zeros((2, n_res), dtype=torch.int32)
    n_for_res[1] = counts_t
    oneb_offsets = torch.zeros_like(n_for_res)
    oneb_offsets[1, 1:] = counts_t.cumsum(0)[:-1]
    res_for_rot = torch.repeat_interleave(
        torch.arange(n_res, dtype=torch.int32), counts_t
    )
    energy1b = torch.cat([torch.arange(n, dtype=torch.float32) for n in counts])
    chunk_offset_offsets = torch.full((2, n_res, n_res), -1, dtype=torch.int64)
    chunk_offsets = []
    energies = []
    for i in range(n_res - 1):
        for a, b in ((i, i + 1), (i + 1, i)):
            chunk_offset_offsets[1, a, b] = len(chunk_offsets)
            for first in range(0, counts[a], chunk_size):
                for second in range(0, counts[b], chunk_size):
                    chunk_offsets.append(len(energies))
                    rows = torch.arange(first, min(first + chunk_size, counts[a]))
                    cols = torch.arange(second, min(second + chunk_size, counts[b]))
                    energies.extend((rows[:, None] + cols[None, :]).flatten().tolist())
    values = (
        n_rots,
        torch.tensor([0, n_res], dtype=torch.int32),
        torch.tensor([0, n_rots], dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
        n_for_res,
        oneb_offsets,
        res_for_rot,
        chunk_size,
        chunk_offset_offsets,
        torch.tensor(chunk_offsets, dtype=torch.int64),
        energy1b,
        torch.tensor(energies, dtype=torch.float32),
    )
    return tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in values)


@requires_cuda
@pytest.mark.parametrize(
    "chunk_size,n_res",
    [(7, 3), (16, 1), (16, 2), (16, 128), (16, 129), (32, 3)],
)
def test_cuda_annealer_assignment_cache_boundaries(chunk_size, n_res):
    from tmol.pack.compiled import pack_anneal

    counts = [2] * n_res
    if n_res <= 2:
        counts = [1] * n_res
    elif n_res == 3:
        counts = [35, 36, 17]
    inputs = independent_minimum_tables(counts, chunk_size, torch.device("cuda"))
    torch.manual_seed(20260911)
    scores, assignments = pack_anneal(*inputs)
    assert scores.shape == (2, 312)
    assert assignments.shape == (2, 312, n_res)
    # Every pair contribution and one-body term has its unique minimum at zero.
    # Full quenching must find it for every trajectory, including the fallback
    # above the shared-cache capacity and dynamic chunk sizes.
    assert torch.count_nonzero(scores).item() == 0
    assert torch.count_nonzero(assignments).item() == 0
