"""Exercise native count limits with small tables, never huge rotamer arrays."""

import math

import pytest
import torch

from .test_dunbrack_chi_sampler import get_compiled


def expanded_counts(base, extra, device):
    n_brt = len(base)
    n_chi = len(extra[0]) if extra else 2

    def t(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    counts = t(base)
    offsets = torch.zeros(n_brt, dtype=torch.int32, device=device)
    expansions = torch.zeros_like(counts)
    products = torch.zeros((n_brt, n_chi), dtype=torch.int32, device=device)
    total = get_compiled().count_expanded_rotamers(
        t([n_chi] * n_brt),
        t([[0, -1]] * n_brt).reshape(n_brt, 2),
        t([0]),
        torch.zeros_like(products),
        t(extra).reshape(n_brt, n_chi),
        expansions,
        products,
        counts,
        offsets,
    )
    return total, counts, offsets, expansions, products


@pytest.mark.parametrize(
    "base,extra",
    [
        ([1], [[65536, 65536]]),
        ([2**30], [[3, 0]]),
        ([2**30] * 4, [[0, 0]] * 4),
        ([1], [[2] * 100]),
        ([-1], [[0, 0]]),
        ([1], [[-1, 0]]),
    ],
)
def test_invalid_counts_raise_before_rotamer_allocation(base, extra, torch_device):
    with pytest.raises(RuntimeError, match="Dunbrack sampling count"):
        expanded_counts(base, extra, torch_device)


@pytest.mark.parametrize(
    "base,extra",
    [
        ([2**31 - 2, 1], [[0, 0], [0, 0]]),
        ([2**30 - 1], [[2, 0]]),
        ([0], [[3, 2]]),
        ([2, 4, 0], [[3, 2], [0, 0], [1, 2]]),
        ([], []),
    ],
)
def test_valid_counts_and_offsets_are_exact(base, extra, torch_device):
    total, counts, offsets, expansions, products = expanded_counts(
        base, extra, torch_device
    )
    expected_expansions = [math.prod(x or 1 for x in row) for row in extra]
    expected_counts = [b * e for b, e in zip(base, expected_expansions)]
    assert total == sum(expected_counts)
    assert counts.tolist() == expected_counts
    assert offsets.tolist() == [sum(expected_counts[:i]) for i in range(len(base))]
    assert expansions.tolist() == expected_expansions
    assert products.tolist() == [
        [math.prod(x or 1 for x in row[j + 1 :]) for j in range(len(row))]
        for row in extra
    ]


def test_library_count_narrowing_preserves_an_error_marker(torch_device):
    library = torch.tensor(
        [2**32, 2**31, -1, 7], dtype=torch.int64, device=torch_device
    )
    table = torch.tensor(
        [[0, x] for x in (0, 1, 2, 3, -1)], dtype=torch.int32, device=torch_device
    )
    counts = torch.zeros(5, dtype=torch.int32, device=torch_device)
    get_compiled().determine_n_possible_rots(table, library, counts)
    assert counts.tolist() == [-1, -1, -1, 7, 1]


@pytest.mark.parametrize("library_count,n_brt", [(2**31, 1), (2**30, 2)])
def test_public_sampler_rejects_possible_count_overflow(
    default_database, torch_device, library_count, n_brt
):
    import attr
    from tmol.score.dunbrack import DunbrackParamResolver
    from tmol.pack.rotamer.dunbrack import DunbrackChiSampler

    resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )
    sampling = attr.evolve(
        resolver.sampling_db,
        n_rotamers_for_tableset=torch.full_like(
            resolver.sampling_db.n_rotamers_for_tableset, library_count
        ),
    )
    sampler = DunbrackChiSampler(attr.evolve(resolver, sampling_db=sampling))

    def zeros(shape, dtype=torch.int32):
        return torch.zeros(shape, dtype=dtype, device=torch_device)

    # No coordinate references or large arrays are needed: possible-library
    # counts must be checked before interpolation and expanded-row allocation.
    with pytest.raises(RuntimeError, match="Dunbrack sampling count"):
        sampler.launch_rotamer_building(
            zeros((1, 3), torch.float32),
            zeros(n_brt),
            zeros(n_brt),
            zeros((0, 4)),
            torch.tensor(
                [[i, 0] for i in range(n_brt)], dtype=torch.int32, device=torch_device
            ),
            zeros((n_brt, 1)),
            zeros((n_brt, 1, 1), torch.float32),
            zeros((n_brt, 1)),
            torch.full((n_brt,), 0.98, device=torch_device),
            torch.ones(n_brt, dtype=torch.int32, device=torch_device),
        )


@pytest.mark.parametrize("position", [0, 1024, 2052])
@pytest.mark.parametrize("kind", ["sum", "negative"])
def test_parallel_scan_cannot_recover_from_an_invalid_prefix(
    kind, position, torch_device
):
    base = [1] * 2053
    if kind == "sum":
        # An unchecked int32 sum wraps twice and becomes positive again.
        base[position] = 2**31 - 1
        base[(position + 1) % len(base)] = 2**31 - 1
    else:
        base[position] = -1
    with pytest.raises(RuntimeError, match="Dunbrack sampling count"):
        expanded_counts(base, [[0, 0]] * len(base), torch_device)


def test_valid_multi_block_scan_has_exact_offsets(torch_device):
    base = [i % 7 for i in range(2053)]
    extra = [[0, 3]] * len(base)
    total, counts, offsets, expansions, products = expanded_counts(
        base, extra, torch_device
    )
    expected = [3 * x for x in base]
    assert total == sum(expected)
    assert counts.tolist() == expected
    assert offsets.tolist() == [sum(expected[:i]) for i in range(len(base))]
    assert expansions.tolist() == [3] * len(base)
    assert products.tolist() == [[3, 1]] * len(base)
