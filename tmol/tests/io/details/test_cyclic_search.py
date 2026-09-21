"""Closure inference cannot cross poses, chains, padding or missing atoms."""

from types import SimpleNamespace

import pytest
import torch

from tmol.io.details._cyclic_search import find_cyclic_closures


def inputs(device):
    co = SimpleNamespace(
        polymer_conn_inds=SimpleNamespace(
            up_atom_for_co_restype=[1, -1], down_atom_for_co_restype=[0, -1]
        )
    )
    chain = torch.tensor(
        [[0, 0, 0, 0, -1], [0, 0, 0, 0, -1]], dtype=torch.int32, device=device
    )
    types = torch.tensor(
        [[0, 0, 0, 1, -1], [0, 0, 0, 1, -1]], dtype=torch.int32, device=device
    )
    coords = torch.full((2, 5, 2, 3), float("nan"), device=device)
    coords[:, 0, 0] = 0
    coords[:, 2, 1] = torch.tensor([1.33, 0, 0], device=device)
    return co, chain, types, coords


def test_closes_each_polymer_before_trailing_ligand(torch_device):
    args = inputs(torch_device)
    result = find_cyclic_closures(*args)
    torch.testing.assert_close(
        result, torch.tensor([[0, 2, 0], [1, 2, 0]], device=torch_device)
    )


@pytest.mark.parametrize("change", ["nan", "far", "different_chain", "single"])
def test_does_not_infer_invalid_closures(change, torch_device):
    co, chain, types, coords = inputs(torch_device)
    if change == "nan":
        coords[:, 2, 1] = float("nan")
    elif change == "far":
        coords[:, 2, 1] = 10
    elif change == "different_chain":
        chain[:, 2] = 1
    else:
        types[:, 1:3] = 1
    assert find_cyclic_closures(co, chain, types, coords).shape == (0, 3)


def test_explicit_rows_survive_disable_and_are_not_duplicated(torch_device):
    args = inputs(torch_device)
    explicit = torch.tensor([[0, 2, 0]], device=torch_device)
    assert find_cyclic_closures(*args, explicit, False) is explicit
    result = find_cyclic_closures(*args, explicit)
    torch.testing.assert_close(
        result, torch.tensor([[0, 2, 0], [1, 2, 0]], device=torch_device)
    )


def test_empty_batch(torch_device):
    co, chain, types, coords = inputs(torch_device)
    assert find_cyclic_closures(
        co, chain[:, :0], types[:, :0], coords[:, :0]
    ).shape == (0, 3)
