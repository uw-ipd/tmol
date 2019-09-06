import torch
import numpy

import tmol.score.common.stack_condense as sc


def test_condense_numpy_inds():

    vals = numpy.array(
        [[0, 1, -1, 3, 4], [-1, 1, 2, -1, -1], [0, -1, 2, 3, -1]], dtype=int
    )
    selection = vals != -1
    condensed_inds = sc.condense_numpy_inds(selection)

    expected = numpy.array([[0, 1, 3, 4], [1, 2, -1, -1], [0, 2, 3, -1]], dtype=int)
    numpy.testing.assert_equal(condensed_inds, expected)


def test_condense_torch_inds(torch_device):

    vals = torch.tensor(
        [[0, 1, -1, 3, 4], [-1, 1, 2, -1, -1], [0, -1, 2, 3, -1]],
        dtype=torch.int64,
        device=torch_device,
    )
    selection = vals != -1
    condensed_inds = sc.condense_torch_inds(selection, device=torch_device)

    expected = torch.tensor(
        [[0, 1, 3, 4], [1, 2, -1, -1], [0, 2, 3, -1]],
        dtype=torch.int64,
        device=torch_device,
    )
    torch.testing.assert_allclose(condensed_inds, expected)
    assert condensed_inds.device == torch_device


def test_take_values_w_sentineled_index1(torch_device):
    values = 2 * torch.arange(10, dtype=torch.int32, device=torch_device)
    index = torch.tensor(
        [[5, 4, 3, 2, -1], [9, 8, -1, 7, 6]], dtype=torch.int64, device=torch_device
    )
    index_values = sc.take_values_w_sentineled_index(values, index)
    expected = torch.tensor(
        [[10, 8, 6, 4, -1], [18, 16, -1, 14, 12]],
        dtype=torch.int32,
        device=torch_device,
    )
    torch.testing.assert_allclose(index_values, expected)
    assert index_values.dtype == torch.int32
    assert index_values.device == torch_device


def test_take_values_w_sentineled_index_and_dest(torch_device):
    vals = 2 * torch.arange(10, dtype=torch.int32, device=torch_device)
    index = torch.tensor(
        [[1, 0, 3, 4, -1], [3, 4, 2, -1, -1], [8, -1, -1, -1, 5]],
        dtype=torch.int64,
        device=torch_device,
    )
    dest = torch.tensor(
        [[1, 1, 1, 1], [1, 1, 1, -1], [1, 1, -1, -1]],
        dtype=torch.int32,
        device=torch_device,
    )
    index_values = sc.take_values_w_sentineled_index_and_dest(vals, index, dest)

    expected = torch.tensor(
        [[2, 0, 6, 8], [6, 8, 4, -1], [16, 10, -1, -1]],
        dtype=torch.int32,
        device=torch_device,
    )

    torch.testing.assert_allclose(index_values, expected)


def test_condense_subset(torch_device):
    vals = 2 * torch.arange(30, dtype=torch.int32, device=torch_device).view(2, 5, 3)
    vals_to_keep = (
        torch.tensor(
            [[1, 1, 0, 1, 1], [0, 1, 1, 0, 1]], dtype=torch.int32, device=torch_device
        )
        != 0
    )
    expected = torch.tensor(
        [
            [[0, 2, 4], [6, 8, 10], [18, 20, 22], [24, 26, 28]],
            [[36, 38, 40], [42, 44, 46], [54, 56, 58], [-1, -1, -1]],
        ],
        dtype=torch.int32,
        device=torch_device,
    )

    subset = sc.condense_subset(vals, vals_to_keep)
    torch.testing.assert_allclose(subset, expected)
