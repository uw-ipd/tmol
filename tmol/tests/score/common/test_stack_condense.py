import pytest
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


def test_condense_numpy_inds_from_doc_string():
    input = numpy.array([[0, 1, 0, 1], [1, 1, 0, 1]], dtype=int) == 1
    expected_output = numpy.array([[1, 3, -1], [0, 1, 3]], dtype=int)
    actual_output = sc.condense_numpy_inds(input)
    numpy.testing.assert_equal(actual_output, expected_output)


def test_condense_torch_inds_from_doc_string():
    input = torch.tensor([[0, 1, 0, 1], [1, 1, 0, 1]], dtype=torch.int32) == 1
    expected_output = torch.tensor([[1, 3, -1], [0, 1, 3]], dtype=torch.int64)
    actual_output = sc.condense_torch_inds(input, torch.device("cpu"))
    torch.testing.assert_allclose(actual_output, expected_output)


def test_take_values_w_sentineled_index_from_doc_string():
    values = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32)
    sentineled_index_tensor = torch.tensor(
        [[2, 1, 2, 5, -1], [1, 4, 1, 5, 2]], dtype=torch.int64
    )
    expected_output = torch.tensor(
        [[12, 11, 12, 15, -1], [11, 14, 11, 15, 12]], dtype=torch.int32
    )
    actual_output = sc.take_values_w_sentineled_index(values, sentineled_index_tensor)
    torch.testing.assert_allclose(actual_output, expected_output)


def test_take_values_w_sentineled_index_and_dest_from_doc_string():
    values = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.float)
    sentineled_index_tensor = torch.tensor(
        [[2, -1, 2, 5, -1], [1, 4, -1, 5, 2]], dtype=torch.int64
    )

    sentineled_dest_tensor = torch.tensor(
        [[1, 1, 1, -1], [1, 1, 1, 1]], dtype=torch.int32
    )

    expected_output = torch.tensor(
        [[12, 12, 15, -1], [11, 14, 15, 12]], dtype=torch.float
    )

    actual_output = sc.take_values_w_sentineled_index_and_dest(
        values, sentineled_index_tensor, sentineled_dest_tensor
    )
    torch.testing.assert_allclose(actual_output, expected_output)


def test_take_values_w_sentineled_dest_from_doc_string():
    values = torch.tensor(
        [[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]], dtype=torch.int32
    )
    values_to_take = (
        torch.tensor([[1, 0, 1, 1, 0], [1, 1, 0, 1, 1]], dtype=torch.int32) == 1
    )

    sentineled_dest_tensor = torch.tensor(
        [[1, 1, 1, -1], [1, 1, 1, 1]], dtype=torch.int32
    )

    expected_output = torch.tensor(
        [[10, 12, 13, -1], [20, 21, 23, 24]], dtype=torch.int32
    )

    actual_output = sc.take_values_w_sentineled_dest(
        values, values_to_take, sentineled_dest_tensor
    )
    torch.testing.assert_allclose(actual_output, expected_output)


def test_condense_subset_from_doc_string():
    values = torch.tensor(
        [
            [[10, 10], [11, 11], [12, 12], [13, 13], [14, 14]],
            [[20, 20], [21, 21], [22, 22], [23, 23], [24, 24]],
        ],
        dtype=torch.int32,
    )
    values_to_keep = (
        torch.tensor([[1, 0, 1, 1, 0], [1, 1, 0, 1, 1]], dtype=torch.int32) == 1
    )

    expected_output = torch.tensor(
        [
            [[10, 10], [12, 12], [13, 13], [-1, -1]],
            [[20, 20], [21, 21], [23, 23], [24, 24]],
        ],
        dtype=torch.int32,
    )
    actual_output = sc.condense_subset(values, values_to_keep)
    torch.testing.assert_allclose(actual_output, expected_output)


def test_take_condensed_3d_subset_from_doc_string():
    values = torch.tensor(
        [
            [[10, 10], [11, 11], [12, 12], [13, 13], [14, 14]],
            [[20, 20], [21, 21], [22, 22], [23, 23], [24, 24]],
        ],
        dtype=torch.int32,
    )
    condensed_inds_to_keep = torch.tensor(
        [[0, -1, 2, 3], [4, 3, 2, 4]], dtype=torch.int64
    )
    condensed_dest_tensor = torch.tensor(
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [1, 3]], dtype=torch.int64
    )

    expected_output = torch.tensor(
        [
            [[10, 10], [12, 12], [13, 13], [-1, -1]],
            [[24, 24], [23, 23], [22, 22], [24, 24]],
        ],
        dtype=torch.int32,
    )
    actual_output = sc.take_condensed_3d_subset(
        values, condensed_inds_to_keep, condensed_dest_tensor
    )
    torch.testing.assert_allclose(actual_output, expected_output)


@pytest.mark.parametrize("torch_dtype", [torch.int32, torch.int64])
def test_tile_subset_indices_torch(torch_device, torch_dtype):
    heavy_inds = torch.tensor(
        [0, 1, 2, 3, 4, 5, 8, 10, 11, 12, 13], dtype=torch_dtype, device=torch_device
    )
    heavy_subset_wi_tile, n_in_tile = sc.tile_subset_indices(heavy_inds, 8)
    heavy_inds = heavy_inds.cpu()
    assert heavy_subset_wi_tile.device == torch_device
    assert n_in_tile.device == torch_device
    assert heavy_subset_wi_tile.dtype == torch_dtype
    assert n_in_tile.dtype == torch_dtype

    heavy_subset_wi_tile = heavy_subset_wi_tile.cpu().numpy()
    n_in_tile = n_in_tile.cpu().numpy()
    gold_subset_wi_tile = numpy.array(
        [0, 1, 2, 3, 4, 5, -1, -1, 0, 2, 3, 4, 5, -1, -1, -1], dtype=numpy.int64
    )
    gold_n_in_tile = numpy.array([6, 5], dtype=numpy.int64)
    numpy.testing.assert_equal(gold_subset_wi_tile, heavy_subset_wi_tile)
    numpy.testing.assert_equal(gold_n_in_tile, n_in_tile)


def test_tile_subset_indices_numpy():
    heavy_inds = numpy.array([0, 1, 2, 3, 4, 5, 8, 10, 11, 12, 13], dtype=numpy.int32)
    heavy_subset_wi_tile, n_in_tile = sc.tile_subset_indices(heavy_inds, 8)
    heavy_inds = heavy_inds

    gold_subset_wi_tile = numpy.array(
        [0, 1, 2, 3, 4, 5, -1, -1, 0, 2, 3, 4, 5, -1, -1, -1], dtype=numpy.int64
    )
    gold_n_in_tile = numpy.array([6, 5], dtype=numpy.int64)
    numpy.testing.assert_equal(gold_subset_wi_tile, heavy_subset_wi_tile)
    numpy.testing.assert_equal(gold_n_in_tile, n_in_tile)
