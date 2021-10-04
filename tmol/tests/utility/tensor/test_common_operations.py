import torch
import numpy

from tmol.utility.tensor.common_operations import (
    exclusive_cumsum1d,
    nplus1d_tensor_from_list,
    cat_differently_sized_tensors,
    join_tensors_and_report_real_entries,
)


def test_exclusive_cumsum():
    t = torch.ones((50,), dtype=torch.long)
    excumsum = exclusive_cumsum1d(t)
    gold = numpy.arange(50, dtype=numpy.int64)
    numpy.testing.assert_equal(excumsum, gold)


def test_nplus1d_tensor_from_list():
    ts = [
        torch.ones([4, 4], dtype=torch.int32),
        2 * torch.ones([3, 4], dtype=torch.int32),
        3 * torch.ones([5, 2], dtype=torch.int32),
        4 * torch.ones([5, 5], dtype=torch.int32),
    ]
    joined, sizes, strides = nplus1d_tensor_from_list(ts)

    gold_sizes = numpy.array([[4, 4], [3, 4], [5, 2], [5, 5]], dtype=numpy.int64)
    numpy.testing.assert_equal(sizes.cpu().numpy(), gold_sizes)
    for i in range(4):
        for j in range(5):
            for k in range(5):
                assert joined[i, j, k] == (
                    (i + 1) if (j < gold_sizes[i, 0] and k < gold_sizes[i, 1]) else 0
                )

    for i in range(4):
        ti = ts[i]
        assert tuple(sizes[i, :]) == ti.shape


def test_cat_diff_sized_tensors_w_same_sizes():
    t1 = torch.full((2, 3, 4), 1, dtype=torch.long)
    t2 = torch.full((3, 3, 4), 2, dtype=torch.long)
    t3 = torch.full((4, 3, 4), 3, dtype=torch.long)

    t, shapes, strides = cat_differently_sized_tensors([t1, t2, t3])

    assert t.shape[0] == 9
    assert t.shape[1] == 3
    assert t.shape[2] == 4

    t2 = torch.cat((t1, t2, t3), dim=0)

    numpy.testing.assert_equal(t.cpu().numpy(), t2.cpu().numpy())

    gold_shapes = numpy.tile([3, 4], (9, 1))
    numpy.testing.assert_equal(shapes.cpu().numpy(), gold_shapes)

    gold_strides = numpy.tile([4, 1], (9, 1))
    numpy.testing.assert_equal(strides.cpu().numpy(), gold_strides)


def test_cat_diff_sized_tensors_w_diff_sizes():

    t1 = torch.full((2, 3, 4), 1, dtype=torch.long)
    t2 = torch.full((3, 3, 3), 2, dtype=torch.long)
    t3 = torch.full((4, 3, 2), 3, dtype=torch.long)

    ts = [t1, t2, t3]

    t, shapes, strides = cat_differently_sized_tensors(ts)

    assert t.shape[0] == 9
    assert t.shape[1] == 3
    assert t.shape[2] == 4

    start = 0
    for ii in range(3):
        ti = ts[ii]
        for jj in range(start, start + ti.shape[0]):
            for kk in range(ti.shape[1]):
                for ll in range(ti.shape[2]):
                    assert ii + 1 == t[jj, kk, ll]
        start += ti.shape[0]


def test_join_tensors_and_report_real_entries(torch_device):
    t1 = torch.full((2, 4, 3), 1, dtype=torch.int32)
    t2 = torch.full((3, 4, 3), 2, dtype=torch.int32)
    t3 = torch.full((4, 4, 3), 3, dtype=torch.int32)
    t4 = torch.full((5, 4, 3), 4, dtype=torch.int32)
    tensors = [t1, t2, t3, t4]

    n_elem, real_elem, joined_elements = join_tensors_and_report_real_entries(tensors)

    n_elem_gold = numpy.array([2, 3, 4, 5], dtype=numpy.int32)
    numpy.testing.assert_equal(n_elem_gold, n_elem.cpu().numpy())

    real_elem_gold = numpy.full((4, 5), False, dtype=bool)
    for i in range(4):
        real_elem_gold[i, : n_elem_gold[i]] = True
    numpy.testing.assert_equal(real_elem_gold, real_elem.cpu().numpy())

    joined_elements_gold = numpy.full((4, 5, 4, 3), -1, dtype=numpy.int32)
    for i in range(4):
        joined_elements_gold[i, : n_elem_gold[i]] = i + 1
    numpy.testing.assert_equal(joined_elements_gold, joined_elements.cpu().numpy())
