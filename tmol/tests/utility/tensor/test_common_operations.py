import torch
import numpy
import pytest

from tmol.tests import requires_cuda
from tmol.utility.tensor import (
    stretch,
    stretch2,
    exclusive_cumsum1d,
    nplus1d_tensor_from_list,
    cat_differently_sized_tensors,
    join_tensors_and_report_real_entries,
    invert_mapping,
    print_row_numbered_tensor,
)


def test_stretch_i32(torch_device):
    t = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=torch_device)
    t2 = stretch(t, 3)
    t2_gold = torch.tensor(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=torch.int32, device=torch_device
    )
    torch.testing.assert_close(t2_gold, t2)
    assert t2.device == torch_device


def test_stretch2_i32(torch_device):
    t = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32, device=torch_device
    )
    t2 = stretch2(t, 3)
    t2_gold = torch.tensor(
        [[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], [4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7]],
        dtype=torch.int32,
        device=torch_device,
    )
    torch.testing.assert_close(t2_gold, t2)
    assert t2.device == torch_device


def test_stretch_accepts_scalar_tensor_count(torch_device):
    count = torch.tensor(2, dtype=torch.int32, device=torch_device)
    one_dim = torch.tensor([1, 2], dtype=torch.int32, device=torch_device)
    two_dim = one_dim.reshape(1, 2)

    torch.testing.assert_close(
        stretch(one_dim, count),
        torch.tensor([1, 1, 2, 2], dtype=torch.int32, device=torch_device),
    )
    torch.testing.assert_close(
        stretch2(two_dim, count),
        torch.tensor([[1, 1, 2, 2]], dtype=torch.int32, device=torch_device),
    )


def test_exclusive_cumsum():
    t = torch.ones((50,), dtype=torch.long)
    excumsum = exclusive_cumsum1d(t)
    gold = numpy.arange(50, dtype=numpy.int64)
    numpy.testing.assert_equal(excumsum, gold)


@pytest.mark.parametrize(
    ("values", "first_row"),
    (([5, 6], "tensor([[0, 5]"), ([[5, 6], [7, 8]], "tensor([[0, 5, 6]")),
)
def test_print_row_numbered_tensor(values, first_row, torch_device, capsys):
    tensor = torch.tensor(values, dtype=torch.int32, device=torch_device)

    print_row_numbered_tensor(tensor)

    output = capsys.readouterr().out
    assert first_row in output


def test_tensor_sequence_validation():
    one_dim = torch.zeros((1,), dtype=torch.float32)
    two_dim = torch.zeros((1, 1), dtype=torch.float32)

    with pytest.raises(ValueError, match="one- or two-dimensional"):
        print_row_numbered_tensor(torch.zeros((1, 1, 1)))
    with pytest.raises(ValueError, match="at least one tensor"):
        nplus1d_tensor_from_list([])
    with pytest.raises(ValueError, match="same number of dimensions"):
        nplus1d_tensor_from_list([one_dim, two_dim])
    with pytest.raises(ValueError, match="same dtype"):
        cat_differently_sized_tensors([one_dim, one_dim.to(torch.float64)])
    with pytest.raises(ValueError, match="same shape after dimension zero"):
        join_tensors_and_report_real_entries([torch.zeros((1, 2)), torch.zeros((2, 3))])


@requires_cuda
def test_tensor_sequence_validation_rejects_mixed_devices():
    cpu = torch.zeros((1, 2))
    cuda = cpu.to("cuda")

    with pytest.raises(ValueError, match="same device"):
        cat_differently_sized_tensors([cpu, cuda])


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
    t1 = torch.full((2, 4, 3), 1, dtype=torch.int32, device=torch_device)
    t2 = torch.full((3, 4, 3), 2, dtype=torch.int32, device=torch_device)
    t3 = torch.full((4, 4, 3), 3, dtype=torch.int32, device=torch_device)
    t4 = torch.full((5, 4, 3), 4, dtype=torch.int32, device=torch_device)
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


def test_invert_mapping(torch_device):
    a_2_b = torch.tensor([5, 4, 7, 1, 2, 0], dtype=torch.int32, device=torch_device)
    b_2_a = invert_mapping(a_2_b, 8)

    assert b_2_a.dtype == torch.int32
    assert b_2_a.device == torch_device
    b_2_a_gold = numpy.array([5, 3, 4, -1, 1, 0, -1, 2], dtype=numpy.int32)
    numpy.testing.assert_equal(b_2_a_gold, b_2_a.cpu().numpy())


def test_invert_mapping_infers_output_size(torch_device):
    a_2_b = torch.tensor([2, 0], dtype=torch.int64, device=torch_device)

    b_2_a = invert_mapping(a_2_b)

    torch.testing.assert_close(
        b_2_a, torch.tensor([1, -1, 0], dtype=torch.int64, device=torch_device)
    )
    assert invert_mapping.__doc__ is not None
