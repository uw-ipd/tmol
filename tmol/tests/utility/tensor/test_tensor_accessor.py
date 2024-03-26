import pytest
import torch
import math

import tmol.utility.cpp_extension as cpp_extension
from tmol.utility.cpp_extension import relpaths, modulename

from tmol.tests.torch import requires_cuda


@pytest.fixture
def tensor_accessor():
    return cpp_extension.load(
        modulename(__name__), relpaths(__file__, "tensor_accessor.cpp"), verbose=True
    )


@pytest.fixture
def accessor_funcs(tensor_accessor):
    targets = ("aten", "accessor", "accessor_arg", "eigen", "eigen_vector_arg")

    return {t: getattr(tensor_accessor, t) for t in targets}


def test_tensor_vector_accessors(accessor_funcs):
    tvec = torch.arange(30, dtype=torch.float).reshape(10, 3)
    expected = tvec.norm(dim=-1)

    results = {t: f(tvec) for t, f in accessor_funcs.items()}

    for _rn, r in results.items():
        torch.testing.assert_close(r, expected)


@pytest.fixture
def matrix_accessor_funcs(tensor_accessor):
    targets = ("eigen_matrix_arg",)

    return {t: getattr(tensor_accessor, t) for t in targets}


def test_tensor_matrix_accessors(matrix_accessor_funcs):
    tvec = torch.arange(90, dtype=torch.float).reshape(10, 3, 3)
    expected = tvec.sum(dim=-1).sum(dim=-1)

    results = {t: f(tvec) for t, f in matrix_accessor_funcs.items()}

    for _rn, r in results.items():
        torch.testing.assert_close(r, expected)


@requires_cuda
def test_tensor_accessor_device_conversion(accessor_funcs):
    """Tensor accessors throw informative errors, rather than segfault, on
    device conversion failures."""

    errors = {
        "aten": None,  # Aten operation supports both devices.
        "accessor": RuntimeError,  # AT_ASSERT failure
        "accessor_arg": TypeError,  # pybind converter
        "eigen": RuntimeError,  # AT_ASSERT failure
        "eigen_vector_arg": TypeError,  # pybind converter
        "eigen_matrix_arg": TypeError,  # pybind converter
    }

    tvec = torch.arange(30, dtype=torch.float, device="cuda").reshape(10, 3)
    expected = tvec.norm(dim=-1)

    for n, f in accessor_funcs.items():
        if errors[n] is not None:
            with pytest.raises(errors[n]):
                f(tvec)
        else:
            torch.testing.assert_close(f(tvec), expected)


def test_tensor_pack_eigen_matrix(tensor_accessor):
    eshape = (2, 5, 3, 3)
    res = tensor_accessor.tensor_pack_construct_eigen_matrix()

    torch.testing.assert_close(res[1], torch.ones(eshape))
    torch.testing.assert_close(res[2], torch.zeros(eshape))
    torch.testing.assert_close(res[3], torch.full(eshape, math.nan), equal_nan=True)


def test_tensor_pack_constructors(tensor_accessor):
    eshape = (2, 5, 3)
    res = tensor_accessor.tensor_pack_construct()

    torch.testing.assert_close(res[1], torch.ones(eshape))
    torch.testing.assert_close(res[2], torch.zeros(eshape))
    torch.testing.assert_close(res[3], torch.full(eshape, math.nan), equal_nan=True)

    t = torch.empty((1, 4))
    eshape = (1, 4, 3)

    # ATen *_like constructors
    res = tensor_accessor.tensor_pack_construct_like_aten(t)

    torch.testing.assert_close(res[1], torch.ones(eshape))
    torch.testing.assert_close(res[2], torch.zeros(eshape))
    torch.testing.assert_close(res[3], torch.full(eshape, math.nan), equal_nan=True)

    with pytest.raises(RuntimeError):
        tensor_accessor.tensor_pack_construct_like_aten(torch.empty(10))

    # TView *_like constructors
    res = tensor_accessor.tensor_pack_construct_like_tview(t)

    torch.testing.assert_close(res[1], torch.ones(eshape))
    torch.testing.assert_close(res[2], torch.zeros(eshape))
    torch.testing.assert_close(res[3], torch.full(eshape, math.nan), equal_nan=True)

    with pytest.raises(TypeError):
        tensor_accessor.tensor_pack_construct_like_tview(torch.empty(10))

    # TPack *_like constructors
    res = tensor_accessor.tensor_pack_construct_like_tpack(t)

    torch.testing.assert_close(res[1], torch.ones(eshape))
    torch.testing.assert_close(res[2], torch.zeros(eshape))
    torch.testing.assert_close(res[3], torch.full(eshape, math.nan), equal_nan=True)

    with pytest.raises(TypeError):
        tensor_accessor.tensor_pack_construct_like_tpack(torch.empty(10))


def test_tview_slice(tensor_accessor):
    sliced = tensor_accessor.tensor_view_take_slice_one()
    gold = torch.tensor([5, 15, 25, 35], dtype=torch.int32)
    torch.testing.assert_close(sliced, gold)
