import pytest
import os.path
import torch
import toolz

import tmol.utility.cpp_extension as cpp_extension


@pytest.fixture(scope="session")
def tensor_struct():
    return cpp_extension.load(
        "tensor_struct",
        [os.path.dirname(__file__) + "/tensor_struct.cpp"],
        verbose=True,
    )


def test_tensor_struct(tensor_struct):
    """Struct-of-test design pattern test."""

    tdata = {"a": torch.arange(100), "b": torch.arange(100, 200)}
    asum = tdata["a"].sum()

    # Data structure w/ required tensor fields a, b is converted successfully.
    tensor_struct.sum_a(tdata) == asum

    # Data structure extra fields are ignored.
    tensor_struct.sum_a(toolz.merge(tdata, {"c": torch.full((100,), 2.998e8)})) == asum

    # Data structure w/ missing field causes an a runtime error.
    with pytest.raises(RuntimeError):
        tensor_struct.sum_a({"a": torch.arange(100)})

    # Data structure w/ invalid field type raises error.
    with pytest.raises(RuntimeError):
        tensor_struct.sum_a(toolz.merge(tdata, {"b": tdata["b"].to(torch.float)}))


def test_tensor_view(tensor_struct):
    """Tesor view conversion."""

    dat = torch.arange(100)
    dsum = dat.sum()

    # Tensor of correct type is converted
    tensor_struct.sum(dat) == dsum

    # Incorrect tensor dtype
    with pytest.raises(RuntimeError):
        tensor_struct.sum(dat.to(torch.float))

    # Incorrect tensor shape
    with pytest.raises(RuntimeError):
        tensor_struct.sum(dat[None, :])
