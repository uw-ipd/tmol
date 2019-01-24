import pytest
import torch

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
    targets = (
        "aten",
        "accessor",
        "accessor_arg",
        "eigen",
        "eigen_squeeze",
        "eigen_arg",
        "eigen_arg_squeeze",
    )

    return {t: getattr(tensor_accessor, t) for t in targets}


def test_tensor_accessors(accessor_funcs):

    tvec = torch.arange(30, dtype=torch.float).reshape(10, 3)
    expected = tvec.norm(dim=-1)

    results = {t: f(tvec) for t, f in accessor_funcs.items()}

    for _rn, r in results.items():
        torch.testing.assert_allclose(r, expected)


@requires_cuda
def test_tensor_accessor_device_conversion(accessor_funcs):
    """Tensor accessors throw informative errors, rather than segfault, on
    device conversion failures."""

    errors = {
        "aten": None,  # Aten operation supports both devices.
        "accessor": RuntimeError,  # AT_ASSERT failure
        "accessor_arg": TypeError,  # pybind converter
        "eigen": RuntimeError,  # AT_ASSERT failure
        "eigen_squeeze": RuntimeError,  # AT_ASSERT failure
        "eigen_arg": TypeError,  # pybind converter
        "eigen_arg_squeeze": TypeError,  # pybind converter
    }

    tvec = torch.arange(30, dtype=torch.float, device="cuda").reshape(10, 3)
    expected = tvec.norm(dim=-1)

    for n, f in accessor_funcs.items():
        if errors[n] is not None:
            with pytest.raises(errors[n]):
                f(tvec)
        else:
            torch.testing.assert_allclose(f(tvec), expected)
