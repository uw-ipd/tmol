import pytest
import torch

import tmol.utility.cpp_extension as cpp_extension
from tmol.utility.cpp_extension import relpaths, modulename


@pytest.fixture
def tensor_accessor():
    return cpp_extension.load(
        modulename(__name__), relpaths(__file__, "tensor_accessor.cpp"), verbose=True
    )


def test_tensor_accessors(tensor_accessor):

    tvec = torch.arange(30, dtype=torch.float).reshape(10, 3)
    expected = tvec.norm(dim=-1)

    targets = (
        "aten",
        "accessor",
        "accessor_arg",
        "eigen",
        "eigen_squeeze",
        "eigen_arg",
        "eigen_arg_squeeze",
    )

    results = {t: getattr(tensor_accessor, t)(tvec) for t in targets}

    for _rn, r in results.items():
        torch.testing.assert_allclose(r, expected)
