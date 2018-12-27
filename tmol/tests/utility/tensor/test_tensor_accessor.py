import pytest
import os.path
import torch

import tmol.utility.cpp_extension as cpp_extension


@pytest.fixture
def tensor_accessor():
    return cpp_extension.load(
        "tensor_accessor",
        [os.path.dirname(__file__) + "/tensor_accessor.cpp"],
        verbose=True,
    )


def test_tensor_accessors(tensor_accessor):

    tvec = torch.arange(30, dtype=torch.float).reshape(10, 3)
    expected = tvec.norm(dim=-1)

    targets = ("aten", "accessor", "eigen", "eigen_squeeze")

    results = {t: getattr(tensor_accessor, t)(tvec) for t in targets}

    for _rn, r in results.items():
        torch.testing.assert_allclose(r, expected)
