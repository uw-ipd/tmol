import pytest
import os.path
import torch

import tmol.utility.cpp_extension as cpp_extension


@pytest.fixture
def vector_magnitude():
    return cpp_extension.load(
        "vector_magnitude",
        [os.path.dirname(__file__) + "/vector_magnitude.cpp"],
        verbose=True,
    )


def test_cpp_extension(vector_magnitude):

    tvec = torch.arange(300, dtype=torch.float).reshape(100, 3)
    expected = tvec.norm(dim=-1)

    results = {"aten": vector_magnitude.aten(tvec)}

    for _rn, r in results.items():
        torch.testing.assert_allclose(r, expected)
