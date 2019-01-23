import pytest

import torch

from tmol.utility.cpp_extension import load, relpaths, modulename


def test_pure_cpp_extension():
    """Pure cpp extension builds cpu-only interface.

    Pure cpp extension loads with "cpp_extension.load" interface, compiles sum
    function operating over CPU tensors, and raises error on CUDA tensors.
    """
    extension = load(modulename(__name__), relpaths(__file__, "pure_cpp_extension.cpp"))

    assert extension.sum(torch.ones(10)) == 10

    if torch.cuda.is_available():
        with pytest.raises(TypeError):
            extension.sum(torch.ones(10, device="cuda"))
