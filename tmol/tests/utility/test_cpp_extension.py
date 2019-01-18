import torch

from tmol.utility.cpp_extension import load, relpaths, modulename


def test_pure_cpp_extension():
    extension = load(modulename(__name__), relpaths(__file__, "pure_cpp_extension.cpp"))

    test = torch.ones(10)

    assert extension.sum(test) == 10
