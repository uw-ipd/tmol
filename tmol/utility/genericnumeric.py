from functools import singledispatch

import numpy
import torch

from torch.autograd import Variable

@singledispatch
def exp(x, out=None):
    """The exponential of the elements of :attr:`x`.

    .. math::
        y_{i} = e^{x_{i}}

    Args:
    x (Tensor): the input tensor
    out (Tensor, optional): the output tensor
    """
    return numpy.exp(x, out=out)

exp.register(Variable)(torch.exp)
