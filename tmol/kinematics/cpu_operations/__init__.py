import torch

from tmol.types.torch import Tensor

from . import jit


def iterative_refold(
        hts: Tensor(torch.double)[:, 4, 4],
        parent: Tensor(torch.long)[:],
        inplace: bool = False
):
    if not inplace:
        hts = hts.clone()

    jit.iterative_refold(
        hts.__array__(),
        parent.__array__(),
    )

    return hts


def iterative_f1f2_summation(
        f1f2s: Tensor(torch.double)[:, 6],
        parent: Tensor(torch.long)[:],
        inplace: bool = False
):
    if not inplace:
        f1f2s = f1f2s.clone()

    jit.iterative_f1f2_summation(
        f1f2s.__array__(),
        parent.__array__(),
    )

    return f1f2s
