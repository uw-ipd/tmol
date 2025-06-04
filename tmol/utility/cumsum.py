import numpy
import torch

from typing import Optional
from tmol.types.functional import validate_args
from tmol.types.array import NDArray
from tmol.types.torch import Tensor


@validate_args
def exclusive_cumsum(inds: NDArray[int][:]) -> NDArray[int][:]:
    """Calculate exclusive cumulative sum over input array"""
    return numpy.concatenate((numpy.zeros((1,), dtype=int), numpy.cumsum(inds)[:-1]))


@validate_args
def exclusive_cumsum1d(inds: Tensor[torch.int32][:]) -> Tensor[torch.int32][:]:
    return torch.cat(
        (
            torch.tensor([0], dtype=torch.int32, device=inds.device),
            torch.cumsum(inds, 0, dtype=torch.int32).narrow(0, 0, inds.shape[0] - 1),
        )
    )


@validate_args
def exclusive_cumsum2d(inds: Tensor[torch.int32][:, :]) -> Tensor[torch.int32][:, :]:
    return torch.cat(
        (
            torch.zeros((inds.shape[0], 1), dtype=torch.int32, device=inds.device),
            torch.cumsum(inds, dim=1, dtype=torch.int32)[:, :-1],
        ),
        dim=1,
    )
