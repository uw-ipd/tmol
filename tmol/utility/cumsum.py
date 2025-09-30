import numpy
import torch

from typing import Optional, Union, Tuple
from tmol.types.functional import validate_args
from tmol.types.array import NDArray
from tmol.types.torch import Tensor


@validate_args
def exclusive_cumsum(inds: NDArray[int][:]) -> NDArray[int][:]:
    """Calculate exclusive cumulative sum over input array"""
    return numpy.concatenate((numpy.zeros((1,), dtype=int), numpy.cumsum(inds)[:-1]))


@validate_args
def exclusive_cumsum1d(
    inds: Union[Tensor[torch.int32][:], Tensor[torch.int64][:]],
) -> Union[Tensor[torch.int32][:], Tensor[torch.int64][:]]:
    return torch.cat(
        (
            torch.tensor([0], dtype=inds.dtype, device=inds.device),
            torch.cumsum(inds, 0, dtype=inds.dtype).narrow(0, 0, inds.shape[0] - 1),
        )
    )


@validate_args
def exclusive_cumsum2d(
    inds: Union[Tensor[torch.int32][:, :], Tensor[torch.int64][:, :]],
) -> Union[Tensor[torch.int32][:, :], Tensor[torch.int64][:, :]]:
    return torch.cat(
        (
            torch.zeros((inds.shape[0], 1), dtype=inds.dtype, device=inds.device),
            torch.cumsum(inds, dim=1, dtype=inds.dtype)[:, :-1],
        ),
        dim=1,
    )


@validate_args
def exclusive_cumsum2d_w_totals(
    inds: Union[Tensor[torch.int32][:, :], Tensor[torch.int64][:, :]],
) -> Union[
    Tuple[Tensor[torch.int32][:, :], Tensor[torch.int32][:]],
    Union[Tensor[torch.int64][:, :], Tensor[torch.int64][:]],
]:
    cumsum = torch.cumsum(inds, dim=1, dtype=inds.dtype)
    return (
        torch.cat(
            (
                torch.zeros((inds.shape[0], 1), dtype=inds.dtype, device=inds.device),
                cumsum[:, :-1],
            ),
            dim=1,
        ),
        cumsum[:, -1],
    )
