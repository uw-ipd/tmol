import torch
from typing import List, Union
from tmol.types.torch import Tensor
from tmol.types.functional import validate_args


@validate_args
def exclusive_cumsum1d(
    inds: Union[Tensor(torch.int32)[:], Tensor(torch.int64)[:]]
) -> Union[Tensor(torch.int32)[:], Tensor(torch.int64)[:]]:
    return torch.cat(
        (
            torch.tensor([0], dtype=inds.dtype, device=inds.device),
            torch.cumsum(inds, 0, dtype=inds.dtype).narrow(0, 0, inds.shape[0] - 1),
        )
    )


@validate_args
def exclusive_cumsum2d(
    inds: Union[Tensor(torch.int32)[:, :], Tensor(torch.int64)[:, :]]
) -> Union[Tensor(torch.int32)[:, :], Tensor(torch.int64)[:, :]]:
    return torch.cat(
        (
            torch.zeros((inds.shape[0], 1), dtype=torch.int32, device=inds.device),
            torch.cumsum(inds, dim=1, dtype=torch.int32)[:, :-1],
        ),
        dim=1,
    )


def print_row_numbered_tensor(tensor):
    if len(tensor.shape) == 1:
        print(
            torch.cat(
                (
                    torch.arange(tensor.shape[0], dtype=tensor.dtype).reshape(-1, 1),
                    tensor.reshape(-1, 1),
                ),
                1,
            )
        )
    else:
        print(
            torch.cat(
                (
                    torch.arange(tensor.shape[0], dtype=tensor.dtype).reshape(-1, 1),
                    tensor,
                ),
                1,
            )
        )


# @validate_args
def nplus1d_tensor_from_list(
    tensors: List
):  # -> Tuple[Tensor, Tensor(torch.long)[:,:], Tensor(torch.long)[:,:]] :
    assert len(tensors) > 0

    for tensor in tensors:
        assert len(tensor.shape) == len(tensors[0].shape)
        assert tensor.dtype == tensors[0].dtype
        assert tensor.device == tensors[0].device

    max_sizes = [max(t.shape[i] for t in tensors) for i in range(len(tensors[0].shape))]
    newdimsizes = [len(tensors)] + max_sizes

    newt = torch.zeros(newdimsizes, dtype=tensors[0].dtype, device=tensors[0].device)
    sizes = torch.zeros(
        (len(tensors), tensors[0].dim()), dtype=torch.int64, device=tensors[0].device
    )
    strides = torch.zeros(
        (len(tensors), tensors[0].dim()), dtype=torch.int64, device=tensors[0].device
    )

    for i, t in enumerate(tensors):
        ti = newt[i, :]
        for j in range(t.dim()):
            ti = ti.narrow(j, 0, t.shape[j])
        ti[:] = t
        sizes[i, :] = torch.tensor(t.shape, dtype=torch.int64, device=t.device)
        strides[i, :] = torch.tensor(ti.stride(), dtype=torch.int64, device=t.device)
    return newt, sizes, strides


def cat_differently_sized_tensors(tensors: List,):
    assert len(tensors) > 0
    for tensor in tensors:
        assert len(tensor.shape) == len(tensors[0].shape)
        assert tensor.dtype == tensors[0].dtype
        assert tensor.device == tensor[0].device

    new_sizes = [max(t.shape[i] for t in tensors) for i in range(len(tensors[0].shape))]
    catdim_sizes = [t.shape[0] for t in tensors]
    n_entries_for_catdim = sum(catdim_sizes)
    new_sizes[0] = n_entries_for_catdim

    device = tensors[0].device

    newt = torch.zeros(new_sizes, dtype=tensors[0].dtype, device=device)

    sizes = torch.zeros(
        (n_entries_for_catdim, tensors[0].dim() - 1), dtype=torch.int64, device=device
    )
    strides = torch.zeros(
        (n_entries_for_catdim, tensors[0].dim() - 1), dtype=torch.int64, device=device
    )
    strides[:] = torch.unsqueeze(
        torch.tensor(newt.stride()[1:], dtype=torch.int64, device=device), dim=0
    )

    start = 0
    for i, t in enumerate(tensors):
        ti = newt[start : (start + catdim_sizes[i]), :]
        for j in range(1, t.dim()):
            ti = ti.narrow(j, 0, t.shape[j])
        ti[:] = t
        size_i = sizes[start : (start + catdim_sizes[i]), :]
        size_i[:] = torch.unsqueeze(
            torch.tensor((t.shape[1:]), dtype=torch.int64, device=device), dim=0
        )

        start += catdim_sizes[i]
    return newt, sizes, strides