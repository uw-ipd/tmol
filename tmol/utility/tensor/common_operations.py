import torch
from typing import List, Tuple, Union, Optional
from tmol.types.torch import Tensor
from tmol.types.functional import validate_args


@validate_args
def stretch(t: Union[Tensor[torch.int32][:], Tensor[torch.int64][:]], count):
    """take an input tensor and "repeat" each element count times.
    stretch(tensor([0, 1, 2, 3]), 3) returns:
         tensor([0 0 0 1 1 1 2 2 2 3 3 3]
    this is equivalent to numpy's repeat
    """
    return t.repeat(count).view(count, -1).permute(1, 0).contiguous().view(-1)


@validate_args
def exclusive_cumsum1d(
    inds: Union[Tensor[torch.int32][:], Tensor[torch.int64][:]]
) -> Union[Tensor[torch.int32][:], Tensor[torch.int64][:]]:
    return torch.cat(
        (
            torch.tensor([0], dtype=inds.dtype, device=inds.device),
            torch.cumsum(inds, 0, dtype=inds.dtype).narrow(0, 0, inds.shape[0] - 1),
        )
    )


@validate_args
def exclusive_cumsum2d(
    inds: Union[Tensor[torch.int32][:, :], Tensor[torch.int64][:, :]]
) -> Union[Tensor[torch.int32][:, :], Tensor[torch.int64][:, :]]:
    return torch.cat(
        (
            torch.zeros((inds.shape[0], 1), dtype=inds.dtype, device=inds.device),
            torch.cumsum(inds, dim=1, dtype=inds.dtype)[:, :-1],
        ),
        dim=1,
    )


@validate_args
def exclusive_cumsum2d_and_totals(
    inds: Union[Tensor[torch.int32][:, :], Tensor[torch.int64][:, :]]
) -> Union[
    Tuple[Tensor[torch.int32][:, :], Tensor[torch.int32][:]],
    Tuple[Tensor[torch.int64][:, :], Tensor[torch.int64][:]],
]:
    cs = torch.cumsum(inds, dim=1, dtype=inds.dtype)
    return (
        torch.cat(
            (
                torch.zeros((inds.shape[0], 1), dtype=inds.dtype, device=inds.device),
                cs[:, :-1],
            ),
            dim=1,
        ),
        cs[:, -1],
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
    tensors: List,
):  # -> Tuple[Tensor, Tensor[torch.long][:,:], Tensor[torch.long][:,:]] :
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


def cat_differently_sized_tensors(
    tensors: List,
):
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


# def real_elements_of_differently_sized_tensors(tensors: List):
#     assert len(tensors) > 0
#     for tensor in tensors:
#         assert len(tensor.shape) == len(tensors[0].shape)
#         assert tensor.dtype == tensors[0].dtype
#         assert tensor.device == tensor[0].device
#
#     device = tensors[0].device
#     new_sizes = [max(t.shape[i] for t in tensors) for i in range(len(tensors[0].shape))]
#     arange_inds = torch.arange(new_sizes.shape[-1], dtype=torch.int64, device=device)


def join_tensors_and_report_real_entries(tensors: List, sentinel: int = -1):
    """Concatenate a bunch of N-dimensional tensors into a single N+1-D tensor
    and report which elements out of the new tensor are real.
    The tensors may have different sizes for dimension 0 but should have the
    same size for all other dimensions. They must all have the same
    dtype and live on the same device.
    """

    assert len(tensors) > 0
    for t in tensors:
        assert t.device == tensors[0].device
        assert t.dtype == tensors[0].dtype
        assert t.shape[1:] == tensors[0].shape[1:]

    device = tensors[0].device
    dtype = tensors[0].dtype

    max_d0 = max(t.shape[0] for t in tensors)
    new_shape = (len(tensors), max_d0) + tensors[0].shape[1:]

    n_elements = torch.tensor(
        [t.shape[0] for t in tensors], dtype=torch.int32, device=device
    )

    combo = torch.full(new_shape, sentinel, dtype=dtype, device=device)
    real = torch.full((len(tensors), max_d0), False, dtype=torch.bool, device=device)
    for i, t in enumerate(tensors):
        combo[i, : n_elements[i]] = t
        real[i, : n_elements[i]] = True

    return n_elements, real, combo


def invert_mapping(
    a_2_b: Union[Tensor[torch.int32][:], Tensor[torch.int64][:]],
    n_elements_b: Optional[int] = None,
    sentinel: Optional[int] = -1,
):
    Union[Tensor[torch.int32][:], Tensor[torch.int64][:]]
    """Create the inverse mapping, b_2_a, given the input mapping, a_2_b"""
    if n_elements_b is None:
        n_elements_b = torch.max(a_2_b) + 1

    b_2_a = torch.full(
        (n_elements_b,), sentinel, dtype=a_2_b.dtype, device=a_2_b.device
    )

    b_2_a[a_2_b.to(torch.int64)] = torch.arange(
        a_2_b.shape[0], dtype=a_2_b.dtype, device=a_2_b.device
    )
    return b_2_a
