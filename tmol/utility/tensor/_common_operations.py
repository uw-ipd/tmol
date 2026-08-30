from collections.abc import Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

from tmol.types import (
    Tensor,
    validate_args,
)

IntTensor1D = Tensor[torch.int32][:] | Tensor[torch.int64][:]
IntTensor2D = Tensor[torch.int32][:, :] | Tensor[torch.int64][:, :]
RepeatCount = int | torch.Tensor


@validate_args
def stretch(t: IntTensor1D, count: RepeatCount) -> IntTensor1D:
    """Repeat each element of a one-dimensional integer tensor.

    Args:
        t: Values to repeat.
        count: Number of repeats, as an integer or scalar integer tensor.

    Returns:
        A flattened tensor with each input element repeated ``count`` times.
    """
    return t.repeat(count).view(count, -1).permute(1, 0).contiguous().view(-1)


@validate_args
def stretch2(t: IntTensor2D, count: RepeatCount) -> IntTensor2D:
    """Repeat each element along the second dimension of an integer tensor.

    Args:
        t: Two-dimensional values to repeat.
        count: Number of repeats, as an integer or scalar integer tensor.

    Returns:
        A tensor with each row element repeated ``count`` times.
    """
    return (
        t.repeat(1, count)
        .view(t.shape[0], count, -1)
        .permute(0, 2, 1)
        .contiguous()
        .view(t.shape[0], -1)
    )


@validate_args
def exclusive_cumsum1d(inds: IntTensor1D) -> IntTensor1D:
    """Compute an exclusive prefix sum over a one-dimensional integer tensor.

    Args:
        inds: Values to accumulate.

    Returns:
        Prefix sums with a zero in the first position.
    """
    return torch.cat(
        (
            torch.tensor([0], dtype=inds.dtype, device=inds.device),
            torch.cumsum(inds, 0, dtype=inds.dtype).narrow(0, 0, inds.shape[0] - 1),
        )
    )


@validate_args
def exclusive_cumsum2d(inds: IntTensor2D) -> IntTensor2D:
    """Compute exclusive prefix sums along each row of an integer tensor.

    Args:
        inds: Values to accumulate along dimension one.

    Returns:
        Row-wise prefix sums with zeros in the first column.
    """
    return torch.cat(
        (
            torch.zeros((inds.shape[0], 1), dtype=inds.dtype, device=inds.device),
            torch.cumsum(inds, dim=1, dtype=inds.dtype)[:, :-1],
        ),
        dim=1,
    )


@validate_args
def exclusive_cumsum2d_and_totals(
    inds: IntTensor2D,
) -> (
    tuple[Tensor[torch.int32][:, :], Tensor[torch.int32][:]]
    | tuple[Tensor[torch.int64][:, :], Tensor[torch.int64][:]]
):
    """Compute row-wise exclusive prefix sums and inclusive row totals.

    Args:
        inds: Values to accumulate along dimension one.

    Returns:
        The exclusive prefix sums and the sum of each row.
    """
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


def print_row_numbered_tensor(tensor: torch.Tensor) -> None:
    """Print a one- or two-dimensional tensor with zero-based row indices."""
    if tensor.ndim not in (1, 2):
        raise ValueError("tensor must be one- or two-dimensional")
    row_numbers = torch.arange(
        tensor.shape[0], dtype=tensor.dtype, device=tensor.device
    ).reshape(-1, 1)
    if len(tensor.shape) == 1:
        print(torch.cat((row_numbers, tensor.reshape(-1, 1)), dim=1))
    else:
        print(torch.cat((row_numbers, tensor), dim=1))


def _validate_tensor_sequence(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Return the first tensor after validating shared structural metadata."""
    if not tensors:
        raise ValueError("at least one tensor is required")

    first = tensors[0]
    for tensor in tensors[1:]:
        if tensor.ndim != first.ndim:
            raise ValueError("all tensors must have the same number of dimensions")
        if tensor.dtype != first.dtype:
            raise ValueError("all tensors must have the same dtype")
        if tensor.device != first.device:
            raise ValueError("all tensors must be on the same device")
    return first


def nplus1d_tensor_from_list(
    tensors: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, Tensor[torch.int64][:, :], Tensor[torch.int64][:, :]]:
    """Pad tensors into a new leading dimension and report shape metadata.

    Args:
        tensors: Same-rank tensors with a shared dtype and device.

    Returns:
        The padded tensor, original sizes, and strides into the padded tensor.
    """
    first = _validate_tensor_sequence(tensors)

    max_sizes = [max(t.shape[i] for t in tensors) for i in range(first.ndim)]
    newdimsizes = [len(tensors)] + max_sizes

    newt = torch.zeros(newdimsizes, dtype=first.dtype, device=first.device)
    sizes = torch.zeros(
        (len(tensors), first.dim()), dtype=torch.int64, device=first.device
    )
    strides = torch.zeros(
        (len(tensors), first.dim()), dtype=torch.int64, device=first.device
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
    tensors: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, Tensor[torch.int64][:, :], Tensor[torch.int64][:, :]]:
    """Concatenate padded tensors along dimension zero and report metadata.

    Args:
        tensors: Same-rank tensors with a shared dtype and device.

    Returns:
        The padded concatenation, original trailing sizes, and output strides.
    """
    first = _validate_tensor_sequence(tensors)

    new_sizes = [max(t.shape[i] for t in tensors) for i in range(first.ndim)]
    catdim_sizes = [t.shape[0] for t in tensors]
    n_entries_for_catdim = sum(catdim_sizes)
    new_sizes[0] = n_entries_for_catdim

    device = first.device

    newt = torch.zeros(new_sizes, dtype=first.dtype, device=device)

    sizes = torch.zeros(
        (n_entries_for_catdim, first.dim() - 1), dtype=torch.int64, device=device
    )
    strides = torch.zeros(
        (n_entries_for_catdim, first.dim() - 1), dtype=torch.int64, device=device
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


def join_tensors_and_report_real_entries(
    tensors: Sequence[torch.Tensor], sentinel: int = -1
) -> tuple[Tensor[torch.int32][:], Tensor[torch.bool][:, :], torch.Tensor]:
    """Pad tensors along dimension zero and identify their real entries.

    Args:
        tensors: Tensors with matching trailing dimensions, dtype, and device.
        sentinel: Value used to pad missing entries.

    Returns:
        Per-tensor lengths, a validity mask, and the padded tensor batch.
    """

    first = _validate_tensor_sequence(tensors)
    if any(t.shape[1:] != first.shape[1:] for t in tensors[1:]):
        raise ValueError("all tensors must have the same shape after dimension zero")

    device = first.device

    n_elements = torch.tensor(
        [t.shape[0] for t in tensors], dtype=torch.int32, device=device
    )
    padding_value = float(sentinel)
    padding_is_exact = (
        first.is_floating_point()
        or first.is_complex()
        or int(padding_value) == sentinel
    )
    combo = pad_sequence(
        tensors,
        batch_first=True,
        padding_value=padding_value if padding_is_exact else 0.0,
    )
    real = torch.arange(
        combo.shape[1], dtype=n_elements.dtype, device=device
    ).unsqueeze(0) < n_elements.unsqueeze(1)
    if not padding_is_exact:
        padding_mask = (~real).reshape(real.shape + (1,) * (combo.ndim - 2))
        combo.masked_fill_(padding_mask, sentinel)

    return n_elements, real, combo


def invert_mapping(
    a_2_b: IntTensor1D,
    n_elements_b: int | torch.Tensor | None = None,
    sentinel: int = -1,
) -> IntTensor1D:
    """Create the inverse mapping ``b_2_a`` for an input mapping ``a_2_b``.

    Args:
        a_2_b: One-dimensional integer mapping from A indices to B indices.
        n_elements_b: Output size, inferred from ``a_2_b`` when omitted.
        sentinel: Value assigned to B indices without a corresponding A index.

    Returns:
        A mapping from B indices back to A indices.
    """
    if n_elements_b is None:
        n_elements_b = torch.max(a_2_b) + 1

    b_2_a = torch.full(
        (n_elements_b,), sentinel, dtype=a_2_b.dtype, device=a_2_b.device
    )

    b_2_a[a_2_b.to(torch.int64)] = torch.arange(
        a_2_b.shape[0], dtype=a_2_b.dtype, device=a_2_b.device
    )
    return b_2_a
