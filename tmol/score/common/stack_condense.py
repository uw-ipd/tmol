import torch
import numpy
from tmol.types.torch import Tensor
from tmol.types.array import NDArray

from tmol.types.functional import validate_args

##################################################################
# For operations on stacked systems, we have to deal with the fact
# that systems will almost always be of different sizes. The
# operations in this file aim to shuffle data into a form where
# the entries that "hand off the end" of a row in a stack (but
# which have to be there because another row of a stack is
# longer) are shifted to the higher indices (the "right" if you
# draw the stack on paper) and the entries that correspond
# to real data are all shifted to the lower indices (the "left").
#
# Call tensors that arrange their data this way "condensed."
#
# The convention used here is to mark all non-real entries
# with a sentinel of -1
#
# The functions here will help:
# 1. map a 2D tensor with sporadically placed sentineled entries
#    into a condensed tensor, and
# 2. index into one tensor into a condensed output tensor


def condense_numpy_inds(selection: NDArray[bool][:, :]):
    """Given a two dimensional boolean tensor, create
    an output tensor holding the column indices of the non-zero
    entries for each row. Pad out the extra entries
    in any given row that do not correspond to a selected
    entry with a sentinel of -1.

    e.g. if the input is

    [[ 0  1  0  1]
    [  1  1  0  1]]

    then the output will be

    [[ 1  3 -1]
    [  0  1  3]]
    """

    nstacks = selection.shape[0]
    nz_selection = numpy.nonzero(selection)
    nkeep = numpy.sum(selection, axis=1).reshape((nstacks, 1))
    max_keep = numpy.max(nkeep)
    inds = numpy.full((nstacks, max_keep), -1, dtype=int)
    counts = numpy.arange(max_keep, dtype=int).reshape((1, max_keep))
    lowinds = counts < nkeep

    inds[lowinds] = nz_selection[1]
    return inds


def condense_torch_inds(selection: Tensor(bool)[:, :], device: torch.device):
    """Given a two dimensional boolean tensor, create
    an output tensor holding the column indices of the non-zero
    entries for each row. Pad out the extra entries
    in any given row that do not correspond to a selected
    entry with a sentinel of -1.

    e.g. if the input is
    [[ 0  1  0  1]
    [  1  1  0  1]]
    then the output will be
    [[ 1  3 -1]
    [  0  1  3]]
    """

    nstacks = selection.shape[0]
    nz_selection = torch.nonzero(selection)
    nkeep = torch.sum(selection, dim=1).view((nstacks, 1))
    max_keep = torch.max(nkeep)
    inds = torch.full((nstacks, max_keep), -1, dtype=torch.int64, device=device)
    counts = torch.arange(max_keep, dtype=torch.int64, device=device).view(
        (1, max_keep)
    )
    lowinds = counts < nkeep

    inds[lowinds] = nz_selection[:, 1]
    return inds


@validate_args
def take_values_w_sentineled_index(
    value_tensor, sentineled_index_tensor: Tensor(torch.int64)[:, :], default_fill=-1
):
    """The sentinel in the sentineled_index_tensor is -1: the positions
    with the sentinel value should not be used as an index into the
    value tensor. This function returns a tensor of the same shape as
    the sentineled_index_tensor with a dtype of the value tensor.

    E.g. if the value tensor is [10 11 12 13 14 15]
    and the sentineled_index_tensor is

    [[ 2 1 2 5 -1]
    [  1 4 1 5  2]]

    then the output tensor will be

    [[ 12 11 12 15 -1]
    [  11 14 11 15  12]]
    """

    assert len(value_tensor.shape) == 1

    output_value_tensor = torch.full(
        sentineled_index_tensor.shape,
        default_fill,
        dtype=value_tensor.dtype,
        device=value_tensor.device,
    )
    output_value_tensor[sentineled_index_tensor != -1] = value_tensor[
        sentineled_index_tensor[sentineled_index_tensor != -1]
    ]
    return output_value_tensor


@validate_args
def take_values_w_sentineled_index_and_dest(
    value_tensor,
    sentineled_index_tensor: Tensor(torch.int64)[:, :],
    sentineled_dest_tensor,
    default_fill=-1,
):
    """The sentinel in the sentineled_index_tensor is -1: the positions
    with the sentinel value should not be used as an index into the
    value tensor. The sentinel in the sentineled_dest_tensor is also
    -1: the positions with the sentinel value should not be written
    to in the output tensor. This function returns a tensor of the
    same shape as the sentineled_dest_tensor with a dtype of the
    value tensor, which is indexed into using the
    sentineled_index_tensor. The values in the sentineled_dest_tensor
    do not matter except where they are -1.

    E.g. if the value tensor is [10 11 12 13 14 15],
    the sentineled_index_tensor is
    [[ 2 -1  2  5 -1]
    [  1  4 -1  5  2]],
    and the sentineled_dest_tensor is
    [[ 1  1  1 -1]
    [  1  1  1  1]]

    then the output tensor will be
    [[ 12 12 15 -1]
    [  11 14 15 12]]
    """

    assert len(value_tensor.shape) == 1
    assert len(sentineled_dest_tensor.shape) == 2

    output_value_tensor = torch.full(
        sentineled_dest_tensor.shape,
        default_fill,
        dtype=value_tensor.dtype,
        device=value_tensor.device,
    )
    output_value_tensor[sentineled_dest_tensor != -1] = value_tensor[
        sentineled_index_tensor[sentineled_index_tensor != -1]
    ]
    return output_value_tensor


def take_values_w_sentineled_dest(
    value_tensor,
    values_to_take,  # boolean mask tensor
    sentineled_dest_tensor,
    default_fill=-1,
):
    """Take a subset of the values from the value_tensor indicated by
    the boolean values_to_take tensor, and write them into an output
    tensor in a shape with non-negative-one values in the
    sentineled_dest_tensor. There need to be as many "true" values in
    the values_to_take tensor as they are non-negative-one values
    in the sentineled_dest_tensor.

    E.g. if the value tensor is
    [[10 11 12 13 14],
    [ 20 21 22 23 24]]
    the values_to_take tensor is
    [[ 1  0  1  1  0]
    [  1  1  0  1  1]],
    and the sentineled_dest_tensor is
    [[ 1  1  1 -1]
    [  1  1  1  1]]

    then the output tensor will be
    [[10 12 13 -1]
    [ 20 21 23 24]]
    """

    assert value_tensor.shape == values_to_take.shape
    output_value_tensor = torch.full(
        sentineled_dest_tensor.shape,
        default_fill,
        dtype=value_tensor.dtype,
        device=value_tensor.device,
    )
    output_value_tensor[sentineled_dest_tensor != -1] = value_tensor[values_to_take]
    return output_value_tensor


def condense_subset(
    values,  # three dimensional tensor of values
    values_to_keep,  # two dimensional boolean tensor
    default_fill=-1,
):
    """Take the values for the third dimension of the 3D "values" tensor,
    (condensing them), corresponding to the positions indicated by
    the values_to_keep tensor.

    E.g. if the values tensor is
    [[[10 10] [11 11] [12 12] [13 13] [14 14]],
    [ [20 20] [21 21] [22 22] [23 23] [24 24]]]
    the values_to_keep tensor is
    [[1 0 1 1 0]
    [ 1 1 0 1 1]]

    then the output tensor will be
    [[ [10 10] [12 12] [13 13] [ -1 -1]]
    [  [20 20] [21 21] [23 23] [24 24]]]
    """

    assert len(values.shape) == 3
    assert len(values_to_keep.shape) == 2
    assert values.shape[:2] == values_to_keep.shape
    cinds = condense_torch_inds(values_to_keep, values_to_keep.device)
    selected_values = torch.full(
        (cinds.shape[0], cinds.shape[1], values.shape[2]),
        default_fill,
        dtype=values.dtype,
        device=values.device,
    )
    nz_cinds = torch.nonzero(cinds >= 0)
    selected_values[nz_cinds[:, 0], nz_cinds[:, 1], :] = values[
        nz_cinds[:, 0], cinds[cinds >= 0].view(-1), :
    ]
    return selected_values


@validate_args
def take_condensed_3d_subset(
    values,  # 3D Tensor of arbitrary dtype
    condensed_inds_to_keep: Tensor(torch.int64)[:, :],
    condensed_dst_inds: Tensor(torch.int64)[:, 2],
    default_fill=-1,
):
    """Take the values for the third dimension of the 3D "values" tensor,
    at the positions indicated by the "condensed_inds_to_keep" tensor,
    and writing them to the indices indicated by the "condensed_dst_inds".
    This function is equivalent to the above "condense_subset" function
    if that function's "values_to_keep" tensor is converted to the
    inputs to this function with the following operations:

    condensed_inds_to_keep = condense_torch_inds(values_to_keep != -1, device)
    condensed_dst_inds = torch.nonzero(inds_to_keep != -1)

    This function is more efficient if you intend to use the
    "condensed_inds_to_keep" or the "condensed_dst_inds" tensors multiple
    times.

    E.g. if the values tensor is
    [[[10 10] [11 11] [12 12] [13 13] [14 14]],
    [ [20 20] [21 21] [22 22] [23 23] [24 24]]]
    the condensed_inds_to_keep tensor is
    [[ 0 -1  2  3]
    [  4  3  2  4]],
    and the condensed_dest_tensor is
    [[ 0 0]
    [  0 1]
    [  0 2]
    [  1 0]
    [  1 1]
    [  1 2]
    [  1 3]]

    then the output tensor will be
    [[ [10 10] [12 12] [13 13] [ -1 -1]]
    [  [24 24] [23 23] [22 22] [24 24]]]
    """

    assert len(values.shape) == 3

    keep = condensed_inds_to_keep
    dst = condensed_dst_inds

    subset = torch.full(
        (keep.shape + values.shape[2:3]),
        default_fill,
        dtype=values.dtype,
        device=values.device,
    )
    subset[dst[:, 0], dst[:, 1], :] = values[dst[:, 0], keep[keep != -1], :]
    return subset
