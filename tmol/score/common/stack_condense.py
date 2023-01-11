import torch
import numpy
from tmol.types.torch import Tensor
from tmol.types.array import NDArray
from typing import Union, Optional

from tmol.types.functional import validate_args

##################################################################
# For operations on stacked systems, we have to deal with the fact
# that systems will almost always be of different sizes. The
# operations in this file aim to shuffle data into a form where
# the entries that "hang off the end" of a row in a stack (but
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


def condense_torch_inds(selection: Tensor[bool][:, :], device: torch.device):
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
    nz_selection = torch.nonzero(selection, as_tuple=False)
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
    value_tensor, sentineled_index_tensor: Tensor[torch.int64][:, :], default_fill=-1
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
    sentineled_index_tensor: Tensor[torch.int64][:, :],
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
    nz_cinds = torch.nonzero(cinds >= 0, as_tuple=False)
    selected_values[nz_cinds[:, 0], nz_cinds[:, 1], :] = values[
        nz_cinds[:, 0], cinds[cinds >= 0].view(-1), :
    ]
    return selected_values


@validate_args
def take_condensed_3d_subset(
    values,  # 3D Tensor of arbitrary dtype
    condensed_inds_to_keep: Tensor[torch.int64][:, :],
    condensed_dst_inds: Tensor[torch.int64][:, 2],
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


@validate_args
def tile_subset_indices(
    indices: Union[
        Tensor[torch.int32][:],
        Tensor[torch.int64][:],
        NDArray[numpy.int32][:],
        NDArray[numpy.int64][:],
    ],
    tile_size: int,
    max_entry: Optional[int] = None,
):
    """Take the indices of a subset of things and "tile" them so that they're
    in groups based on the equivalence class `i // tile_size` and left-justify
    the indices within the tile.

    E.g.
    If the subset indices are [0, 3, 4, 7, 10, 12, 14]
    and the tile_size is 8,
    then the output will be:
    [0, 3, 4, 7, -1, -1, -1, -1, 2, 4, 6, -1, -1, -1, -1, -1] and
    [4, 3]
    representing the tiling of the indices and the number of indices per tile,
    reflecting there being two tiles where there are four values in
    the first tile and three values in the second tile.
    The indices are given as tile indices so that 10-->2,
    12-->4, 14-->6. The entries that are in the first tile remain
    unchanged, of course.

    If desired, a maximum index can be given so that a desired number
    of tiles can be created even if the subset includes no entries for
    the last tile.

    E.g.
    If the subset indices are [0, 3, 4, 7]
    and the tile size is 8 and the max_entry is 15, then two tiles are desired
    and the output will be:
    [0, 3, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1] and
    [4, 0]
    representing the tiling of the indices and the number of indices per tile

    Works for both torch and numpy inputs.
    """
    return _value_or_arg_tile_subset_indices(indices, tile_size, False, max_entry)


@validate_args
def arg_tile_subset_indices(
    indices: Union[
        Tensor[torch.int32][:],
        Tensor[torch.int64][:],
        NDArray[numpy.int32][:],
        NDArray[numpy.int64][:],
    ],
    tile_size: int,
    max_entry: Optional[int] = None,
):
    """Take the indices of a subset of things and return the indices (args) that
    would "tile" them so that they're in groups based on the equivalence class
    `i // tile_size` and left-justify those indices within the tiles.
    Having the indices of the tiled subset is desired in cases when there
    is additional data for the subset that also needs to be tiled.

    E.g.
    If the subset indices are [0, 3, 4, 7, 10, 12, 14]
    and the tile_size is 8,
    then the output will be:
    [0, 1, 2, 3, -1, -1, -1, -1, 4, 5, 6, -1, -1, -1, -1, -1] and
    [4, 3]
    representing the tiling of the indices by their indices in the input array
    (confusingly named) indices (!) and the number of indices per tile,
    reflecting there being two tiles, where there are four values in the
    first tile and three values in the second tile.

    If desired, a maximum index can be given so that a desired number
    of tiles can be created even if the subset includes no entries for
    the last tile.

    E.g.
    If the subset indices are [0, 3, 4, 7]
    and the tile size is 8 and the max_entry is 15, then two tiles are desired
    and the output will be:
    [0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1] and
    [4, 0]
    representing the tiling of the indices by their indices and the number
    of indices per tile.

    Works for both torch and numpy inputs.
    """
    return _value_or_arg_tile_subset_indices(indices, tile_size, True, max_entry)


def _value_or_arg_tile_subset_indices(
    indices: Union[
        Tensor[torch.int32][:],
        Tensor[torch.int64][:],
        NDArray[numpy.int32][:],
        NDArray[numpy.int64][:],
    ],
    tile_size: int,
    return_args: bool,
    max_entry: Optional[int] = None,
):
    if type(indices) == torch.Tensor:
        if max_entry is None:
            max_entry = torch.max(indices)
        n_tiles = max_entry // tile_size + 1
        tiled_indices = torch.full(
            (n_tiles * tile_size,), -1, dtype=indices.dtype, device=indices.device
        )
        n_in_tile = torch.full(
            (n_tiles,), 0, dtype=indices.dtype, device=indices.device
        )
        if return_args:
            ind_arange = torch.arange(
                indices.shape[0], dtype=indices.dtype, device=indices.device
            )
    elif type(indices) == numpy.ndarray:
        if max_entry is None:
            max_entry = numpy.amax(indices)
        n_tiles = max_entry // tile_size + 1
        tiled_indices = numpy.full_like(indices, -1, shape=(n_tiles * tile_size,))
        n_in_tile = numpy.full_like(indices, 0, shape=(n_tiles,))
        if return_args:
            ind_arange = numpy.arange(indices.shape[0], dtype=indices.dtype)
    else:
        raise ValueError
    for i in range(n_tiles):
        subset = (indices >= i * tile_size) & (indices < (i + 1) * tile_size)
        if type(tiled_indices) == torch.Tensor:
            subset_size = torch.sum(subset).cpu()
        else:
            subset_size = numpy.sum(subset)
        s = slice(i * tile_size, i * tile_size + subset_size)
        if return_args:
            tiled_indices[s] = ind_arange[subset]
        else:
            tiled_indices[s] = indices[subset] - i * tile_size
        n_in_tile[i] = subset_size

    return tiled_indices, n_in_tile
