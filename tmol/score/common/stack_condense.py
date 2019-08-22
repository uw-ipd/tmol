import torch
import numpy
from tmol.types.torch import Tensor
from tmol.types.array import NDArray


def condense_numpy_inds(selection: NDArray(bool)[:, :]):
    """Given a two dimensional boolean tensor, create
    an output tensor holding the column indices of the non-zero
    entries for each row. Pad out the extra entries
    in any given row that do not correspond to a selected
    entry with a sentinel of -1.
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
