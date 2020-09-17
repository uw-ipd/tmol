import attr
import cattr

import pandas

import torch
import numpy


from tmol.types.torch import Tensor
from tmol.types.tensor import TensorGroup
from tmol.types.array import NDArray
from tmol.types.attrs import ValidateAttrs
from tmol.types.functional import validate_args

from tmol.database.scoring.ljlk import LJLKDatabase
from tmol.database.chemical import ChemicalDatabase

from ..chemical_database import AtomTypeParamResolver


@attr.s(frozen=True, slots=True, auto_attribs=True)
class ConstraintResolver(ValidateAttrs):
    """Container for constraint parameters.

    Param resolver stores atom pair indices and corresponding functional form
    """

    device: torch.device

    spline_xs: torch.Tensor
    spline_ys: torch.Tensor

    # Since we use CB coords but not all sidechains have a CB
    #   we build up using a coordinate frame defined by N,C,CA
    # Note that individual peptides do not need to be complete
    #   or identical, HOWEVER, we assume that residues that contain
    #   _any_ of C/CA/N contain _all_ of them (an assertion fails
    #   if this is not the case)
    cb_frames: torch.Tensor  # indices of N/C/CA
    cb_stacks: torch.Tensor  # stack # of each frame
    cb_res_indices: torch.Tensor  # residue number of each frame
    nres: int

    @classmethod
    @validate_args
    def from_dense_CB_spline_data(
        cls,
        device: torch.device,
        atm_names: NDArray(object)[...],
        res_indices: NDArray("d")[...],
        spline_xs,
        spline_ys,
    ):
        """Get frame for each residue, N-C-CA, from which pseudo-CBs will be built."""
        iCA = (atm_names == "CA").nonzero()
        iN = (atm_names == "N").nonzero()
        iC = (atm_names == "C").nonzero()

        assert iCA[0].shape == iC[0].shape
        assert iCA[0].shape == iN[0].shape

        iRes = res_indices[iCA]
        assert numpy.all(iRes == res_indices[iC])
        assert numpy.all(iRes == res_indices[iN])

        cb_frames = torch.tensor(numpy.vstack((iN[1], iC[1], iCA[1])).T, device=device)
        cb_stacks = torch.tensor(iCA[0], device=device)
        cb_res_indices = torch.tensor(res_indices[iCA], dtype=torch.long, device=device)
        nres = spline_ys.shape[0]

        """Initialize param resolver for all atom types in database."""
        return cls(
            device=device,
            spline_xs=spline_xs.to(device),
            spline_ys=spline_ys.to(device),
            cb_frames=cb_frames,
            cb_stacks=cb_stacks,
            cb_res_indices=cb_res_indices,
            nres=nres,
        )
