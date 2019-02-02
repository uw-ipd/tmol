import attr
from attr import asdict
from toolz import merge, valmap
from typing import Callable, Mapping

import torch

from tmol.types.functional import validate_args
from tmol.utility.args import ignore_unused_kwargs


class ElecScoreFun(torch.autograd.Function):
    def forward(
        ctx,
        I,
        pc_I,
        J,
        pc_J,
        cp_bonded_path_lengths,
        min_dis,
        max_dis,
        die_D,
        die_D0,
        die_S,
    ):
        assert I.dim() == 2
        assert I.shape[1] == 3
        assert I.shape[:1] == atom_type_I.shape
        assert not atom_type_I.requires_grad

        assert J.dim() == 2
        assert J.shape[1] == 3
        assert J.shape[:1] == atom_type_J.shape
        assert not atom_type_J.requires_grad

        from .potentials import compiled

        inds, E, *dE_dC = compiled.fa_elec(
            I,
            pc_I,
            J,
            pc_J,
            cp_bonded_path_lengths,
            min_dis,
            max_dis,
            die_D,
            die_D0,
            die_S,
        )

        # Assert of returned shape of indicies and scores. Seeing strange
        # results w/ reversed ordering if mgpu::tuple converted std::tuple
        assert inds.dim() == 2
        assert inds.shape[1] == 2
        assert inds.shape[0] == E.shape[0]

        inds = inds.transpose(0, 1)

        ctx.shape_I = atom_type_I.shape
        ctx.shape_J = atom_type_J.shape

        ctx.save_for_backward(*([inds] + dE_dC))

        return (inds, E)

    def backward(ctx, _ind_grads, dV_dE):
        ind, dE_dI, dE_dJ = ctx.saved_tensors
        ind_I, ind_J = ind

        dV_dI = torch.sparse_coo_tensor(
            ind_I[None, :], dV_dE[..., None] * dE_dI, ctx.shape_I + (3,)
        ).to_dense()

        dV_dJ = torch.sparse_coo_tensor(
            ind_J[None, :], dV_dE[..., None] * dE_dJ, ctx.shape_J + (3,)
        ).to_dense()

        return (dV_dI, None, dV_dJ, None, None, None, None, None, None, None)
