import attr
from attr import asdict
from toolz import merge, valmap
from typing import Callable, Mapping, Union

import torch
import numpy

from tmol.types.functional import validate_args
from tmol.utility.args import ignore_unused_kwargs

from .numba.lj import lj_intra, lj_intra_backward, lj_inter, lj_inter_backward
from .params import LJLKDatabase, LJLKParamResolver


@attr.s(auto_attribs=True, frozen=True)
class LJOp:
    """torch.autograd lj baseline operator."""

    param_resolver: LJLKParamResolver
    params: Mapping[str, Union[float, numpy.ndarray]]
    lj_intra: Callable
    lj_intra_backward: Callable
    lj_inter: Callable
    lj_inter_backward: Callable

    @classmethod
    @validate_args
    def from_param_resolver(cls, param_resolver: LJLKParamResolver):
        return cls(
            param_resolver=param_resolver,
            params=merge(
                valmap(float, asdict(param_resolver.global_params)),
                valmap(torch.Tensor.numpy, asdict(param_resolver.type_params)),
            ),
            lj_intra=ignore_unused_kwargs(lj_intra),
            lj_intra_backward=ignore_unused_kwargs(lj_intra_backward),
            lj_inter=ignore_unused_kwargs(lj_inter),
            lj_inter_backward=ignore_unused_kwargs(lj_inter_backward),
        )

    @classmethod
    @validate_args
    def from_database(cls, ljlk_database: LJLKDatabase):

        return cls.from_param_resolver(
            param_resolver=LJLKParamResolver.from_database(
                ljlk_database, torch.device("cpu")
            )
        )

    def intra(self, coords, atom_types, bonded_path_lengths):
        # Detach grad from output indicies, which are int-valued
        i, v = self.LJIntraFun(self)(coords, atom_types, bonded_path_lengths)
        return (i.detach(), v)

    def inter(
        self, coords_a, atom_types_a, coords_b, atom_types_b, bonded_path_lengths
    ):
        # Detach grad from output indicies, which are int-valued
        i, v = self.LJInterFun(self)(
            coords_a, atom_types_a, coords_b, atom_types_b, bonded_path_lengths
        )
        return (i.detach(), v)

    class LJIntraFun(torch.autograd.Function):
        def __init__(self, op):
            self.op = op
            super().__init__()

        def forward(ctx, coords, atom_types, bonded_path_lengths):
            assert coords.dim() == 2
            assert coords.shape[1] == 3
            assert atom_types.shape == coords.shape[:1]

            assert bonded_path_lengths.shape == (coords.shape[0], coords.shape[0])

            assert all(
                t.device.type == "cpu"
                for t in (coords, atom_types, bonded_path_lengths)
            )

            assert not atom_types.requires_grad
            assert not bonded_path_lengths.requires_grad

            inds, vals = map(
                torch.from_numpy,
                ctx.op.lj_intra(
                    coords.detach().numpy(),
                    atom_types.numpy(),
                    bonded_path_lengths.numpy(),
                    **ctx.op.params,
                ),
            )

            ctx.save_for_backward(inds, coords, atom_types, bonded_path_lengths)

            return (inds, vals)

        def backward(ctx, ind_grads, val_grads):

            inds, coords, atom_types, bonded_path_lengths = ctx.saved_tensors

            coord_grads = ctx.op.lj_intra_backward(
                inds,
                val_grads.numpy(),
                coords.detach().numpy(),
                atom_types.numpy(),
                bonded_path_lengths.numpy(),
                **ctx.op.params,
            )

            return torch.from_numpy(coord_grads), None, None

    class LJInterFun(torch.autograd.Function):
        def __init__(self, op):
            self.op = op
            super().__init__()

        def forward(
            ctx, coords_a, atom_types_a, coords_b, atom_types_b, bonded_path_lengths
        ):
            assert coords_a.dim() == 2
            assert coords_a.shape[1] == 3
            assert atom_types_a.shape == coords_a.shape[:1]

            assert coords_b.dim() == 2
            assert coords_b.shape[1] == 3
            assert atom_types_b.shape == coords_b.shape[:1]

            assert bonded_path_lengths.shape == (coords_a.shape[0], coords_b.shape[0])

            assert all(
                t.device.type == "cpu"
                for t in (
                    coords_a,
                    atom_types_a,
                    coords_b,
                    atom_types_b,
                    bonded_path_lengths,
                )
            )

            assert not atom_types_a.requires_grad
            assert not atom_types_b.requires_grad
            assert not bonded_path_lengths.requires_grad

            inds, vals = map(
                torch.from_numpy,
                ctx.op.lj_inter(
                    coords_a.detach().numpy(),
                    atom_types_a.numpy(),
                    coords_b.detach().numpy(),
                    atom_types_b.numpy(),
                    bonded_path_lengths.numpy(),
                    **ctx.op.params,
                ),
            )

            ctx.save_for_backward(
                inds,
                coords_a,
                atom_types_a,
                coords_b,
                atom_types_b,
                bonded_path_lengths,
            )

            return (inds, vals)

        def backward(ctx, ind_grads, val_grads):

            (
                inds,
                coords_a,
                atom_types_a,
                coords_b,
                atom_types_b,
                bonded_path_lengths,
            ) = ctx.saved_tensors

            coord_a_grads, coord_b_grads = ctx.op.lj_inter_backward(
                inds,
                val_grads.numpy(),
                coords_a.detach().numpy(),
                atom_types_a.numpy(),
                coords_b.detach().numpy(),
                atom_types_b.numpy(),
                bonded_path_lengths.numpy(),
                **ctx.op.params,
            )

            return (
                torch.from_numpy(coord_a_grads),
                None,
                torch.from_numpy(coord_b_grads),
                None,
                None,
            )
