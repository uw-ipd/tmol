import attr
from attr import asdict
from toolz import merge, valmap
from typing import Callable, Mapping, Union

import torch
import numpy

from tmol.types.functional import validate_args
from tmol.utility.args import ignore_unused_kwargs

from .lj import lj_intra, lj_intra_backward
from ..params import LJLKDatabase, LJLKParamResolver


@attr.s(auto_attribs=True, frozen=True)
class LJOp:
    """torch.autograd lj baseline operator."""

    param_resolver: LJLKParamResolver
    params: Mapping[str, Union[float, numpy.ndarray]]
    lj_intra: Callable
    lj_intra_backward: Callable

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
        i, v = LJIntraFun(self)(coords, atom_types, bonded_path_lengths)
        return (i.detach(), v)


class LJIntraFun(torch.autograd.Function):
    def __init__(self, op: LJOp):
        self.op = op
        super().__init__()

    def forward(ctx, coords, atom_types, bonded_path_lengths):

        assert all(
            t.device.type == "cpu" for t in (coords, atom_types, bonded_path_lengths)
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
