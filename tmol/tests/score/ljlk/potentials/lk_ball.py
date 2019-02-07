from typing import Tuple

import attr
import torch
from toolz import merge, keymap

from tmol.types.torch import Tensor

from tmol.utility.args import ignore_unused_kwargs
from tmol.score.ljlk.params import LJLKDatabase

from tmol.utility.cpp_extension import load, relpaths, modulename

_compiled = load(modulename(__name__), relpaths(__file__, ["lk_ball.pybind.cpp"]))


def detach_maybe_requires_grad(
    *inputs: torch.tensor
) -> Tuple[bool, Tuple[torch.tensor, ...]]:
    requires_grad = any(t.requires_grad for t in inputs)

    if requires_grad:
        return requires_grad, tuple(t.detach() for t in inputs)
    else:
        return requires_grad, inputs


class BuildAcceptorWater(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        A: Tensor(float)[:, 3],
        B: Tensor(float)[:, 3],
        B0: Tensor(float)[:, 3],
        dist: float,
        angle: float,
        torsion: float,
    ) -> Tensor(float)[:, 3]:

        rgrad, (A, B, B0) = detach_maybe_requires_grad(A, B, B0)
        inputs = (A, B, B0, dist, angle, torsion)

        if rgrad:
            ctx.inputs = inputs

        return torch.from_numpy(_compiled.build_acc_water_V(*inputs))

    @staticmethod
    def backward(ctx, dE_dW: Tensor(float)[:, 3]):
        inputs = ctx.inputs
        dW_dA, dW_dB, dW_dB0 = map(
            torch.from_numpy, _compiled.build_acc_water_dV(*inputs)
        )

        return dW_dA @ dE_dW, dW_dB @ dE_dW, dW_dB0 @ dE_dW, None, None, None


class BuildDonorWater(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, D: Tensor(float)[:, 3], H: Tensor(float)[:, 3], dist: float
    ) -> Tensor(float)[:, 3]:

        rgrad, (D, H) = detach_maybe_requires_grad(D, H)
        inputs = (D, H, dist)

        if rgrad:
            ctx.inputs = inputs

        return torch.from_numpy(_compiled.build_don_water_V(*inputs))

    @staticmethod
    def backward(ctx, dE_dW: Tensor(float)[:, 3]):
        inputs = ctx.inputs

        dW_dD, dW_dH = map(torch.from_numpy, _compiled.build_don_water_dV(*inputs))

        return dW_dD @ dE_dW, dW_dH @ dE_dW, None


class LKFraction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        args = list(args)
        rgrad, args[:2] = detach_maybe_requires_grad(*args[:2])

        if rgrad:
            ctx.args = args

        return torch.tensor(_compiled.lk_fraction_V(*args)).to(args[0].dtype)

    @staticmethod
    def backward(ctx, dE_dF: Tensor(float)):
        args = ctx.args
        d_grad_args = map(torch.from_numpy, _compiled.lk_fraction_dV(*args))

        return tuple(a * dE_dF for a in d_grad_args) + tuple(None for a in args[2:])


class LKBridgeFraction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        args = list(args)
        rgrad, args[:4] = detach_maybe_requires_grad(*args[:4])

        if rgrad:
            ctx.args = args

        return torch.tensor(_compiled.lk_bridge_fraction_V(*args)).to(args[0].dtype)

    @staticmethod
    def backward(ctx, dE_dF: Tensor(float)):
        args = ctx.args
        d_grad_args = map(torch.from_numpy, _compiled.lk_bridge_fraction_dV(*args))

        return tuple(a * dE_dF for a in d_grad_args) + tuple(None for a in args[4:])


@attr.s(auto_attribs=True, frozen=True)
class LKBallScore:

    ljlk_database: LJLKDatabase

    def apply(
        self,
        coord_i,
        coord_j,
        waters_i,
        waters_j,
        bonded_path_length,
        atom_type_i,
        atom_type_j,
    ):
        params_i = {p.name: p for p in self.ljlk_database.atom_type_parameters}[
            atom_type_i
        ]
        params_j = {p.name: p for p in self.ljlk_database.atom_type_parameters}[
            atom_type_j
        ]

        params = merge(
            keymap(lambda k: f"i_{k}", attr.asdict(params_i)),
            keymap(lambda k: f"j_{k}", attr.asdict(params_j)),
            attr.asdict(self.ljlk_database.global_parameters),
        )

        return LKBallScoreFun(params)(
            coord_i, coord_j, waters_i, waters_j, bonded_path_length
        )


class LKBallScoreFun(torch.autograd.Function):
    def __init__(self, params):
        self.params = params
        self.lk_ball_score_V = ignore_unused_kwargs(_compiled.lk_ball_score_V)
        super().__init__()

    def forward(
        ctx,
        coord_i: Tensor(float)[3],
        coord_j: Tensor(float)[3],
        waters_i: Tensor(float)[2, 3],
        waters_j: Tensor(float)[2, 3],
        bonded_path_length: Tensor(float)[1],
    ) -> Tensor(float):

        rgrad, (coord_i, coord_j, waters_i, waters_j) = detach_maybe_requires_grad(
            coord_i, coord_j, waters_i, waters_j
        )
        inputs = (coord_i, coord_j, waters_i, waters_j, bonded_path_length)

        if rgrad:
            ctx.inputs = inputs

        return torch.tensor(ctx.lk_ball_score_V(*inputs, **ctx.params)).to(
            coord_i.dtype
        )
