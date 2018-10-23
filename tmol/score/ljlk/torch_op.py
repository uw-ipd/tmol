from toolz import first

import attr
import torch


from .params import LJLKParamResolver
from . import cpp_potential

import typing


@attr.s(auto_attribs=True, frozen=True)
class LJOp:
    device: torch.device
    params: typing.Mapping[str, typing.Any]

    def intra(self, coords, types, bonded_path_length):
        return LJIntraFun(self)(coords, types, bonded_path_length)

    @classmethod
    def from_params(cls, param_resolver: LJLKParamResolver):

        pair_params = param_resolver.pair_params
        global_params = param_resolver.global_params

        params = dict(
            lj_sigma=pair_params.lj_sigma,
            lj_switch_slope=pair_params.lj_switch_slope,
            lj_switch_intercept=pair_params.lj_switch_intercept,
            lj_coeff_sigma12=pair_params.lj_coeff_sigma12,
            lj_coeff_sigma6=pair_params.lj_coeff_sigma6,
            lj_spline_y0=pair_params.lj_spline_y0,
            lj_spline_dy0=pair_params.lj_spline_dy0,
            # Global param_resolver
            lj_switch_dis2sigma=global_params.lj_switch_dis2sigma.reshape(1),
            spline_start=global_params.spline_start.reshape(1),
            max_dis=global_params.max_dis.reshape(1),
        )

        assert len(set((p.device.type, p.device.index) for p in params.values())) == 1

        device = first(params.values()).device

        return cls(device=device, params=params)


class LJIntraFun(torch.autograd.Function):
    def __init__(self, op: LJOp):
        self.op = op
        super().__init__()

    def forward(ctx, coords, types, bonded_path_length):

        assert coords.device == ctx.op.device
        assert not coords.requires_grad

        assert types.device == ctx.op.device
        assert not types.requires_grad

        assert bonded_path_length.device == ctx.op.device
        assert not bonded_path_length.requires_grad

        blocked_interactions = cpp_potential.lj_intra(
            coords, types, bonded_path_length, **ctx.op.params
        )
        return blocked_interactions._indices(), blocked_interactions._values()
