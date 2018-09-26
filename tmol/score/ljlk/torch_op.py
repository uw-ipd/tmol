from toolz import first, valmap

import attr
import torch
import numba.cuda

import inspect

# monkey-patch torch tensor support
import tmol.utility.numba

from tmol.types.torch import Tensor
from tmol.types.functional import validate_args

from .params import LJLKParamResolver
from . import numba_potential

import typing


@attr.s(auto_attribs=True, frozen=True)
class LJOp:
    device: torch.device
    params: typing.Mapping[str, typing.Any]
    kernel_signature: inspect.Signature
    parallel_cpu: bool = True

    @staticmethod
    def _prepare_input(tensor, for_device):
        if for_device.type == "cpu":
            return tensor.__array__()
        elif for_device.type == "cuda":
            return numba.cuda.as_cuda_array(tensor)
        else:
            raise ValueError(f"Invalid target device: {for_device}")

    def pairwise(self, a_coords, a_types, b_coords, b_types, a_b_bonded_path_length):
        return LJPairwiseFun(self)(
            a_coords, a_types, b_coords, b_types, a_b_bonded_path_length
        )

    @classmethod
    def from_params(cls, param_resolver: LJLKParamResolver):

        raw_params = dict(
            lj_sigma=param_resolver.pair_params.lj_sigma,
            lj_switch_slope=param_resolver.pair_params.lj_switch_slope,
            lj_switch_intercept=param_resolver.pair_params.lj_switch_intercept,
            lj_coeff_sigma12=param_resolver.pair_params.lj_coeff_sigma12,
            lj_coeff_sigma6=param_resolver.pair_params.lj_coeff_sigma6,
            lj_spline_y0=param_resolver.pair_params.lj_spline_y0,
            lj_spline_dy0=param_resolver.pair_params.lj_spline_dy0,
            # Global param_resolver
            lj_switch_dis2sigma=param_resolver.global_params.lj_switch_dis2sigma.reshape(
                1
            ),
            spline_start=param_resolver.global_params.spline_start.reshape(1),
            max_dis=param_resolver.global_params.max_dis.reshape(1),
        )

        assert (
            len(set((p.device.type, p.device.index) for p in raw_params.values())) == 1
        )

        device = first(raw_params.values()).device

        params = valmap(lambda t: cls._prepare_input(t, device), raw_params)

        kernel_signature = inspect.signature(numba_potential.lj_kernel_cuda.py_func)

        return cls(device=device, params=params, kernel_signature=kernel_signature)


class LJPairwiseFun(torch.autograd.Function):
    def __init__(self, op: LJOp):
        self.op = op
        super().__init__()

    def forward(ctx, a_coords, a_types, b_coords, b_types, a_b_bonded_path_length):
        assert a_coords.device == ctx.op.device
        assert not a_coords.requires_grad

        assert a_types.device == ctx.op.device
        assert not a_types.requires_grad

        assert b_coords.device == ctx.op.device
        assert not b_coords.requires_grad

        assert b_types.device == ctx.op.device
        assert not b_types.requires_grad

        assert a_b_bonded_path_length.device == ctx.op.device
        assert not a_b_bonded_path_length.requires_grad

        result = a_coords.new_empty((a_coords.shape[0], b_coords.shape[0]))

        if ctx.op.device.type == "cpu":
            if ctx.op.parallel_cpu:
                kernel = numba_potential.lj_kernel_parallel_cpu
            else:
                kernel = numba_potential.lj_kernel_serial_cpu

            kernel(
                a_coords.__array__(),
                a_types.__array__(),
                b_coords.__array__(),
                b_types.__array__(),
                a_b_bonded_path_length.__array__(),
                result.__array__(),
                **ctx.op.params,
            )
        else:
            blocks_per_grid = (
                (a_coords.shape[0] // 32) + 1,
                (b_coords.shape[0] // 32) + 1,
            )
            threads_per_block = (32, 32)

            numba_potential.lj_kernel_cuda[blocks_per_grid, threads_per_block](
                *ctx.op.kernel_signature.bind(
                    a_coords,
                    a_types,
                    b_coords,
                    b_types,
                    a_b_bonded_path_length,
                    result,
                    **ctx.op.params,
                ).args
            )

        return result
