from toolz import first, valmap

import attr
import torch
import numba.cuda

import inspect

# monkey-patch torch tensor support
import tmol.utility.numba  # noqa

from .params import LJLKParamResolver
from . import numba_potential
from . import cpp_potential

import typing


@attr.s(auto_attribs=True, frozen=True)
class LJOp:
    device: torch.device
    raw_params: typing.Mapping[str, typing.Any]
    params: typing.Mapping[str, typing.Any]
    kernel_signature: inspect.Signature
    parallel_cpu: bool
    jit_type: str

    @staticmethod
    def _prepare_input(tensor, for_device):
        if for_device.type == "cpu":
            return tensor.__array__()
        elif for_device.type == "cuda":
            return numba.cuda.as_cuda_array(tensor)
        else:
            raise ValueError(f"Invalid target device: {for_device}")

    def intra(self, coords, types, bonded_path_length, block_distances=None):
        if self.jit_type == "numba":
            if block_distances is not None:
                return LJIntraNumbaFun(self)(
                    coords, types, bonded_path_length, block_distances
                )
            else:
                return LJIntraNumbaFun(self)(coords, types, bonded_path_length)
        elif self.jit_type == "cpp":
            if block_distances is not None:
                return LJIntraCppFun(self)(
                    coords, types, bonded_path_length, block_distances
                )
            else:
                return LJIntraCppFun(self)(coords, types, bonded_path_length)

    @classmethod
    def from_params(
        cls, param_resolver: LJLKParamResolver, parallel_cpu=True, jit_type="numba"
    ):

        pair_params = param_resolver.pair_params
        global_params = param_resolver.global_params

        raw_params = dict(
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

        assert (
            len(set((p.device.type, p.device.index) for p in raw_params.values())) == 1
        )

        device = first(raw_params.values()).device

        params = valmap(lambda t: cls._prepare_input(t, device), raw_params)

        kernel_signature = inspect.signature(
            numba_potential.lj_intra_kernel_cuda.py_func
        )

        return cls(
            device=device,
            raw_params=raw_params,
            params=params,
            kernel_signature=kernel_signature,
            parallel_cpu=parallel_cpu,
            jit_type=jit_type,
        )


class LJIntraCppFun(torch.autograd.Function):
    def __init__(self, op: LJOp):
        self.op = op
        super().__init__()

    def forward(ctx, coords, types, bonded_path_length, block_distances=None):

        assert coords.device == ctx.op.device
        assert not coords.requires_grad

        assert types.device == ctx.op.device
        assert not types.requires_grad

        assert bonded_path_length.device == ctx.op.device
        assert not bonded_path_length.requires_grad

        return cpp_potential.lj_intra(
            coords, types, bonded_path_length, **ctx.op.raw_params
        )


class LJIntraNumbaFun(torch.autograd.Function):
    def __init__(self, op: LJOp):
        self.op = op
        super().__init__()

    def forward(ctx, coords, types, bonded_path_length, block_distances=None):
        nblocks = coords.shape[0] // numba_potential.BLOCK_SIZE

        assert coords.device == ctx.op.device
        assert not coords.requires_grad

        assert types.device == ctx.op.device
        assert not types.requires_grad

        assert bonded_path_length.device == ctx.op.device
        assert not bonded_path_length.requires_grad

        result = coords.new_zeros((coords.shape[0], coords.shape[0]))

        # if block_distances is None:
        block_distances = coords.new_zeros((nblocks, nblocks))

        if ctx.op.device.type == "cpu":
            if ctx.op.parallel_cpu:
                kernel = numba_potential.lj_intra_kernel_parallel_cpu
            else:
                kernel = numba_potential.lj_intra_kernel_serial_cpu

            kernel(
                coords.__array__(),
                types.__array__(),
                bonded_path_length.__array__(),
                block_distances.__array__(),
                result.__array__(),
                **ctx.op.params,
            )
        else:
            bdim = 16
            blocks_per_grid = (nblocks // bdim) + (1 if nblocks % bdim else 0)
            threads_per_block = bdim

            numba_potential.lj_intra_kernel_cuda[
                (blocks_per_grid, blocks_per_grid),
                (threads_per_block, threads_per_block),
            ](
                *ctx.op.kernel_signature.bind(
                    coords,
                    types,
                    bonded_path_length,
                    block_distances,
                    result,
                    **ctx.op.params,
                ).args
            )

        return result
