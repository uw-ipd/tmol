import attr
from typing import Callable

import torch

from tmol.types.functional import validate_args

from .params import LJLKDatabase, LJLKParamResolver

from tmol.utility.nvtx import nvtx_range


@attr.s(auto_attribs=True, frozen=True)
class AtomOp:
    """torch.autograd atom pair score baseline operator."""

    param_resolver: LJLKParamResolver

    @classmethod
    @validate_args
    def from_database(
        cls, chemical_database, ljlk_database: LJLKDatabase, device: torch.device
    ):
        return cls(
            param_resolver=LJLKParamResolver.from_database(
                chemical_database, ljlk_database, device
            )
        )

    def inter(
        self, coords_a, atom_types_a, coords_b, atom_types_b, bonded_path_lengths
    ):
        return _AtomScoreFun(self, self.f)(
            coords_a, atom_types_a, coords_b, atom_types_b, bonded_path_lengths
        )

    def intra(self, coords, atom_types, bonded_path_lengths):
        # Call triu score function on coords, results in idependent autograd
        # paths for i and j axis
        return _AtomScoreFun(self, self.f_triu)(
            coords, atom_types, coords, atom_types, bonded_path_lengths
        )


class _AtomScoreFun(torch.autograd.Function):
    def __init__(self, op, f):
        self.op = op
        self.f = f
        super().__init__()

    def forward(ctx, I, atom_type_I, J, atom_type_J, bonded_path_lengths):
        with nvtx_range("_AtomScoreFun.forward"):

            assert I.dim() == 2
            assert I.shape[1] == 3
            assert I.shape[:1] == atom_type_I.shape
            assert not atom_type_I.requires_grad

            assert J.dim() == 2
            assert J.shape[1] == 3
            assert J.shape[:1] == atom_type_J.shape
            assert not atom_type_J.requires_grad

            with nvtx_range("cast_params"):
                # Cast parameter tensors to precision required for input tensors.
                # Required to support double-precision inputs for gradcheck tests,
                # inputs/params will be single-precision in standard case.
                type_params = ctx.op.param_resolver.type_params
                global_params = ctx.op.param_resolver.global_params

                if I.dtype != type_params.lj_radius.dtype:
                    type_params = attr.evolve(
                        ctx.op.param_resolver.type_params,
                        **{
                            n: t.to(I.dtype) if t.is_floating_point() else t
                            for n, t in attr.asdict(type_params).items()
                        },
                    )

                    global_params = attr.evolve(
                        ctx.op.param_resolver.global_params,
                        **{
                            n: t.to(I.dtype) if t.is_floating_point() else t
                            for n, t in attr.asdict(global_params).items()
                        },
                    )

            with nvtx_range("ctx.f"):
                E, *dE = ctx.f(
                    I,
                    atom_type_I,
                    J,
                    atom_type_J,
                    bonded_path_lengths,
                    type_params,
                    global_params,
                )

            ctx.save_for_backward(*dE)

            return E

    def backward(ctx, dV_dE):
        with nvtx_range("_AtomScoreFun.backward"):
            dE_dI, dE_dJ = ctx.saved_tensors
            return dV_dE * dE_dI, None, dV_dE * dE_dJ, None, None


@attr.s(auto_attribs=True, frozen=True)
class LJOp(AtomOp):
    f: Callable = attr.ib()
    f_triu: Callable = attr.ib()

    @f.default
    def _load_f(self):
        from .potentials import compiled

        return compiled.lj

    @f_triu.default
    def _load_f_triu(self):
        from .potentials import compiled

        return compiled.lj_triu


@attr.s(auto_attribs=True, frozen=True)
class LKOp(AtomOp):
    """torch.autograd hbond baseline operator."""

    f: Callable = attr.ib()
    f_triu: Callable = attr.ib()

    @f.default
    def _load_f(self):
        from .potentials import compiled

        return compiled.lk_isotropic

    @f_triu.default
    def _load_f_triu(self):
        from .potentials import compiled

        return compiled.lk_isotropic_triu
