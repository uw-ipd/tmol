import attr

from typing import Mapping, Union, Callable

import torch

from tmol.utility.dicttoolz import flat_items, merge

from tmol.database.scoring import HBondDatabase
from .params import HBondParamResolver

from tmol.utility.nvtx import nvtx_range


@attr.s(auto_attribs=True, frozen=True)
class HBondOp:
    """torch.autograd hbond baseline operator."""

    params: Mapping[str, Union[float, torch.Tensor]]
    hbond_pair_score: Callable = attr.ib()

    @hbond_pair_score.default
    def _load_hbond_pair_score(self):
        from .potentials.compiled import hbond_pair_score

        return hbond_pair_score

    @staticmethod
    def _setup_pair_params(param_resolver, dtype):
        def _t(n, v):
            t = torch.tensor(v)
            if t.is_floating_point():
                if any(dkey in n for dkey in ("range", "bound", "coeffs")):
                    # High degree polynomial parameters stored as double precision
                    # to allow accurate double evaluation.
                    t = t.to(torch.float64)
                else:
                    t = t.to(dtype)
            return t

        return {
            "_".join(k): _t(k, v)
            for k, v in flat_items(attr.asdict(param_resolver.pair_params))
        }

    @classmethod
    def from_database(
        cls,
        database: HBondDatabase,
        param_resolver: HBondParamResolver,
        dtype=torch.float32,
    ):

        pair_params = cls._setup_pair_params(param_resolver, dtype)

        global_params = {
            n: torch.tensor(v, device=param_resolver.device).expand(1).to(dtype)
            for n, v in attr.asdict(database.global_parameters).items()
        }

        return cls(params=merge(pair_params, global_params))

    def score(
        self, donor_coords, acceptor_coords, D, H, donor_type, A, B, B0, acceptor_type
    ):
        score = HBondFun(self)(
            donor_coords, acceptor_coords, D, H, donor_type, A, B, B0, acceptor_type
        )
        return score


class HBondFun(torch.autograd.Function):
    def __init__(self, op):
        self.op = op
        super().__init__()

    def forward(
        ctx, donor_coords, acceptor_coords, D, H, donor_type, A, B, B0, acceptor_type
    ):
        with nvtx_range("HBondFun.forward.assert"):
            assert D.dim() == 1
            assert D.shape == H.shape
            assert D.shape == donor_type.shape

            assert A.dim() == 1
            assert A.shape == B.shape
            assert A.shape == B0.shape
            assert A.shape == acceptor_type.shape

        with nvtx_range("HBondFun.forward.pair_score"):
            E, *dE_d_coords = ctx.op.hbond_pair_score(
                donor_coords,
                acceptor_coords,
                D,
                H,
                donor_type,
                A,
                B,
                B0,
                acceptor_type,
                **ctx.op.params,
            )

        ctx.save_for_backward(*dE_d_coords)

        return E

    def backward(ctx, dV_dE):
        with nvtx_range("HBondFun.backward"):
            dE_d_don, dE_d_acc = ctx.saved_tensors

            return (
                dV_dE * dE_d_don,
                dV_dE * dE_d_acc,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
