import attr

from typing import Mapping, Union, Callable

import torch

from tmol.utility.dicttoolz import flat_items, merge

from tmol.database.scoring import HBondDatabase
from .params import HBondParamResolver

from tmol.utility.nvtx import nvtx_range


@attr.s(auto_attribs=True, frozen=True)
class _HBondScoreModule(torch.jit.ScriptModule):
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

    def __init__(self, database: HBondDatabase, param_resolver: HBondParamResolver):
        super().__init__()

        pair_params = _HBondScoreModule._setup_pair_params(param_resolver, dtype)

        global_params = {
            n: torch.tensor(v, device=param_resolver.device).expand(1).to(dtype)
            for n, v in attr.asdict(database.global_parameters).items()
        }

        self.params = merge(pair_params, global_params)


class HBondIntraModule(_HBondScoreModule):
    @torch.jit.script_method
    def forward(
        ctx, donor_coords, acceptor_coords, D, H, donor_type, A, B, B0, acceptor_type
    ):
        return torch.ops.tmol.score_hbond(
            donor_coords,
            acceptor_coords,
            D,
            H,
            donor_type,
            A,
            B,
            B0,
            acceptor_type,
            ctx.params.donor_weight,
            ctx.params.acceptor_weight,
            ctx.params.acceptor_hybridization,
            ctx.params.AHdist_range,
            ctx.params.AHdist_bound,
            ctx.params.AHdist_coeffs,
            ctx.params.cosBAH_range,
            ctx.params.cosBAH_bound,
            ctx.params.cosBAH_coeffs,
            ctx.params.cosAHD_range,
            ctx.params.cosAHD_bound,
            ctx.params.cosAHD_coeffs,
        )


# ?? class HBondInterModule(_HBondScoreModule):
# ??     @torch.jit.script_method
# ??     def forward(
# ??         ctx, donor_coords, acceptor_coords, D, H, donor_type, A, B, B0, acceptor_type
# ??     ):
# ??         return torch.ops.tmol.score_hbond(
# ??             donor_coords,
# ??             acceptor_coords,
# ??             D,
# ??             H,
# ??             donor_type,
# ??             A,
# ??             B,
# ??             B0,
# ??             acceptor_type,
# ??             **ctx.params,
# ??         )
