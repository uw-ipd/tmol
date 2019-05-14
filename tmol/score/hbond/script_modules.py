import attr

from typing import Mapping, Union, Callable

import torch

from tmol.utility.dicttoolz import flat_items, merge

from tmol.database.scoring import HBondDatabase
from .params import HBondParamResolver

from tmol.utility.nvtx import nvtx_range
import tmol.score.hbond.potentials.compiled  # noqa


# Workaround for https://github.com/pytorch/pytorch/pull/15340
# on torch<1.0.1
if "to" in torch.jit.ScriptModule.__dict__:
    delattr(torch.jit.ScriptModule, "to")


class _HBondScoreModule(torch.jit.ScriptModule):
    """torch.autograd hbond baseline operator."""

    # params: Mapping[str, Union[float, torch.Tensor]]
    # hbond_pair_score: Callable = attr.ib()

    # @hbond_pair_score.default
    # def _load_hbond_pair_score(self):
    #    from .potentials.compiled import hbond_pair_score
    #
    #    return hbond_pair_score

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

    def __init__(self, param_dict):
        print("_HBondScoreModule.__init__")
        super().__init__()

        database = param_dict["database"]
        param_resolver = param_dict["param_resolver"]
        pair_params = _HBondScoreModule._setup_pair_params(
            param_resolver, dtype=torch.float32
        )

        global_params = {
            n: torch.tensor(v, device=param_resolver.device)
            .expand(1)
            .to(dtype=torch.float32)
            for n, v in attr.asdict(database.global_parameters).items()
        }

        self.acceptor_hybridization = torch.nn.Parameter(
            torch.zeros(5), requires_grad=False
        )
        for n, v in merge(pair_params, global_params).items():
            # print("setting attribute", n)
            setattr(self, n, torch.nn.Parameter(v, requires_grad=False))

        pad = torch.zeros(
            [self.AHdist_coeffs.shape[0], self.AHdist_coeffs.shape[1], 1],
            device=self.AHdist_coeffs.device,
            dtype=torch.float64,
        )
        self.AHdist_poly = torch.nn.Parameter(
            torch.cat(
                [self.AHdist_coeffs, pad, self.AHdist_range, self.AHdist_bound], 2
            ),
            requires_grad=False,
        )
        print("AHdist_poly", self.AHdist_poly.shape, self.AHdist_poly.stride())

        print("self.AHdist_coeffs[0,0,:]", self.AHdist_coeffs[0, 0, :])
        print("self.AHdist_range[0,0,:]", self.AHdist_range[0, 0, :])
        print("self.AHdist_bound[0,0,:]", self.AHdist_bound[0, 0, :])
        print("self.AHdist_poly[0,0,:]", self.AHdist_poly[0, 0, :])


class HBondIntraModule(_HBondScoreModule):
    def __init__(self, param_dict):
        super().__init__(param_dict)

    @torch.jit.script_method
    def forward(
        self, donor_coords, acceptor_coords, D, H, donor_type, A, B, B0, acceptor_type
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
            self.acceptor_hybridization,
            self.acceptor_weight,
            self.donor_weight,
            self.AHdist_poly,
            self.cosBAH_coeffs,
            self.cosBAH_range,
            self.cosBAH_bound,
            self.cosAHD_coeffs,
            self.cosAHD_range,
            self.cosAHD_bound,
            self.hb_sp2_range_span,
            self.hb_sp2_BAH180_rise,
            self.hb_sp2_outer_width,
            self.hb_sp3_softmax_fade,
            self.threshold_distance,
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
