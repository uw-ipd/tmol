import attr

from typing import Mapping, Union, Callable

import torch

from tmol.utility.dicttoolz import flat_items, merge

from tmol.database.scoring import HBondDatabase
from .params import HBondParamResolver, CompactedHBondDatabase

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

    def __init__(self, compact_db: CompactedHBondDatabase):
        super().__init__()
        self.pair_param_table = compact_db.pair_param_table
        self.pair_poly_table = compact_db.pair_poly_table
        self.global_param_table = compact_db.global_param_table


class HBondIntraModule(_HBondScoreModule):
    def __init__(self, compact_db: CompactedHBondDatabase):
        super().__init__(compact_db)

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
            self.pair_param_table,
            self.pair_poly_table,
            self.global_param_table,
        )
