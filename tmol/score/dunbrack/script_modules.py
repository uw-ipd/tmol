import attr
from attr import asdict
from typing import Mapping, Callable

from .params import DunbrackParamResolver, DunbrackParams

# Import compiled components to load torch_ops
import tmol.score.dunbrack.potentials.compiled  # noqa

import torch


class DunbrackScoreModule(torch.jit.ScriptModule):
    def __init__(self, params: DunbrackParams, param_resolver: DunbrackParamResolver):
        super().__init__()

        def _p(t):
            return torch.nn.Parameter(t, requires_grad=False)

        def _tint(ts):
            return tuple(map(lambda t: t.to(torch.int32), ts))

        def _tfloat(ts):
            return tuple(map(lambda t: t.to(torch.float), ts))

        self.rotameric_tables = _p(param_resolver.packed_db.rotameric_prob_tables)

        self.rotameric_table_params = _p(
            torch.cat(
                _tfloat(
                    [
                        param_resolver.packed_db.rotameric_bbstarts,
                        param_resolver.packed_db.rotameric_bbsteps,
                    ]
                ),
                dim=1,
            )
        )

        self.semirotameric_tables = _p(
            param_resolver.packed_db.semirotameric_prob_tables
        )
        self.semirotameric_table_params = _p(
            torch.cat(
                _tfloat(
                    [
                        param_resolver.packed_db.semirotameric_bbstarts,
                        param_resolver.packed_db.semirotameric_bbsteps,
                    ]
                ),
                dim=1,
            )
        )

        ndunres = params.aa_indices.shape[0]
        self.residue_params = _p(
            torch.cat(
                _tint(
                    [
                        params.bb_indices.reshape(ndunres, -1),
                        params.chi_indices.reshape(ndunres, -1),
                        params.aa_indices.reshape(ndunres, -1),
                    ]
                ),
                dim=1,
            )
        )

        self.residue_lookup_params = _p(
            torch.cat(
                _tint(
                    [
                        param_resolver.packed_db.rotind2rotprobind,
                        param_resolver.packed_db.rotind2rotmeanind,
                        param_resolver.packed_db.rotind2semirotprobind,
                        param_resolver.packed_db.nrotchi_aa[:, None],
                        param_resolver.packed_db.semirotchi_aa[:, None],
                    ]
                ),
                dim=1,
            )
        )

    @torch.jit.script_method
    def forward(self, coords):
        return torch.ops.tmol.score_dun(
            coords,
            self.rotameric_tables,
            self.rotameric_table_params,
            self.semirotameric_tables,
            self.semirotameric_table_params,
            self.residue_params,
            self.residue_lookup_params,
        )
