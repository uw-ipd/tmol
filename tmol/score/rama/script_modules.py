import torch

import attr
from attr import asdict
from typing import Mapping, Callable

from .params import RamaDatabase, RamaParamResolver, RamaParams

# Import compiled components to load torch_ops
import tmol.score.rama.potentials.compiled  # noqa

# Workaround for https://github.com/pytorch/pytorch/pull/15340
# on torch<1.0.1
if "to" in torch.jit.ScriptModule.__dict__:
    delattr(torch.jit.ScriptModule, "to")


class RamaScoreModule(torch.jit.ScriptModule):
    @torch.jit.script_method
    def forward(self, coords, atoms):
        return torch.ops.tmol.score_cartbonded_length(
            coords, atoms, self.bondlength_params
        )

    def __init__(self, params: RamaParams, param_resolver: RamaParamResolver):
        super().__init__()

        def _p(t):
            return torch.nn.Parameter(t, requires_grad=False)

        def _tint(ts):
            return tuple(map(lambda t: t.to(torch.int32), ts))

        def _tfloat(ts):
            return tuple(map(lambda t: t.to(torch.float), ts))

        self.params = _p(
            torch.cat(
                _tint(
                    [
                        params.phi_indices,
                        params.psi_indices,
                        params.param_indices[:, None],
                    ]
                ),
                dim=1,
            )
        )

        self.tables = _p(param_resolver.rama_params.tables)

        self.table_params = _p(
            torch.cat(
                _tfloat(
                    [
                        param_resolver.rama_params.bbstarts,
                        param_resolver.rama_params.bbsteps,
                    ]
                ),
                dim=1,
            )
        )

    @torch.jit.script_method
    def forward(self, coords):
        return torch.ops.tmol.score_rama(
            coords, self.params, self.tables, self.table_params
        )
