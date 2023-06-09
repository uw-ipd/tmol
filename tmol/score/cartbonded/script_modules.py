import torch

from tmol.score.cartbonded.params import CartBondedParamResolver
from tmol.database.scoring import CartBondedDatabase

# Import compiled components to load torch_ops
from tmol.score.cartbonded.potentials.compiled import score_cartbonded


# Workaround for https://github.com/pytorch/pytorch/pull/15340
# on torch<1.0.1
if "to" in torch.jit.ScriptModule.__dict__:
    delattr(torch.jit.ScriptModule, "to")


class CartBondedModule(torch.jit.ScriptModule):
    def __init__(self, param_resolver: CartBondedParamResolver):
        super().__init__()

        def _p(t):
            return torch.nn.Parameter(t, requires_grad=False)

        def _t(ts):
            return tuple(map(lambda t: t.to(torch.float), ts))

        self.bondlength_params = _p(
            torch.stack(
                _t(
                    [
                        param_resolver.bondlength_params.K,
                        param_resolver.bondlength_params.x0,
                    ]
                ),
                dim=1,
            )
        )

        self.bondangle_params = _p(
            torch.stack(
                _t(
                    [
                        param_resolver.bondangle_params.K,
                        param_resolver.bondangle_params.x0,
                    ]
                ),
                dim=1,
            )
        )

        self.torsion_params = _p(
            torch.stack(
                _t(
                    [
                        param_resolver.torsion_params.K,
                        param_resolver.torsion_params.x0,
                        param_resolver.torsion_params.period,
                    ]
                ),
                dim=1,
            )
        )

        self.improper_params = _p(
            torch.stack(
                _t(
                    [
                        param_resolver.improper_params.K,
                        param_resolver.improper_params.x0,
                        param_resolver.improper_params.period,
                    ]
                ),
                dim=1,
            )
        )

        self.hxltorsion_params = _p(
            torch.stack(
                _t(
                    [
                        param_resolver.hxltorsion_params.k1,
                        param_resolver.hxltorsion_params.k2,
                        param_resolver.hxltorsion_params.k3,
                        param_resolver.hxltorsion_params.phi1,
                        param_resolver.hxltorsion_params.phi2,
                        param_resolver.hxltorsion_params.phi3,
                    ]
                ),
                dim=1,
            )
        )

    @classmethod
    def from_database(cls, cb_database: CartBondedDatabase, device: torch.device):
        return cls(
            param_resolver=CartBondedParamResolver.from_database(cb_database, device)
        )

    @torch.jit.script_method
    def forward(self, coords, cbl_atoms, cba_atoms, cbt_atoms, cbi_atoms, cbhxl_atoms):
        return score_cartbonded(
            coords,
            cbl_atoms,
            cba_atoms,
            cbt_atoms,
            cbi_atoms,
            cbhxl_atoms,
            self.bondlength_params,
            self.bondangle_params,
            self.torsion_params,
            self.improper_params,
            self.hxltorsion_params,
        )
