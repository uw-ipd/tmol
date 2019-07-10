import torch

from tmol.score.cartbonded.params import CartBondedParamResolver
from tmol.database.scoring import CartBondedDatabase

from tmol.utility.cuda.synchronize import synchronize_if_cuda_available

# Import compiled components to load torch_ops
import tmol.score.cartbonded.potentials.compiled  # noqa


# Workaround for https://github.com/pytorch/pytorch/pull/15340
# on torch<1.0.1
if "to" in torch.jit.ScriptModule.__dict__:
    delattr(torch.jit.ScriptModule, "to")


def _p(t):
    return torch.nn.Parameter(t, requires_grad=False)


def _t(ts):
    return tuple(map(lambda t: t.to(torch.float), ts))


class _CartBondedScoreModule(torch.jit.ScriptModule):
    @classmethod
    def from_database(cls, cb_database: CartBondedDatabase, device: torch.device):
        return cls(
            param_resolver=CartBondedParamResolver.from_database(cb_database, device)
        )


class CartBondedLengthModule(_CartBondedScoreModule):
    @torch.jit.script_method
    def forward(self, coords, atoms):
        """Non-blocking score evaluation"""
        return torch.ops.tmol.score_cartbonded_length(
            coords, atoms, self.bondlength_params
        )

    def final(self, coords, atoms):
        """Blocking score evaluation"""
        res = self(coords, atoms)
        synchronize_if_cuda_available()
        return res

    def __init__(self, param_resolver: CartBondedParamResolver):
        super().__init__()

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


class CartBondedAngleModule(_CartBondedScoreModule):
    @torch.jit.script_method
    def forward(self, coords, atoms):
        """Non-blocking score evaluation"""
        return torch.ops.tmol.score_cartbonded_angle(
            coords, atoms, self.bondangle_params
        )

    def final(self, coords, atoms):
        """Blocking score evaluation"""
        res = self(coords, atoms)
        synchronize_if_cuda_available()
        return res

    def __init__(self, param_resolver: CartBondedParamResolver):
        super().__init__()

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


class CartBondedTorsionModule(_CartBondedScoreModule):
    @torch.jit.script_method
    def forward(self, coords, atoms):
        """Non-blocking score evaluation"""
        return torch.ops.tmol.score_cartbonded_torsion(
            coords, atoms, self.torsion_params
        )

    def final(self, coords, atoms):
        """Blocking score evaluation"""
        res = self(coords, atoms)
        synchronize_if_cuda_available()
        return res

    def __init__(self, param_resolver: CartBondedParamResolver):
        super().__init__()

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


class CartBondedImproperModule(_CartBondedScoreModule):
    @torch.jit.script_method
    def forward(self, coords, atoms):
        """Non-blocking score evaluation"""
        return torch.ops.tmol.score_cartbonded_torsion(
            coords, atoms, self.improper_params
        )

    def final(self, coords, atoms):
        """Blocking score evaluation"""
        res = self(coords, atoms)
        synchronize_if_cuda_available()
        return res

    def __init__(self, param_resolver: CartBondedParamResolver):
        super().__init__()

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


class CartBondedHxlTorsionModule(_CartBondedScoreModule):
    @torch.jit.script_method
    def forward(self, coords, atoms):
        """Non-blocking score evaluation"""
        return torch.ops.tmol.score_cartbonded_hxltorsion(
            coords, atoms, self.hxltorsion_params
        )

    def final(self, coords, atoms):
        """Blocking score evaluation"""
        res = self(coords, atoms)
        synchronize_if_cuda_available()
        return res

    def __init__(self, param_resolver: CartBondedParamResolver):
        super().__init__()

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
