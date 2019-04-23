import torch
from tmol.score.ljlk.params import LJLKParamResolver

from tmol.database.chemical import ChemicalDatabase
from tmol.database.scoring.ljlk import LJLKDatabase

# Import compiled components to load torch_ops
import tmol.score.ljlk.potentials.compiled  # noqa

# Workaround for https://github.com/pytorch/pytorch/pull/15340
# on torch<1.0.1
if "to" in torch.jit.ScriptModule.__dict__:
    delattr(torch.jit.ScriptModule, "to")


class _LJScoreModule(torch.jit.ScriptModule):
    @classmethod
    def from_database(
        cls,
        chemical_database: ChemicalDatabase,
        ljlk_database: LJLKDatabase,
        device: torch.device,
    ):
        return cls(
            param_resolver=LJLKParamResolver.from_database(
                chemical_database, ljlk_database, device
            )
        )

    def __init__(self, param_resolver: LJLKParamResolver):

        super().__init__()

        def _p(t):
            return torch.nn.Parameter(t, requires_grad=False)

        def _t(ts):
            return tuple(map(lambda t: t.to(torch.float), ts))

        # Pack parameters into dense tensor. Parameter ordering must match
        # struct layout declared in `potentials/params.hh`.
        self.type_params = _p(
            torch.stack(
                _t(
                    [
                        param_resolver.type_params.lj_radius,
                        param_resolver.type_params.lj_wdepth,
                        param_resolver.type_params.is_donor,
                        param_resolver.type_params.is_hydroxyl,
                        param_resolver.type_params.is_polarh,
                        param_resolver.type_params.is_acceptor,
                    ]
                ),
                dim=1,
            )
        )

        self.global_params = _p(
            torch.stack(
                _t(
                    [
                        param_resolver.global_params.lj_hbond_dis,
                        param_resolver.global_params.lj_hbond_OH_donor_dis,
                        param_resolver.global_params.lj_hbond_hdis,
                    ]
                ),
                dim=1,
            )
        )


class LJIntraModule(_LJScoreModule):
    @torch.jit.script_method
    def forward(self, I, atom_type_I, bonded_path_lengths):
        return torch.ops.tmol.score_ljlk_lj_triu(
            I,
            atom_type_I,
            I,
            atom_type_I,
            bonded_path_lengths,
            self.type_params,
            self.global_params,
        )


class LJInterModule(_LJScoreModule):
    @torch.jit.script_method
    def forward(self, I, atom_type_I, J, atom_type_J, bonded_path_lengths):
        return torch.ops.tmol.score_ljlk_lj(
            I,
            atom_type_I,
            J,
            atom_type_J,
            bonded_path_lengths,
            self.type_params,
            self.global_params,
        )


class _LKIsotropicScoreModule(torch.jit.ScriptModule):
    @classmethod
    def from_database(
        cls,
        chemical_database: ChemicalDatabase,
        ljlk_database: LJLKDatabase,
        device: torch.device,
    ):
        return cls(
            param_resolver=LJLKParamResolver.from_database(
                chemical_database, ljlk_database, device
            )
        )

    def __init__(self, param_resolver: LJLKParamResolver):

        super().__init__()

        def _p(t):
            return torch.nn.Parameter(t, requires_grad=False)

        def _t(ts):
            return tuple(map(lambda t: t.to(torch.float), ts))

        # Pack parameters into dense tensor. Parameter ordering must match
        # struct layout declared in `potentials/params.hh`.
        self.type_params = _p(
            torch.stack(
                _t(
                    [
                        param_resolver.type_params.lj_radius,
                        param_resolver.type_params.lk_dgfree,
                        param_resolver.type_params.lk_lambda,
                        param_resolver.type_params.lk_volume,
                        param_resolver.type_params.is_donor,
                        param_resolver.type_params.is_hydroxyl,
                        param_resolver.type_params.is_polarh,
                        param_resolver.type_params.is_acceptor,
                    ]
                ),
                dim=1,
            )
        )

        self.global_params = _p(
            torch.stack(
                _t(
                    [
                        param_resolver.global_params.lj_hbond_dis,
                        param_resolver.global_params.lj_hbond_OH_donor_dis,
                        param_resolver.global_params.lj_hbond_hdis,
                    ]
                ),
                dim=1,
            )
        )


class LKIsotropicIntraModule(_LKIsotropicScoreModule):
    @torch.jit.script_method
    def forward(self, I, atom_type_I, bonded_path_lengths):
        return torch.ops.tmol.score_ljlk_lk_isotropic_triu(
            I,
            atom_type_I,
            I,
            atom_type_I,
            bonded_path_lengths,
            self.type_params,
            self.global_params,
        )


class LKIsotropicInterModule(_LKIsotropicScoreModule):
    @torch.jit.script_method
    def forward(self, I, atom_type_I, J, atom_type_J, bonded_path_lengths):
        return torch.ops.tmol.score_ljlk_lk_isotropic(
            I,
            atom_type_I,
            J,
            atom_type_J,
            bonded_path_lengths,
            self.type_params,
            self.global_params,
        )
