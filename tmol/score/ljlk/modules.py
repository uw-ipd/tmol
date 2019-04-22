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


class LJIntraModule(torch.jit.ScriptModule):
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

        self.lj_radius = _p(param_resolver.type_params.lj_radius)
        self.lj_wdepth = _p(param_resolver.type_params.lj_wdepth)
        self.is_donor = _p(param_resolver.type_params.is_donor)
        self.is_hydroxyl = _p(param_resolver.type_params.is_hydroxyl)
        self.is_polarh = _p(param_resolver.type_params.is_polarh)
        self.is_acceptor = _p(param_resolver.type_params.is_acceptor)

        self.lj_hbond_dis = _p(param_resolver.global_params.lj_hbond_dis)
        self.lj_hbond_OH_donor_dis = _p(
            param_resolver.global_params.lj_hbond_OH_donor_dis
        )
        self.lj_hbond_hdis = _p(param_resolver.global_params.lj_hbond_hdis)

    @torch.jit.script_method
    def forward(self, I, atom_type_I, bonded_path_lengths):
        return torch.ops.tmol.score_ljlk_lj_triu(
            I,
            atom_type_I,
            I,
            atom_type_I,
            bonded_path_lengths,
            [
                self.lj_radius,
                self.lj_wdepth,
                self.is_donor,
                self.is_hydroxyl,
                self.is_polarh,
                self.is_acceptor,
            ],
            [self.lj_hbond_dis, self.lj_hbond_OH_donor_dis, self.lj_hbond_hdis],
        )
