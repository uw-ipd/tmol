import torch

from .params import ElecParamResolver

from tmol.score.elec.potentials.compiled import score_elec, score_elec_triu

# Workaround for https://github.com/pytorch/pytorch/pull/15340
# on torch<1.0.1
if "to" in torch.jit.ScriptModule.__dict__:
    delattr(torch.jit.ScriptModule, "to")


class _ElecScoreModule(torch.jit.ScriptModule):
    def __init__(self, param_resolver: ElecParamResolver):

        super().__init__()

        # Pack parameters into dense tensor. Parameter ordering must match
        # struct layout declared in `potentials/params.hh`.
        self.global_params = torch.nn.Parameter(
            torch.tensor(
                [
                    [
                        param_resolver.global_params.elec_sigmoidal_die_D,
                        param_resolver.global_params.elec_sigmoidal_die_D0,
                        param_resolver.global_params.elec_sigmoidal_die_S,
                        param_resolver.global_params.elec_min_dis,
                        param_resolver.global_params.elec_max_dis,
                    ]
                ],
                device=param_resolver.device,
                dtype=torch.float,
            ),
            requires_grad=False,
        )


class ElecInterModule(_ElecScoreModule):
    @torch.jit.script_method
    def forward(self, I, atom_type_I, J, atom_type_J, bonded_path_lengths):
        return score_elec(
            I, atom_type_I, J, atom_type_J, bonded_path_lengths, self.global_params
        )


class ElecIntraModule(_ElecScoreModule):
    @torch.jit.script_method
    def forward(self, I, atom_type_I, bonded_path_lengths):
        # print("I", I.shape)
        # print("atom_type_I", atom_type_I.shape)
        # print("bonded_path_lengths", bonded_path_lengths.shape)
        return score_elec_triu(
            I, atom_type_I, I, atom_type_I, bonded_path_lengths, self.global_params
        )
