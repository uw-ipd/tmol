from tmol.types.torch import Tensor

from tmol.score.omega.potentials.compiled import score_omega

import torch


class OmegaScoreModule(torch.jit.ScriptModule):
    def __init__(self, indices: Tensor, K: Tensor):
        super().__init__()

        def _tfloat(ts):
            return tuple(map(lambda t: t.to(torch.float), ts))

        self.params = torch.nn.Parameter(
            torch.cat(
                _tfloat([indices, K.expand((indices.shape[0], indices.shape[1], 1))]),
                dim=2,
            ),
            requires_grad=False,
        )

    @torch.jit.script_method
    def forward(self, coords):
        return score_omega(coords, self.params)
