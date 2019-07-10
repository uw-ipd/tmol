from tmol.types.torch import Tensor

import tmol.score.omega.potentials.compiled  # noqa
from tmol.utility.cuda.synchronize import synchronize_if_cuda_available

import torch


class OmegaScoreModule(torch.jit.ScriptModule):
    def __init__(self, indices: Tensor, K: Tensor):
        super().__init__()

        def _tfloat(ts):
            return tuple(map(lambda t: t.to(torch.float), ts))

        self.params = torch.nn.Parameter(
            torch.cat(_tfloat([indices, K.expand((indices.shape[0], 1))]), dim=1),
            requires_grad=False,
        )

    @torch.jit.script_method
    def forward(self, coords):
        return torch.ops.tmol.score_omega(coords, self.params)

    def final(self, coords):
        res = self(coords)
        synchronize_if_cuda_available()
        return res
