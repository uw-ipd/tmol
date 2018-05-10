import torch
import math

import properties
from tmol.properties.array import VariableT
from tmol.properties.reactive import derived_from
from tmol.kinematics.torch_op import KinematicOp


class RealSpaceScoreGraph(properties.HasProperties):
    coords = VariableT("source atomic coordinates")

    def step(self):
        """Recalculate total_score and gradients wrt/ coords. Does not clear coord grads."""

        self._notify(
            dict(
                name="coords",
                prev=getattr(self, "coords"),
                mode="observe_set"
            )
        )

        self.total_score.backward()
        return self.total_score


class DofSpaceScoreGraph(properties.HasProperties):
    dofs = VariableT("source atomic coordinates")
    kinop = properties.Instance("kinematic op", KinematicOp)

    @derived_from(("dofs", "kinop", "system_size"),
                  VariableT("source atomic coordinates"))
    def coords(self):
        kincoords = self.kinop(self.dofs)

        coords = torch.full(
            (self.system_size, 3),
            math.nan,
            dtype=self.dofs.dtype,
            layout=self.dofs.layout,
            device=self.dofs.device,
            requires_grad=False,
        )

        coords[self.kinop.kintree.id[1:]] = kincoords[1:] # yapf: disable

        return coords.to(torch.float)

    def step(self):
        """Recalculate total_score and gradients wrt/ dofs. Does not clear dof grads."""

        self._notify(
            dict(name="dofs", prev=getattr(self, "dofs"), mode="observe_set")
        )

        self.total_score.backward()
        return self.total_score
