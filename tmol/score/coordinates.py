import torch
import math

from tmol.kinematics.torch_op import KinematicOp

from tmol.utility.reactive import reactive_attrs, reactive_property

from tmol.types.torch import Tensor


@reactive_attrs(auto_attribs=True)
class CartesianAtomicCoordinateProvider:

    # Source atomic coordinates
    coords: Tensor("f4")[:, 3]

    def step(self):
        """Recalculate total_score and gradients wrt/ coords. Does not clear coord grads."""

        self.coords = self.coords

        # TODO asford broken by total score issue?
        self.total_score.backward()
        return self.total_score


@reactive_attrs(auto_attribs=True)
class KinematicAtomicCoordinateProvider:
    # Source mobile dofs
    dofs: Tensor("f4")[:]

    # Kinematic operation of the mobile dofs
    kinop: KinematicOp

    @reactive_property
    def coords(
            dofs: Tensor("f4")[:],
            kinop: KinematicOp,
            system_size: int,
    ) -> Tensor("f4")[:, 3]:
        """System cartesian atomic coordinates."""
        kincoords = kinop(dofs)

        coords = torch.full(
            (system_size, 3),
            math.nan,
            dtype=dofs.dtype,
            layout=dofs.layout,
            device=dofs.device,
            requires_grad=False,
        )

        coords[kinop.kintree.id[1:]] = kincoords[1:] # yapf: disable

        return coords.to(torch.float)

    def step(self):
        """Recalculate total_score and gradients wrt/ dofs. Does not clear dof grads."""

        self.coords = self.coords

        # TODO asford broken by total score issue?
        self.total_score.backward()
        return self.total_score
