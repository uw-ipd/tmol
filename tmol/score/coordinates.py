from typing import Optional
from functools import singledispatch

import torch
import math

from tmol.kinematics.torch_op import KinematicOp

from tmol.utility.reactive import reactive_attrs, reactive_property

from tmol.types.torch import Tensor

from .factory import Factory
from .device import TorchDevice


@reactive_attrs(auto_attribs=True)
class CartesianAtomicCoordinateProvider(TorchDevice, Factory):
    @staticmethod
    @singledispatch
    def factory_for(
        other, device: torch.device, requires_grad: Optional[bool] = None, **_
    ):
        """`clone`-factory, extract coords from other."""
        if requires_grad is None:
            requires_grad = other.coords.requires_grad

        coords = torch.tensor(
            other.coords, dtype=torch.float, device=device
        ).requires_grad_(requires_grad)

        return dict(coords=coords)

    # Source atomic coordinates
    coords: Tensor(torch.float)[:, 3]

    @reactive_property
    def coords64(coords):
        return coords.to(torch.double)

    def reset_total_score(self):
        self.coords = self.coords


@reactive_attrs(auto_attribs=True)
class KinematicAtomicCoordinateProvider(TorchDevice, Factory):
    @staticmethod
    @singledispatch
    def factory_for(
        other, device: torch.device, requires_grad: Optional[bool] = None, **_
    ):
        """`clone`-factory, extract kinop and dofs from other."""

        if requires_grad is None:
            requires_grad = other.dofs.requires_grad

        kinop = other.kinop

        if other.dofs.device != device:
            raise ValueError("Unable to change device for kinematic ops.")

        dofs = torch.tensor(other.dofs, device=device).requires_grad_(requires_grad)

        return dict(kinop=kinop, dofs=dofs)

    # Source mobile dofs
    dofs: Tensor("f4")[:]

    # Kinematic operation of the mobile dofs
    kinop: KinematicOp

    @reactive_property
    def coords64(
        dofs: Tensor("f4")[:], kinop: KinematicOp, system_size: int
    ) -> Tensor("f8")[:, 3]:
        """System cartesian atomic coordinates at double precision."""
        kincoords = kinop(dofs)

        coords64 = torch.full(
            (system_size, 3),
            math.nan,
            dtype=dofs.dtype,
            layout=dofs.layout,
            device=dofs.device,
            requires_grad=False,
        )

        coords64[kinop.kintree.id[1:]] = kincoords[1:]
        return coords64

    @reactive_property
    def coords(coords64: Tensor("f8")[:, 3]) -> Tensor("f4")[:, 3]:
        """System cartesian atomic coordinates at single precision."""
        return coords64.to(torch.float)

    def reset_total_score(self):
        self.dofs = self.dofs
