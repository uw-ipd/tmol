from typing import Optional
from functools import singledispatch

import torch
import math

from tmol.kinematics.torch_op import KinematicOp

from tmol.utility.reactive import reactive_property

from tmol.types.torch import Tensor

from .device import TorchDevice
from .score_graph import score_graph
from .stacked_system import StackedSystem


@score_graph
class CartesianAtomicCoordinateProvider(StackedSystem, TorchDevice):
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
    coords: Tensor(torch.float)[:, :, 3]

    def reset_coords(self):
        """Reset coordinate state in compute graph, clearing dependent properties."""
        self.coords = self.coords


@score_graph
class KinematicAtomicCoordinateProvider(StackedSystem, TorchDevice):
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
    def coords(
        dofs: Tensor("f4")[:], kinop: KinematicOp, system_size: int
    ) -> Tensor("f4")[:, :, 3]:
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

        coords[kinop.kintree.id[1:]] = kincoords[1:]

        return coords.to(torch.float)[None, ...]

    def reset_coords(self):
        """Reset coordinate state in compute graph, clearing dependent properties."""
        self.dofs = self.dofs
