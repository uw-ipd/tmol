from typing import Optional
from functools import singledispatch

import torch
import math

from tmol.kinematics.metadata import DOFMetadata
from tmol.kinematics.datatypes import KinTree
from tmol.kinematics.script_modules import KinematicModule

from tmol.utility.reactive import reactive_property

from tmol.types.torch import Tensor

from .bonded_atom import BondedAtomScoreGraph
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
class KinematicAtomicCoordinateProvider(
    BondedAtomScoreGraph, StackedSystem, TorchDevice
):
    @staticmethod
    @singledispatch
    def factory_for(
        other, device: torch.device, requires_grad: Optional[bool] = None, **_
    ):
        """`clone`-factory, extract kinop and dofs from other."""

        if requires_grad is None:
            requires_grad = other.dofs.requires_grad

        kintree = other.kintree.to(device)

        if other.dofs.device != device:
            raise ValueError("Unable to change device for kinematic ops.")

        dofs = torch.tensor(other.dofs, device=device).requires_grad_(requires_grad)

        dofmetadata = other.dofmetadata

        return dict(kintree=kintree, dofs=dofs, dofmetadata=dofmetadata)

    # Source dofs
    dofs: Tensor(torch.float)[:, 9]

    # dof info for masking
    dofmetadata: DOFMetadata

    # kinematic tree (= rosetta atomtree)
    kintree: KinTree

    @reactive_property
    def kin_module(kintree: KinTree) -> KinematicModule:
        return KinematicModule(kintree, kintree.id.device)

    @reactive_property
    def coords(
        dofs: Tensor(torch.float)[:, 9],
        kintree: KinTree,
        kin_module: KinematicModule,
        system_size: int,
        stack_depth: int,
    ) -> Tensor(torch.float)[:, :, 3]:
        """System cartesian atomic coordinates."""
        kincoords = kin_module(dofs)

        coords = torch.full(
            (stack_depth, system_size, 3),
            math.nan,
            dtype=dofs.dtype,
            layout=dofs.layout,
            device=dofs.device,
            requires_grad=False,
        )
        coords_flat = coords.reshape((-1, 3))

        idIdx = kintree.id[1:].to(dtype=torch.long)
        coords_flat[idIdx] = kincoords[1:]

        return coords.to(torch.float)

    def reset_coords(self):
        """Reset coordinate state in compute graph, clearing dependent properties."""
        self.dofs = self.dofs
