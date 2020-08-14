import attr

from functools import singledispatch

import torch

from ..device import TorchDevice
from ..bonded_atom import BondedAtomScoreGraph
from ..score_components import ScoreComponentClasses, IntraScore
from ..score_graph import score_graph

from .script_modules import OmegaScoreModule

from tmol.utility.reactive import reactive_attrs, reactive_property

from tmol.types.functional import validate_args
from tmol.types.array import NDArray

from tmol.types.torch import Tensor
from tmol.types.tensor import TensorGroup

from tmol.score.common.stack_condense import condense_subset


@attr.s(auto_attribs=True)
class OmegaParams(TensorGroup):
    omega_indices: Tensor[torch.int32][:, :, 4]


@reactive_attrs
class OmegaIntraScore(IntraScore):
    @reactive_property
    def total_omega(target):
        return target.omega_module(target.coords)


@score_graph
class OmegaScoreGraph(BondedAtomScoreGraph, TorchDevice):
    total_score_components = [
        ScoreComponentClasses(
            "omega", intra_container=OmegaIntraScore, inter_container=None
        )
    ]

    @staticmethod
    @singledispatch
    def factory_for(val, device: torch.device, **_):
        return dict(allomegas=val.allomegas, device=device)

    allomegas: NDArray[int][:, :, 4]
    device: torch.device

    @reactive_property
    @validate_args
    def omega_module(
        omega_resolve_indices: OmegaParams, spring_constant: Tensor[torch.float]
    ) -> OmegaScoreModule:
        return OmegaScoreModule(omega_resolve_indices.omega_indices, spring_constant)

    @reactive_property
    @validate_args
    def spring_constant(device: torch.device) -> Tensor[torch.float]:
        """ The spring constant for omega (in radians)"""
        return torch.tensor(32.8, device=device, dtype=torch.float)

    @reactive_property
    @validate_args
    def omega_resolve_indices(
        device: torch.device, allomegas: NDArray[int][:, :, 4]
    ) -> OmegaParams:
        # remove undefined indices and send to device
        allomegas = torch.tensor(allomegas, device=device)
        omega_defined = torch.all(allomegas != -1, dim=2)
        omega_indices = condense_subset(allomegas, omega_defined).to(torch.int32)

        return OmegaParams(omega_indices=omega_indices)
