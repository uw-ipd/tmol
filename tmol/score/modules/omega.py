import attr
from attrs_strict import type_validator
from typing import Set, Type
import torch
from functools import singledispatch

from tmol.score.omega.script_modules import OmegaScoreModule

from tmol.score.common.stack_condense import condense_subset
from tmol.score.modules.bases import ScoreSystem, ScoreModule, ScoreMethod
from tmol.score.modules.device import TorchDevice
from tmol.score.modules.stacked_system import StackedSystem

from tmol.system.score_support import (
    allomegas_from_packed_residue_system,
    allomegas_from_packed_residue_system_stack,
)
from tmol.system.packed import PackedResidueSystemStack

from tmol.types.array import NDArray
from tmol.types.torch import Tensor


@attr.s(slots=True, auto_attribs=True, kw_only=True, frozen=True)
class OmegaParameters(ScoreModule):
    @staticmethod
    def depends_on() -> Set[Type[ScoreModule]]:
        return {StackedSystem, TorchDevice}

    @staticmethod
    @singledispatch
    def build_for(val, system: ScoreSystem, **_):
        """Override constructor.
        """

        allomegas = allomegas_from_packed_residue_system(val)

        return OmegaParameters(system=system, allomegas=allomegas)

    allomegas: NDArray[int][:, :, 4] = attr.ib(validator=type_validator())
    omega_indices: Tensor[torch.int32][:, :, 4] = attr.ib(init=False)
    spring_constant: Tensor[torch.float] = attr.ib(init=False)

    @omega_indices.default
    def _init_omega_indices(self) -> Tensor[torch.int32][:, :, 4]:
        # remove undefined indices and send to device
        allomegas_torch = torch.tensor(
            self.allomegas, device=TorchDevice.get(self.system).device
        )
        omega_defined = torch.all(allomegas_torch != -1, dim=2)
        omega_indices = condense_subset(allomegas_torch, omega_defined).to(torch.int32)

        return omega_indices

    @spring_constant.default
    def _init_spring_constant(self) -> Tensor[torch.float]:
        """ The spring constant for omega (in radians)"""
        return torch.tensor(
            32.8, device=TorchDevice.get(self.system).device, dtype=torch.float
        )


@OmegaParameters.build_for.register(ScoreSystem)
def _clone_for_score_system(old, system: ScoreSystem, **_):
    """Override constructor.
        """

    return OmegaParameters(system=system, allomegas=OmegaParameters.get(old).allomegas)


@OmegaParameters.build_for.register(PackedResidueSystemStack)
def _build_for_stack(stack, system: ScoreSystem, **_):
    """Override constructor.
    """

    allomegas = allomegas_from_packed_residue_system_stack(stack)

    return OmegaParameters(system=system, allomegas=allomegas)


@attr.s(slots=True, auto_attribs=True, kw_only=True)
class OmegaScore(ScoreMethod):
    @staticmethod
    def depends_on() -> Set[Type[ScoreModule]]:
        return {OmegaParameters}

    @staticmethod
    def build_for(val, system: ScoreSystem, **_) -> "OmegaScore":
        return OmegaScore(system=system)

    omega_module: OmegaScoreModule = attr.ib(init=False)

    @omega_module.default
    def _init_omega_module(self) -> OmegaScoreModule:
        return OmegaScoreModule(
            OmegaParameters.get(self).omega_indices,
            OmegaParameters.get(self).spring_constant,
        )

    def intra_forward(self, coords: torch.Tensor):
        return {"omega": self.omega_module(coords)}
