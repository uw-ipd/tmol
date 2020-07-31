import attr
from attrs_strict import type_validator
import torch
from typing import Optional
from functools import singledispatch

from tmol.score.modules.bases import ScoreModule, ScoreSystem


@attr.s(slots=True, auto_attribs=True, kw_only=True, frozen=True)
class TorchDevice(ScoreModule):
    """Score module specifying the target compute device.

    Attributes:
        device: The common torch compute device used for all operations.
    """

    @classmethod
    def depends_on(cls):
        return set()

    @staticmethod
    @singledispatch
    def build_for(
        val, system: ScoreSystem, device: Optional[torch.device] = None, **_
    ) -> "TorchDevice":
        """Overridable clone-constructor."""
        if device is None:
            device = torch.device("cpu")

        return TorchDevice(system=system, device=device)

    device: torch.device = attr.ib(validator=type_validator())


@TorchDevice.build_for.register(ScoreSystem)
def _clone_for_score_system(
    old, system, *, device: Optional[torch.device] = None, **_
) -> "TorchDevice":
    """Clone-constructor for score system, default to source device."""

    if device is None:
        device = TorchDevice.get(old).device

    return TorchDevice(system=system, device=device)
