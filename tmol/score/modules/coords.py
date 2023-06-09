import torch
import numpy
from functools import singledispatch
from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.stacked_system import StackedSystem
from tmol.score.modules.device import TorchDevice
from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack


@singledispatch
def coords_for(val, score_system: ScoreSystem, *, requires_grad=True) -> torch.Tensor:
    """Shim function to extract forward-pass coords."""
    raise NotImplementedError(f"coords_for: {val}")


@coords_for.register(PackedResidueSystem)
def coords_for_system(
    system: PackedResidueSystem,
    score_system: ScoreSystem,
    *,
    requires_grad: bool = True,
):
    stack_params = StackedSystem.get(score_system)
    device = TorchDevice.get(score_system).device

    assert stack_params.stack_depth == 1
    assert stack_params.system_size == len(system.coords)

    coords = torch.tensor(
        system.coords.reshape(1, len(system.coords), 3),
        dtype=torch.float,
        device=device,
    ).requires_grad_(requires_grad)

    return coords


@coords_for.register(PackedResidueSystemStack)
def coords_for_system_stack(
    stack: PackedResidueSystemStack,
    score_system: ScoreSystem,
    *,
    requires_grad: bool = True,
):
    stack_params = StackedSystem.get(score_system)
    device = TorchDevice.get(score_system).device

    assert stack_params.stack_depth == len(stack.systems)
    assert stack_params.system_size == max(
        int(system.system_size) for system in stack.systems
    )

    coords = torch.full(
        (stack_params.stack_depth, stack_params.system_size, 3),
        numpy.nan,
        dtype=torch.float,
        device=device,
    )

    for i, s in enumerate(stack.systems):
        coords[i, : s.system_size] = torch.tensor(
            s.coords, dtype=torch.float, device=device
        )

    return coords.requires_grad_(requires_grad)
