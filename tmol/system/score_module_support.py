import numpy
import torch
from typing import List

from ..score.modules.bases import ScoreSystem
from ..score.modules.stacked_system import StackedSystem
from ..score.modules.bonded_atom import BondedAtoms
from ..score.modules.device import TorchDevice
from ..score.modules.coords import coords_for

from .packed import PackedResidueSystem, PackedResidueSystemStack


@StackedSystem.build_for.register(PackedResidueSystem)
def stack_for_system(
    system: PackedResidueSystem, score_system: ScoreSystem, **_
) -> StackedSystem:
    return StackedSystem(
        system=score_system, stack_depth=1, system_size=int(system.system_size)
    )


@StackedSystem.build_for.register(PackedResidueSystemStack)
def stack_for_stacked_system(
    stack: PackedResidueSystemStack, score_system: ScoreSystem, **_
) -> StackedSystem:
    return StackedSystem(
        system=score_system,
        stack_depth=len(stack.systems),
        system_size=max(int(system.system_size) for system in stack.systems),
    )


@BondedAtoms.build_for.register(PackedResidueSystem)
def bonded_atoms_for_system(
    system: PackedResidueSystem,
    score_system: ScoreSystem,
    *,
    drop_missing_atoms: bool = False,
    **_,
) -> BondedAtoms:
    bonds = numpy.empty((len(system.bonds), 3), dtype=int)
    bonds[:, 0] = 0
    bonds[:, 1:] = system.bonds

    atom_types = system.atom_metadata["atom_type"].copy()[None, :]
    atom_names = system.atom_metadata["atom_name"].copy()[None, :]
    res_indices = system.atom_metadata["residue_index"].copy()[None, :]
    res_names = system.atom_metadata["residue_name"].copy()[None, :]

    if drop_missing_atoms:
        atom_types[0, numpy.any(numpy.isnan(system.coords), axis=-1)] = None

    return BondedAtoms(
        system=score_system,
        bonds=bonds,
        atom_types=atom_types,
        atom_names=atom_names,
        res_indices=res_indices,
        res_names=res_names,
    )


@BondedAtoms.build_for.register(PackedResidueSystemStack)
def stacked_bonded_atoms_for_system(
    stack: PackedResidueSystemStack,
    system: ScoreSystem,
    *,
    drop_missing_atoms: bool = False,
    **_,
):

    system_size = StackedSystem.get(system).system_size

    bonds_for_systems: List[BondedAtoms] = [
        BondedAtoms.get(
            ScoreSystem._build_with_modules(
                sys, {BondedAtoms}, drop_missing_atoms=drop_missing_atoms
            )
        )
        for sys in stack.systems
    ]

    for i, d in enumerate(bonds_for_systems):
        d.bonds[:, 0] = i
    bonds = numpy.concatenate(tuple(d.bonds for d in bonds_for_systems))

    def expand_atoms(atdat):
        atdat2 = numpy.full((1, system_size), None, dtype=object)
        atdat2[0, : atdat.shape[1]] = atdat
        return atdat2

    def stackem(key):
        return numpy.concatenate(
            [expand_atoms(getattr(d, key)) for d in bonds_for_systems]
        )

    return BondedAtoms(
        system=system,
        bonds=bonds,
        atom_types=stackem("atom_types"),
        atom_names=stackem("atom_names"),
        res_indices=stackem("res_indices"),
        res_names=stackem("res_names"),
    )


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
