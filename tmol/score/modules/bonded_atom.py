import numpy
import torch

import attr
from attrs_strict import type_validator
from functools import singledispatch

from typing import Set, Type

from tmol.score.bonded_atom import IndexedBonds, bonded_path_length_stacked

from tmol.score.modules.bases import ScoreSystem, ScoreModule
from tmol.score.modules.device import TorchDevice
from tmol.score.modules.database import ParamDB
from tmol.score.modules.stacked_system import StackedSystem
from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack


@attr.s(slots=True, auto_attribs=True, kw_only=True, frozen=True)
class BondedAtoms(ScoreModule):
    """Score graph component describing a system's atom types and bonds.

    Attributes:
        atom_types: [layer, atom_index] String atom type descriptors.
            Type descriptions defined in :py:mod:`tmol.database.chemical`.

        atom_names: [layer, atom_index] String residue-specific atom name.

        res_names: [layer, atom_index] String residue name descriptors.

        res_indices: [layer, atom_index] Integer residue index descriptors.

        bonds: [ind, (layer=0, atom_index=1, atom_index=2)] Inter-atomic bond indices.
            Note that bonds are strictly intra-layer, and are defined by a
            single layer index for both atoms of the bond.


        MAX_BONDED_PATH_LENGTH: Maximum relevant inter-atomic path length.
            Limits search depth used in ``bonded_path_length``, all longer
            paths reported as ``inf``.

    """

    @staticmethod
    def depends_on() -> Set[Type[ScoreModule]]:
        return {TorchDevice, ParamDB, StackedSystem}

    @staticmethod
    @singledispatch
    def build_for(val: object, system: ScoreSystem, **_) -> "BondedAtoms":
        """singledispatch hook to resolve system size."""
        raise NotImplementedError(f"BondedAtoms.factory_for: {val}")

    MAX_BONDED_PATH_LENGTH = 6

    # TODO re-enable type validation
    atom_types: numpy.ndarray = attr.ib(
        validator=type_validator()
    )  # NDArray[object][:, :]
    atom_names: numpy.ndarray = attr.ib(
        validator=type_validator()
    )  # NDArray[object][:, :]
    res_names: numpy.ndarray = attr.ib(
        validator=type_validator()
    )  # NDArray[object][:, :]
    res_indices: numpy.ndarray = attr.ib(
        validator=type_validator()
    )  # NDArray[int][:, :]
    bonds: numpy.ndarray = attr.ib(validator=type_validator())  # NDArray[int][:, 3]

    # TODO restore type annotation
    # real_atoms: Tensor[bool][:, :] = attr.ib(init=False)
    real_atoms: torch.Tensor = attr.ib(init=False)

    @real_atoms.default
    def _init_real_atoms(self):
        """Mask of non-null atomic indices in the system."""
        return torch.ByteTensor((self.atom_types != None).astype(numpy.ubyte))

    indexed_bonds: IndexedBonds = attr.ib(init=False)

    @indexed_bonds.default
    def _init_indexed_bonds(self):
        """Sorted, constant time access to bond graph."""
        assert self.bonds.ndim == 2
        assert self.bonds.shape[1] == 3

        ## fd lkball needs this on the device
        ibonds = IndexedBonds.from_bonds(
            IndexedBonds.to_directed(self.bonds),
            minlength=StackedSystem.get(self).system_size,
        ).to(TorchDevice.get(self).device)

        return ibonds

    # TODO restore type annotation
    # bonded_path_length: Tensor[float][:, :, :] = attr.ib(init=False)
    bonded_path_length: torch.Tensor = attr.ib(init=False)

    @bonded_path_length.default
    def _init_bonded_path_length(self):
        """Dense inter-atomic bonded path length distance tables.

        Returns:
            [layer, from_atom, to_atom]
            Per-layer interatomic bonded path length entries.
        """

        return torch.from_numpy(
            bonded_path_length_stacked(
                self.bonds,
                StackedSystem.get(self).stack_depth,
                StackedSystem.get(self).system_size,
                self.MAX_BONDED_PATH_LENGTH,
            )
        ).to(TorchDevice.get(self).device)


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


@BondedAtoms.build_for.register(ScoreSystem)
def _clone_for_score_system(old_system, system, **_) -> BondedAtoms:
    old = BondedAtoms.get(old_system)

    return BondedAtoms(
        system=system,
        atom_types=old.atom_types,
        atom_names=old.atom_names,
        res_names=old.res_names,
        res_indices=old.res_indices,
        bonds=old.bonds,
    )
