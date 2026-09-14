import numpy
import torch

from ._annotation_cache import AnnotationKey, cached_annotation, store_annotation

from tmol.database import ParameterDatabase
from tmol.chemical import (
    MAX_SIG_BOND_SEPARATION,
    RefinedResidueType,
)
from tmol.pose import (
    PackedBlockTypes,
    PoseStack,
)
from tmol.score import (
    EnergyTerm,
    IndexedBonds,
)


# @attr.s(auto_attribs=True)
class BondDependentTerm(EnergyTerm):
    device: torch.device

    def __init__(self, param_db: ParameterDatabase, device: torch.device, **kwargs):
        super(BondDependentTerm, self).__init__(param_db=param_db, device=device)
        self.device = device
        self._bond_block_key = AnnotationKey.from_sources()
        self._bond_packed_key = AnnotationKey.from_sources(settings=(self.device,))
        self._bond_pose_key = AnnotationKey.from_sources(settings=(self.device,))

    def setup_block_type(self, block_type: RefinedResidueType):
        super(BondDependentTerm, self).setup_block_type(block_type)
        cached = cached_annotation(
            block_type, "_bond_dependent_annotation", self._bond_block_key
        )
        if cached is not None:
            return cached

        bonds = numpy.zeros((block_type.bond_indices.shape[0], 3), dtype=numpy.int32)
        bonds[:, 1:] = block_type.bond_indices.astype(numpy.int32)
        ib = IndexedBonds.from_bonds(bonds, minlength=block_type.n_atoms)
        setattr(block_type, "intrares_indexed_bonds", ib)
        return store_annotation(
            block_type,
            "_bond_dependent_annotation",
            self._bond_block_key,
            ib,
            fields=("intrares_indexed_bonds",),
        )

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(BondDependentTerm, self).setup_packed_block_types(packed_block_types)

        cached = cached_annotation(
            packed_block_types,
            "_bond_dependent_annotation",
            self._bond_packed_key,
        )
        if cached is not None:
            return cached

        # Concatenate the block-type path-distances arrays into a single array
        bond_separation = numpy.full(
            (
                packed_block_types.n_types,
                packed_block_types.max_n_atoms,
                packed_block_types.max_n_atoms,
            ),
            MAX_SIG_BOND_SEPARATION,
            dtype=numpy.int32,
        )
        for i, rt in enumerate(packed_block_types.active_block_types):
            i_nats = rt.n_atoms
            bond_separation[i, :i_nats, :i_nats] = rt.path_distance

        max_n_all_bonds = max(
            bt.all_bonds.shape[0] for bt in packed_block_types.active_block_types
        )
        n_all_bonds = numpy.full((packed_block_types.n_types,), -1, dtype=numpy.int32)
        all_bonds = numpy.full(
            (packed_block_types.n_types, max_n_all_bonds, 3),
            -1,
            dtype=numpy.int32,
        )
        atom_all_bond_ranges = numpy.full(
            (packed_block_types.n_types, packed_block_types.max_n_atoms, 2),
            -1,
            dtype=numpy.int32,
        )

        for i, bt in enumerate(packed_block_types.active_block_types):
            i_n_bonds = bt.all_bonds.shape[0]
            n_all_bonds[i] = i_n_bonds
            all_bonds[i, :i_n_bonds, :] = bt.all_bonds
            atom_all_bond_ranges[i, : bt.n_atoms] = bt.atom_all_bond_ranges

        setattr(
            packed_block_types,
            "bond_separation",
            torch.as_tensor(bond_separation, device=self.device),
        )
        setattr(
            packed_block_types,
            "n_all_bonds",
            torch.as_tensor(n_all_bonds, device=self.device),
        )
        setattr(
            packed_block_types,
            "all_bonds",
            torch.as_tensor(all_bonds, device=self.device),
        )
        setattr(
            packed_block_types,
            "atom_all_bond_ranges",
            torch.as_tensor(atom_all_bond_ranges, device=self.device),
        )
        return store_annotation(
            packed_block_types,
            "_bond_dependent_annotation",
            self._bond_packed_key,
            (
                packed_block_types.bond_separation,
                packed_block_types.n_all_bonds,
                packed_block_types.all_bonds,
                packed_block_types.atom_all_bond_ranges,
            ),
            fields=(
                "bond_separation",
                "n_all_bonds",
                "all_bonds",
                "atom_all_bond_ranges",
            ),
        )

    def setup_poses(self, pose_stack: PoseStack):
        super(BondDependentTerm, self).setup_poses(pose_stack)

        cached = cached_annotation(
            pose_stack, "_bond_dependent_annotation", self._bond_pose_key
        )
        if cached is not None:
            return cached

        min_block_bondsep = torch.amin(pose_stack.inter_block_bondsep, dim=(3, 4))

        setattr(pose_stack, "min_block_bondsep", min_block_bondsep)
        return store_annotation(
            pose_stack,
            "_bond_dependent_annotation",
            self._bond_pose_key,
            min_block_bondsep,
            fields=("min_block_bondsep",),
        )
