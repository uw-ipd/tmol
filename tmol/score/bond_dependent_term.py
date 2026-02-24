import numpy
import torch

from tmol.database import ParameterDatabase
from tmol.chemical.constants import MAX_SIG_BOND_SEPARATION
from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack
from tmol.score.energy_term import EnergyTerm
from tmol.score.bonded_atom import IndexedBonds


# @attr.s(auto_attribs=True)
class BondDependentTerm(EnergyTerm):
    device: torch.device

    def __init__(self, param_db: ParameterDatabase, device: torch.device, **kwargs):
        super(BondDependentTerm, self).__init__(param_db=param_db, device=device)
        self.device = device

    def setup_block_type(self, block_type: RefinedResidueType):
        super(BondDependentTerm, self).setup_block_type(block_type)
        if hasattr(block_type, "intrares_indexed_bonds"):
            return

        bonds = numpy.zeros((block_type.bond_indices.shape[0], 3), dtype=numpy.int32)
        bonds[:, 1:] = block_type.bond_indices.astype(numpy.int32)
        ib = IndexedBonds.from_bonds(bonds, minlength=block_type.n_atoms)
        setattr(block_type, "intrares_indexed_bonds", ib)

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(BondDependentTerm, self).setup_packed_block_types(packed_block_types)

        if hasattr(packed_block_types, "bond_separation"):
            assert hasattr(packed_block_types, "n_all_bonds")
            assert hasattr(packed_block_types, "all_bonds")
            assert hasattr(packed_block_types, "atom_all_bond_ranges")
            return

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
            i_nats = packed_block_types.n_atoms[i]
            bond_separation[i, :i_nats, :i_nats] = rt.path_distance
        bond_separation = torch.tensor(bond_separation, device=self.device)

        n_all_bonds = torch.full(
            (packed_block_types.n_types,),
            -1,
            dtype=torch.int32,
            device=packed_block_types.device,
        )
        max_n_all_bonds = max(
            bt.all_bonds.shape[0] for bt in packed_block_types.active_block_types
        )
        all_bonds = torch.full(
            (packed_block_types.n_types, max_n_all_bonds, 3),
            -1,
            dtype=torch.int32,
            device=packed_block_types.device,
        )
        atom_all_bond_ranges = torch.full(
            (packed_block_types.n_types, packed_block_types.max_n_atoms, 2),
            -1,
            dtype=torch.int32,
            device=packed_block_types.device,
        )

        def _t(arr):
            return torch.tensor(arr, dtype=torch.int32, device=self.device)

        for i, bt in enumerate(packed_block_types.active_block_types):
            i_n_bonds = bt.all_bonds.shape[0]
            n_all_bonds[i] = i_n_bonds
            all_bonds[i, :i_n_bonds, :] = _t(bt.all_bonds)
            atom_all_bond_ranges[i, : bt.n_atoms] = _t(bt.atom_all_bond_ranges)

        setattr(packed_block_types, "bond_separation", bond_separation)
        setattr(packed_block_types, "n_all_bonds", n_all_bonds)
        setattr(packed_block_types, "all_bonds", all_bonds)
        setattr(packed_block_types, "atom_all_bond_ranges", atom_all_bond_ranges)

    def setup_poses(self, pose_stack: PoseStack):
        super(BondDependentTerm, self).setup_poses(pose_stack)

        if hasattr(pose_stack, "min_block_bondsep"):
            return

        min_block_bondsep, _ = torch.min(pose_stack.inter_block_bondsep, dim=4)
        min_block_bondsep, _ = torch.min(min_block_bondsep, dim=3)

        setattr(pose_stack, "min_block_bondsep", min_block_bondsep)
