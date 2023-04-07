import torch
import numpy

from ..energy_term import EnergyTerm

from tmol.database import ParameterDatabase
from tmol.score.cartbonded.params import CartBondedGlobalParams
from tmol.score.cartbonded.cartbonded_whole_pose_module import (
    CartBondedWholePoseScoringModule
)

from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack


class CartBondedEnergyTerm(EnergyTerm):
    device: torch.device  # = attr.ib()

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super(CartBondedEnergyTerm, self).__init__(param_db=param_db, device=device)

        self.global_params = CartBondedGlobalParams.from_database(
            param_db.scoring.cartbonded, device
        )
        self.device = device

    @classmethod
    def score_types(cls):
        import tmol.score.terms.omega_creator

        return tmol.score.terms.omega_creator.CartBondedTermCreator.score_types()

    def n_bodies(self):
        return 1

    def setup_block_type(self, block_type: RefinedResidueType):
        super(CartBondedEnergyTerm, self).setup_block_type(block_type)

        foreign_atoms = ["CA", "CN", "H", "N", "C", "O"]
        cartbonded_foreign_atoms = numpy.full(
            (len(foreign_atoms)), -1, dtype=numpy.int32
        )
        for i, atm in enumerate(foreign_atoms):
            cartbonded_foreign_atoms[i] = block_type.atom_to_idx[atm]

        """if hasattr(block_type, "omega_quad_uaids"):
            return

        setattr(block_type, "omega_quad_uaids", omega_quad_uaids)"""

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(CartBondedEnergyTerm, self).setup_packed_block_types(packed_block_types)

        setup_cartbonded_lengths(packed_block_types)
        # setup_cartbonded_angles(packed_block_types)
        # setup_cartbonded_torsions(packed_block_types)

        setup_cartbonded_foreign_atoms(packed_block_types)

    def setup_cartbonded_foreign_atoms(self, packed_block_types: PackedBlockTypes):
        cartbonded_foreign_atoms = torch.full(())

    def set_cartbonded_lengths(self, packed_block_types: PackedBlockTypes):
        offset = 0

        def _t(arr):
            return torch.tensor(arr, dtype=torch.int32, device=self.device)

        if hasattr(packed_block_types, "cartbonded_length_atoms"):
            return
        num_lengths = numpy.concatenate(
            [
                block_type.cartbonded_lengths
                for block_type in packed_block_types.active_block_types
            ]
        ).size()

        cartbonded_length_atoms = torch.full(
            (num_lengths, 2), 0, dtype=torch.int32, device=self.device
        )
        cartbonded_length_x0 = torch.full(
            (num_lengths), 0, dtype=torch.float32, device=self.device
        )
        cartbonded_length_K = torch.full(
            (num_lengths), 0, dtype=torch.float32, device=self.device
        )
        cartbonded_length_offsets = torch.full(
            (len(packed_block_types.active_block_types)),
            -1,
            dtype=torch.int32,
            device=self.device,
        )

        for i, bt in enumerate(packed_block_types.active_block_types):
            cartbonded_length_offsets[i] = offset
            for j, cbl in enumerate(bt.cartbonded_lengths):
                cartbonded_length_atoms[offset + j] = cbl.atoms
                cartbonded_length_x0[offset + j] = cbl.x0
                cartbonded_length_K[offset + j] = cbl.K
            offset += len(bt.cartbonded_lengths)

        setattr(packed_block_types, "cartbonded_length_atoms", cartbonded_length_atoms)
        setattr(packed_block_types, "cartbonded_length_x0", cartbonded_length_x0)
        setattr(packed_block_types, "cartbonded_length_K", cartbonded_length_K)

    def setup_poses(self, poses: PoseStack):
        super(CartBondedEnergyTerm, self).setup_poses(poses)

    def render_whole_pose_scoring_module(self, pose_stack: PoseStack):
        pbt = pose_stack.packed_block_types

        return CartBondedWholePoseScoringModule(
            pose_stack_block_coord_offset=pose_stack.block_coord_offset,
            pose_stack_block_types=pose_stack.block_type_ind,
            pose_stack_inter_block_connections=pose_stack.inter_residue_connections,
            bt_cartbonded_length_atoms=pbt.cartbonded_length_atoms,
            bt_cartbonded_length_x0=pbt.cartbonded_length_x0,
            bt_cartbonded_length_K=pbt.cartbonded_length_K,
            global_params=self.global_params,
        )
