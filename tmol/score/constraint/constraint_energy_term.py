import torch
import numpy

from ..energy_term import EnergyTerm

from tmol.database import ParameterDatabase
from tmol.score.constraint.constraint_whole_pose_module import (
    ConstraintWholePoseScoringModule,
)

from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack


class ConstraintEnergyTerm(EnergyTerm):
    device: torch.device  # = attr.ib()

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super(ConstraintEnergyTerm, self).__init__(param_db=param_db, device=device)

        self.device = device

    @classmethod
    def score_types(cls):
        import tmol.score.terms.constraint_creator

        return tmol.score.terms.constraint_creator.ConstraintTermCreator.score_types()

    def n_bodies(self):
        return 1

    def setup_block_type(self, block_type: RefinedResidueType):
        super(ConstraintEnergyTerm, self).setup_block_type(block_type)

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(ConstraintEnergyTerm, self).setup_packed_block_types(packed_block_types)

    def setup_poses(self, poses: PoseStack):
        super(ConstraintEnergyTerm, self).setup_poses(poses)

    def render_whole_pose_scoring_module(self, pose_stack: PoseStack):
        pbt = pose_stack.packed_block_types

        return ConstraintWholePoseScoringModule(
            pose_stack_block_coord_offset=pose_stack.block_coord_offset,
            pose_stack_block_types=pose_stack.block_type_ind,
            pose_stack_inter_block_connections=pose_stack.inter_residue_connections,
            bt_atom_downstream_of_conn=pbt.atom_downstream_of_conn,
        )
