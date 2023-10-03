import torch

from ..energy_term import EnergyTerm

from tmol.database import ParameterDatabase
from tmol.score.ref.ref_whole_pose_module import RefWholePoseScoringModule

from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack


class RefEnergyTerm(EnergyTerm):
    device: torch.device  # = attr.ib()

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super(RefEnergyTerm, self).__init__(param_db=param_db, device=device)

        self.ref_weights = param_db.scoring.ref.weights
        self.device = device

    @classmethod
    def score_types(cls):
        import tmol.score.terms.ref_creator

        return tmol.score.terms.ref_creator.RefTermCreator.score_types()

    def n_bodies(self):
        return 1

    def setup_block_type(self, block_type: RefinedResidueType):
        super(RefEnergyTerm, self).setup_block_type(block_type)

        if hasattr(block_type, "ref_weight"):
            return

        ref_weight = 0.0

        if block_type.base_name in self.ref_weights:
            ref_weight = self.ref_weights[block_type.base_name]

        setattr(block_type, "ref_weight", ref_weight)

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(RefEnergyTerm, self).setup_packed_block_types(packed_block_types)

        if hasattr(packed_block_types, "ref_weights"):
            return

        ref_weights = []
        for bt in packed_block_types.active_block_types:
            ref_weights += [bt.ref_weight]

        ref_weights = torch.as_tensor(
            ref_weights, dtype=torch.float32, device=self.device
        )

        setattr(packed_block_types, "ref_weights", ref_weights)

    def setup_poses(self, poses: PoseStack):
        super(RefEnergyTerm, self).setup_poses(poses)

    def render_whole_pose_scoring_module(self, pose_stack: PoseStack):
        pbt = pose_stack.packed_block_types

        return RefWholePoseScoringModule(
            pose_stack_block_coord_offset=pose_stack.block_coord_offset,
            pose_stack_block_types=pose_stack.block_type_ind,
            pose_stack_inter_block_connections=pose_stack.inter_residue_connections,
            bt_atom_downstream_of_conn=pbt.atom_downstream_of_conn,
            ref_weights=pbt.ref_weights,
        )
