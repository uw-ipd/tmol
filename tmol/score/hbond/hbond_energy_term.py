import torch

from .hbond_dependent_term import HBondDependentTerm
from .params import CompactedHBondDatabase
from ..atom_type_dependent_term import AtomTypeDependentTerm

from tmol.database import ParameterDatabase

# from tmol.score.hbond.hbond_whole_pose_module import HBondWholePoseScoringModule
from tmol.score.hbond.hbond_scoring_module import HBondWholePoseScoringModule
from tmol.score.hbond.hbond_scoring_module import HBondBlockPairScoringModule
from tmol.score.hbond.hbond_rotamer_scoring_module import HBondRotamerScoringModule

from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack


class HBondEnergyTerm(AtomTypeDependentTerm, HBondDependentTerm):
    tile_size: int = 32
    hb_param_db: CompactedHBondDatabase

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super(HBondEnergyTerm, self).__init__(param_db=param_db, device=device)
        self.tile_size = HBondEnergyTerm.tile_size
        self.hb_param_db = CompactedHBondDatabase.from_database(
            param_db.chemical, param_db.scoring.hbond, device
        )

    @classmethod
    def score_types(cls):
        import tmol.score.terms.hbond_creator

        return tmol.score.terms.hbond_creator.HBondTermCreator.score_types()

    def n_bodies(self):
        return 2

    def setup_block_type(self, block_type: RefinedResidueType):
        super(HBondEnergyTerm, self).setup_block_type(block_type)

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(HBondEnergyTerm, self).setup_packed_block_types(packed_block_types)

    def setup_poses(self, poses: PoseStack):
        super(HBondEnergyTerm, self).setup_poses(poses)

    def render_whole_pose_scoring_module(self, pose_stack: PoseStack):
        return HBondWholePoseScoringModule(
            pose_stack,
            self.hb_param_db,
        )

    def render_block_pair_scoring_module(self, pose_stack: PoseStack):
        return HBondBlockPairScoringModule(
            pose_stack,
            self.hb_param_db,
        )

    def render_rotamer_scoring_module(
        self, pose_stack: PoseStack, rotamer_set, rot_coord_offset
    ):
        pbt = pose_stack.packed_block_types

        return HBondRotamerScoringModule(
            rotamer_set,
        )
