import torch

from .hbond_dependent_term import HBondDependentTerm
from .params import CompactedHBondDatabase
from ..atom_type_dependent_term import AtomTypeDependentTerm

from tmol.database import ParameterDatabase

# from tmol.score.hbond.hbond_whole_pose_module import HBondWholePoseScoringModule
from tmol.score.hbond.potentials.compiled import hbond_pose_scores
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

    def get_score_term_function(self):
        return hbond_pose_scores

    def get_score_term_attributes(self, pose_stack: PoseStack):
        return [
            pose_stack.inter_residue_connections,
            pose_stack.min_block_bondsep,
            pose_stack.inter_block_bondsep,
            pose_stack.packed_block_types.n_atoms,
            pose_stack.packed_block_types.n_conn,
            pose_stack.packed_block_types.conn_atom,
            pose_stack.packed_block_types.n_all_bonds,
            pose_stack.packed_block_types.all_bonds,
            pose_stack.packed_block_types.atom_all_bond_ranges,
            pose_stack.packed_block_types.bond_separation,
            pose_stack.packed_block_types.hbpbt_params.tile_n_donH,
            pose_stack.packed_block_types.hbpbt_params.tile_n_acc,
            pose_stack.packed_block_types.hbpbt_params.tile_donH_inds,
            pose_stack.packed_block_types.hbpbt_params.tile_acc_inds,
            pose_stack.packed_block_types.hbpbt_params.tile_donorH_type,
            pose_stack.packed_block_types.hbpbt_params.tile_acceptor_type,
            pose_stack.packed_block_types.hbpbt_params.tile_acceptor_hybridization,
            pose_stack.packed_block_types.hbpbt_params.is_hydrogen,
            self.hb_param_db.pair_param_table,
            self.hb_param_db.pair_poly_table,
            self.hb_param_db.global_param_table,
        ]
