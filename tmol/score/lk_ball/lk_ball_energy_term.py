import torch

from .lk_ball_whole_pose_module import LKBallWholePoseScoringModule
from ..atom_type_dependent_term import AtomTypeDependentTerm
from ..bond_dependent_term import BondDependentTerm
from ..hbond.hbond_dependent_term import HBondDependentTerm
from ..ljlk.params import LJLKGlobalParams, LJLKParamResolver
from tmol.database import ParameterDatabase

from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack

# from tmol.types.torch import Tensor


class LKBallEnergyTerm(AtomTypeDependentTerm, HBondDependentTerm):
    tile_size: int = 32
    ljlk_global_params: LJLKGlobalParams
    ljlk_param_resolver: LJLKParamResolver

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super(LKBallEnergyTerm, self).__init__(param_db=param_db, device=device)

        self.ljlk_param_resolver = LJLKParamResolver.from_database(
            param_db.chemical, param_db.scoring.ljlk, device=device
        )
        # self.type_params = ljlk_param_resolver.type_params
        # self.global_params = ljlk_param_resolver.global_params
        self.tile_size = LKBallEnergyTerm.tile_size

    @classmethod
    def score_types(cls):
        import tmol.score.terms.lk_ball_creator

        return tmol.score.terms.lk_ball_creator.LKBallTermCreator.score_types()

    def n_bodies(self):
        return 2

    def setup_block_type(self, block_type: RefinedResidueType):
        super(LKBallEnergyTerm, self).setup_block_type(block_type)
        if hasattr(block_type, "lk_ball_params"):
            return
        # TO DO!

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(LKBallEnergyTerm, self).setup_packed_block_types(packed_block_types)

    def setup_poses(self, pose_stack: PoseStack):
        super(LKBallEnergyTerm, self).setup_poses(pose_stack)

    def render_whole_pose_scoring_module(self, pose_stack: PoseStack):
        pbt = pose_stack.packed_block_types
        ljlk_global_params = self.ljlk_param_resolver.global_params

        return LKBallWholePoseScoringModule(
            pose_stack_block_coord_offset=pose_stack.block_coord_offset,
            pose_stack_block_type=pose_stack.block_type_ind,
            pose_stack_inter_residue_connections=pose_stack.inter_residue_connections,
            pose_stack_min_bond_separation=pose_stack.min_block_bondsep,
            pose_stack_inter_block_bondsep=pose_stack.inter_block_bondsep,
            bt_n_atoms=pbt.n_atoms,
            bt_n_interblock_bonds=pbt.n_conn,
            bt_atoms_forming_chemical_bonds=pbt.conn_atom,
            bt_n_all_bonds=pbt.n_all_bonds,
            bt_all_bonds=pbt.all_bonds,
            bt_atom_all_bond_ranges=pbt.atom_all_bond_ranges,
            bt_tile_n_donH=pbt.hbpbt_params.tile_n_donH,
            bt_tile_n_acc=pbt.hbpbt_params.tile_n_acc,
            bt_tile_donH_inds=pbt.hbpbt_params.tile_donH_inds,
            bt_tile_don_hvy_inds=None,  # TO DO!!!
            bt_tile_which_donH_for_hvy=None,  # TO DO!!
            bt_tile_acc_inds=pbt.hbpbt_params.tile_acc_inds,
            bt_tile_acceptor_hybridization=pbt.hbpbt_params.tile_acceptor_hybridization,
            bt_tile_acc_n_attached_H=None,  # TO DO!!
            bt_atom_is_hydrogen=pbt.hbpbt_params.is_hydrogen,
            bt_tile_n_polar_atoms=None,  # TO DO!!
            bt_tile_n_occluder_atoms=None,  # TO DO!!
            bt_tile_pol_occ_inds=None,  # TO DO!!
            bt_tile_lk_ball_params=None,  # TO DO!!
            bt_path_distance=pbt.bond_separation,
            lk_ball_global_params=self.stack_lk_ball_global_params(),
            water_gen_global_params=self.stack_water_gen_global_params(),
            sp2_water_tors=ljlk_global_params.lkb_water_tors_sp2,
            sp3_water_tors=ljlk_global_params.lkb_water_tors_sp3,
            ring_water_tors=ljlk_global_params.lkb_water_tors_ring,
        )

    def _tfloat(self, ts):
        return tuple(map(lambda t: t.to(torch.float), ts))

    def stack_lk_ball_global_params(self):
        return torch.stack(
            self._tfloat(
                [
                    self.ljlk_param_resolver.global_params.lj_hbond_dis,
                    self.ljlk_param_resolver.global_params.lj_hbond_OH_donor_dis,
                    self.ljlk_param_resolver.global_params.lj_hbond_hdis,
                    self.ljlk_param_resolver.global_params.lkb_water_dist,
                    self.ljlk_param_resolver.global_params.max_dis,
                ]
            ),
            dim=1,
        )

    def stack_lkball_water_gen_global_params(self):
        return torch.stack(
            self._tfloat(
                [
                    self.ljlk_param_resolver.global_params.lkb_water_dist,
                    self.ljlk_param_resolver.global_params.lkb_water_angle_sp2,
                    self.ljlk_param_resolver.global_params.lkb_water_angle_sp3,
                    self.ljlk_param_resolver.global_params.lkb_water_angle_ring,
                ]
            ),
            dim=1,
        )
