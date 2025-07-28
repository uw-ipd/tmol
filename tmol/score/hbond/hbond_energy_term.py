import torch

from .hbond_dependent_term import HBondDependentTerm
from .params import CompactedHBondDatabase
from ..atom_type_dependent_term import AtomTypeDependentTerm

from tmol.database import ParameterDatabase

from tmol.score.hbond.hbond_whole_pose_module import HBondWholePoseScoringModule
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
        pbt = pose_stack.packed_block_types

        identity_map = pose_stack.block_identity_map()

        return HBondWholePoseScoringModule(
            identity_map=identity_map,
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
            bt_tile_acc_inds=pbt.hbpbt_params.tile_acc_inds,
            bt_tile_donor_type=pbt.hbpbt_params.tile_donorH_type,
            bt_tile_acceptor_type=pbt.hbpbt_params.tile_acceptor_type,
            bt_tile_acceptor_hybridization=pbt.hbpbt_params.tile_acceptor_hybridization,
            bt_atom_is_hydrogen=pbt.hbpbt_params.is_hydrogen,
            bt_path_distance=pbt.bond_separation,
            pair_params=self.hb_param_db.pair_param_table,
            pair_polynomials=self.hb_param_db.pair_poly_table,
            global_params=self.hb_param_db.global_param_table,
        )

    def render_rotamer_scoring_module(self, pose_stack: PoseStack, rotamer_set):
        pbt = pose_stack.packed_block_types

        return HBondRotamerScoringModule(
            rotamer_set,
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
            bt_tile_acc_inds=pbt.hbpbt_params.tile_acc_inds,
            bt_tile_donor_type=pbt.hbpbt_params.tile_donorH_type,
            bt_tile_acceptor_type=pbt.hbpbt_params.tile_acceptor_type,
            bt_tile_acceptor_hybridization=pbt.hbpbt_params.tile_acceptor_hybridization,
            bt_atom_is_hydrogen=pbt.hbpbt_params.is_hydrogen,
            bt_path_distance=pbt.bond_separation,
            pair_params=self.hb_param_db.pair_param_table,
            pair_polynomials=self.hb_param_db.pair_poly_table,
            global_params=self.hb_param_db.global_param_table,
        )
