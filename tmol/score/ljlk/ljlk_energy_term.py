import torch

from ..atom_type_dependent_term import AtomTypeDependentTerm
from ..bond_dependent_term import BondDependentTerm
from .params import LJLKTypeParams, LJLKGlobalParams

from tmol.database import ParameterDatabase
from tmol.score.common.stack_condense import tile_subset_indices
from tmol.score.ljlk.params import LJLKParamResolver

from tmol.score.ljlk.potentials.compiled import ljlk_pose_scores, ljlk_rotamer_scores

from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack


class LJLKEnergyTerm(AtomTypeDependentTerm, BondDependentTerm):
    type_params: LJLKTypeParams
    global_params: LJLKGlobalParams
    tile_size: int = 32

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        ljlk_param_resolver = LJLKParamResolver.from_database(
            param_db.chemical, param_db.scoring.ljlk, device=device
        )
        super(LJLKEnergyTerm, self).__init__(param_db=param_db, device=device)
        self.type_params = ljlk_param_resolver.type_params
        self.global_params = ljlk_param_resolver.global_params
        self.tile_size = LJLKEnergyTerm.tile_size

    @classmethod
    def class_name(cls):
        return "LJLK"

    @classmethod
    def score_types(cls):
        import tmol.score.terms.ljlk_creator

        return tmol.score.terms.ljlk_creator.LJLKTermCreator.score_types()

    def n_bodies(self):
        return 2

    def setup_block_type(self, block_type: RefinedResidueType):
        super(LJLKEnergyTerm, self).setup_block_type(block_type)
        if hasattr(block_type, "ljlk_heavy_atoms_in_tile"):
            assert hasattr(block_type, "ljlk_n_heavy_atoms_in_tile")
            return
        heavy_atoms_in_tile, n_in_tile = tile_subset_indices(
            block_type.heavy_atom_inds, self.tile_size
        )
        setattr(block_type, "ljlk_heavy_atoms_in_tile", heavy_atoms_in_tile)
        setattr(block_type, "ljlk_n_heavy_atoms_in_tile", n_in_tile)

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(LJLKEnergyTerm, self).setup_packed_block_types(packed_block_types)
        if hasattr(packed_block_types, "ljlk_heavy_atoms_in_tile"):
            assert hasattr(packed_block_types, "ljlk_n_heavy_atoms_in_tile")
            return
        max_n_tiles = (packed_block_types.max_n_atoms - 1) // self.tile_size + 1
        heavy_atoms_in_tile = torch.full(
            (packed_block_types.n_types, max_n_tiles * self.tile_size),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        n_heavy_ats_in_tile = torch.full(
            (packed_block_types.n_types, max_n_tiles),
            0,
            dtype=torch.int32,
            device=self.device,
        )

        def _t(arr):
            return torch.tensor(arr, dtype=torch.int32, device=self.device)

        for i, rt in enumerate(packed_block_types.active_block_types):
            i_n_tiles = rt.ljlk_n_heavy_atoms_in_tile.shape[0]
            i_n_tile_ats = i_n_tiles * self.tile_size
            heavy_atoms_in_tile[i, :i_n_tile_ats] = _t(rt.ljlk_heavy_atoms_in_tile)
            n_heavy_ats_in_tile[i, :i_n_tiles] = _t(rt.ljlk_n_heavy_atoms_in_tile)

        setattr(packed_block_types, "ljlk_heavy_atoms_in_tile", heavy_atoms_in_tile)
        setattr(packed_block_types, "ljlk_n_heavy_atoms_in_tile", n_heavy_ats_in_tile)

    def setup_poses(self, poses: PoseStack):
        super(LJLKEnergyTerm, self).setup_poses(poses)

    def get_pose_score_term_function(self):
        return ljlk_pose_scores

    def get_rotamer_score_term_function(self):
        return ljlk_rotamer_scores

    # def get_score_term_function(self):
    #     return ljlk_pose_scores

    def get_score_term_attributes(self, pose_stack):
        def _t(ts):
            return tuple(map(lambda t: t.to(torch.float), ts))

        type_params = torch.stack(
            _t(
                [
                    self.type_params.lj_radius,
                    self.type_params.lj_wdepth,
                    self.type_params.lk_dgfree,
                    self.type_params.lk_lambda,
                    self.type_params.lk_volume,
                    self.type_params.is_donor,
                    self.type_params.is_hydroxyl,
                    self.type_params.is_polarh,
                    self.type_params.is_acceptor,
                ]
            ),
            dim=1,
        )
        global_params = torch.stack(
            _t(
                [
                    self.global_params.lj_hbond_dis,
                    self.global_params.lj_hbond_OH_donor_dis,
                    self.global_params.lj_hbond_hdis,
                ]
            ),
            dim=1,
        )
        return [
            pose_stack.min_block_bondsep,
            pose_stack.inter_block_bondsep,
            pose_stack.packed_block_types.n_atoms,
            pose_stack.packed_block_types.ljlk_n_heavy_atoms_in_tile,
            pose_stack.packed_block_types.ljlk_heavy_atoms_in_tile,
            pose_stack.packed_block_types.atom_types,
            pose_stack.packed_block_types.n_conn,
            pose_stack.packed_block_types.conn_atom,
            pose_stack.packed_block_types.bond_separation,
            type_params,
            global_params,
        ]
