import torch

from .._atom_type_dependent_term import AtomTypeDependentTerm
from .._bond_dependent_term import BondDependentTerm
from ._params import LJLKTypeParams, LJLKGlobalParams

from tmol.database import ParameterDatabase
from tmol.score.common import tile_subset_indices
from tmol.score.ljlk import LJLKParamResolver

from tmol.chemical import RefinedResidueType
from tmol.pose import (
    PackedBlockTypes,
    PoseStack,
)


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
        self._max_dis = float(param_db.scoring.ljlk.global_parameters.max_dis)
        self.tile_size = LJLKEnergyTerm.tile_size
        self.soft_repulsive = False
        self.rosetta_typed = param_db.scoring.genbonded.rosetta_typed

    @classmethod
    def class_name(cls):
        return "LJLK"

    @classmethod
    def score_types(cls):
        import tmol.score.terms._ljlk_creator

        return tmol.score.terms._ljlk_creator.LJLKTermCreator.score_types()

    def n_bodies(self):
        return 2

    def set_options(self, options: dict):
        if "soft_rep" in options:
            self.soft_repulsive = options["soft_rep"]

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
            assert hasattr(packed_block_types, "ljlk_bond_separation")
            assert hasattr(packed_block_types, "ljlk_all_atoms_ligand_typed")
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

        # A pair of ligand-typed atoms uses CP_CROSSOVER_3FULL: 1-4 pairs get
        # full weight (1.0), encoded by rewriting path_dist 3 and 4 to 5 so
        # connectivity_weight returns 1.0. The convention follows the atoms,
        # not the residue: a noncanonical whose sidechain is ligand-typed is
        # still a polymer. Keep bond_separation unchanged for hbond, which
        # excludes on a binary rule.
        rosetta_typed = self.rosetta_typed
        ljlk_bond_separation = packed_block_types.bond_separation.clone()
        for i, bt in enumerate(packed_block_types.active_block_types):
            n = packed_block_types.n_atoms[i]
            ligand = torch.tensor(
                [a.atom_type not in rosetta_typed for a in bt.atoms],
                dtype=torch.bool,
                device=ljlk_bond_separation.device,
            )
            both = ligand.unsqueeze(1) & ligand.unsqueeze(0)
            slab = ljlk_bond_separation[i, :n, :n]
            slab[both[:n, :n] & ((slab == 3) | (slab == 4))] = 5
        setattr(packed_block_types, "ljlk_bond_separation", ljlk_bond_separation)
        setattr(
            packed_block_types,
            "ljlk_all_atoms_ligand_typed",
            torch.tensor(
                [
                    all(a.atom_type not in self.rosetta_typed for a in bt.atoms)
                    for bt in packed_block_types.active_block_types
                ],
                dtype=torch.int32,
                device=self.device,
            ),
        )

    def setup_poses(self, poses: PoseStack):
        super(LJLKEnergyTerm, self).setup_poses(poses)

    def get_pose_score_term_function(self):
        from tmol.score.ljlk.potentials import ljlk_pose_scores

        return ljlk_pose_scores

    def get_rotamer_score_term_function(self):
        from tmol.score.ljlk.potentials import ljlk_rotamer_scores

        return ljlk_rotamer_scores

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
                    self.type_params.is_carbon_lk,
                ]
            ),
            dim=1,
        )
        global_params = torch.stack(
            _t(
                [
                    self.global_params.max_dis,
                    (
                        self.global_params.lj_dlin_sigma_factor_soft
                        if self.soft_repulsive
                        else self.global_params.lj_dlin_sigma_factor
                    ),
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
            pose_stack.packed_block_types.ljlk_bond_separation,
            pose_stack.packed_block_types.ljlk_all_atoms_ligand_typed,
            type_params,
            global_params,
            # max_dis as host scalar for detect-neighbors call
            self._max_dis,
        ]
