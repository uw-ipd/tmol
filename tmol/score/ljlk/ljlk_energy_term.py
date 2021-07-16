import attr
import torch

from ..atom_type_dependent_term import AtomTypeDependentTerm
from ..bond_dependent_term import BondDependentTerm
from .params import LJLKTypeParams, LJLKGlobalParams
from tmol.score.common.stack_condense import tile_subset_indices

from tmol.system.restypes import RefinedResidueType
from tmol.system.pose import PackedBlockTypes, Poses
from tmol.types.torch import Tensor

from tmol.score.ljlk.potentials.compiled import score_ljlk_inter_system_scores


class LJLKScoreTerm(AtomTypeDependentTerm, BondDependentTerm):
    type_params: LJLKTypeParams
    global_params: LJLKGlobalParams
    tile_size: int = 32

    def n_bodies(self):
        return 2

    def __init__(self, paramdb: tmol.database.ParameterDatabase, device: torch.device):
        ljlk_param_resolver = LJLKParamResolver.from_database(
            default_database.chemical, default_database.scoring.ljlk, device=device
        )
        super(LJLKEnergy, self).__init__(paramdb=paramdb, device=device)
        self.type_params = ljlk_param_resolver.type_params
        self.global_params = ljlk_params.global_params
        self.tile_size = LJLKEnergy.tile_size

    def setup_block_type(self, block_type: RefinedResidueType):
        super(LJLKEnergy, self).setup_block_type(block_type)
        if hasattr(block_type, "ljlk_heavy_atoms_in_tile"):
            assert hasattr(block_type, "ljlk_n_heavy_atoms_in_tile")
            return
        heavy_atoms_in_tile, n_in_tile = tile_subset_indices(
            block_type.heavy_atom_inds, self.tile_size
        )
        setattr(block_type, "ljlk_heavy_atoms_in_tile", heavy_atoms_in_tile)
        setattr(block_type, "ljlk_n_heavy_atoms_in_tile", n_in_tile)

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(LJLKEnergy, self).setup_packed_block_types(packed_block_types)
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

    def setup_poses(self, poses: Poses):
        super(LJLKEnergy, self).setup_poses(poses)
