import torch

from ..atom_type_dependent_term import AtomTypeDependentTerm
from ..bond_dependent_term import BondDependentTerm
from .params import LJLKTypeParams, LJLKGlobalParams
from tmol.score.common.stack_condense import tile_subset_indices

from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack
from tmol.types.torch import Tensor

from tmol.score.ljlk.potentials.compiled import (
    score_ljlk_inter_system_scores,
    register_lj_lk_rotamer_pair_energy_eval,
)


class LJLKEnergy(AtomTypeDependentTerm, BondDependentTerm):
    type_params: LJLKTypeParams
    global_params: LJLKGlobalParams
    tile_size: int = 32

    def __init__(
        self, type_params: LJLKTypeParams, global_params: LJLKGlobalParams, **kwargs
    ):
        super(LJLKEnergy, self).__init__(
            type_params=type_params, global_params=global_params, **kwargs
        )
        self.type_params = type_params
        self.global_params = global_params
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

    def setup_poses(self, poses: PoseStack):
        super(LJLKEnergy, self).setup_poses(poses)

    def inter_module(
        self,
        packed_block_types: PackedBlockTypes,
        systems: PoseStack,
        context_system_ids: Tensor[int][:, :],
        system_bounding_spheres: Tensor[float][:, :, 4],
        weights,  # map string->Real
    ):
        system_neighbor_list = self.create_block_neighbor_lists(
            systems, system_bounding_spheres
        )
        lj_lk_weights = torch.zeros((2,), dtype=torch.float32, device=self.device)
        lj_lk_weights[0] = weights["lj"] if "lj" in weights else 0
        lj_lk_weights[1] = weights["lk"] if "lk" in weights else 0

        pbt = packed_block_types
        return LJLKInterSystemModule(
            context_system_ids=context_system_ids,
            system_min_block_bondsep=systems.min_block_bondsep,
            system_inter_block_bondsep=systems.inter_block_bondsep,
            system_neighbor_list=system_neighbor_list,
            bt_n_atoms=pbt.n_atoms,
            bt_n_heavy_atoms=pbt.n_heavy_atoms,
            bt_n_heavy_atoms_in_tile=pbt.ljlk_n_heavy_atoms_in_tile,
            bt_heavy_atoms_in_tile=pbt.ljlk_heavy_atoms_in_tile,
            bt_atom_types=pbt.atom_types,
            bt_heavy_atom_inds=pbt.heavy_atom_inds,
            bt_n_interblock_bonds=pbt.n_interblock_bonds,
            bt_atoms_forming_chemical_bonds=pbt.atoms_for_interblock_bonds,
            bt_path_distance=pbt.bond_separation,
            type_params=self.type_params,
            global_params=self.global_params,
            lj_lk_weights=lj_lk_weights,
        )

    def create_block_neighbor_lists(
        self, systems: PoseStack, system_bounding_spheres: Tensor[float][:, :, 4]
    ):
        # we need to make lists of all block pairs within
        # striking distances of each other that we will use to
        # decide which atom-pair calculations to perform during
        # rotamer substititions
        sphere_centers = system_bounding_spheres[:, :, :3].clone().detach()
        n_sys = system_bounding_spheres.shape[0]
        max_n_blocks = system_bounding_spheres.shape[1]
        sphere_centers_1 = sphere_centers.view((n_sys, -1, max_n_blocks, 3))
        sphere_centers_2 = sphere_centers.view((n_sys, max_n_blocks, -1, 3))
        sphere_dists = torch.norm(sphere_centers_1 - sphere_centers_2, dim=3)
        expanded_radii = (
            system_bounding_spheres[:, :, 3] + self.global_params.max_dis / 2
        )
        expanded_radii_1 = expanded_radii.view(n_sys, -1, max_n_blocks)
        expanded_radii_2 = expanded_radii.view(n_sys, max_n_blocks, -1)
        radii_sum = expanded_radii_1 + expanded_radii_2
        spheres_overlap = sphere_dists < radii_sum

        # great -- now how tf are we going to condense this into the lists
        # of neighbors for each block?

        neighbor_counts = torch.sum(spheres_overlap, dim=2)
        max_n_neighbors = torch.max(neighbor_counts)

        neighbor_list = torch.full(
            (n_sys, max_n_blocks, max_n_neighbors),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        nz_spheres_overlap = torch.nonzero(spheres_overlap)
        inc_inds = (
            torch.arange(max_n_neighbors, device=self.device)
            .repeat(n_sys * max_n_blocks)
            .view(n_sys, max_n_blocks, max_n_neighbors)
        )
        store_neighbor = inc_inds < neighbor_counts.view(n_sys, max_n_blocks, 1)

        neighbor_list[
            nz_spheres_overlap[:, 0],
            nz_spheres_overlap[:, 1],
            inc_inds[store_neighbor].view(-1),
        ] = nz_spheres_overlap[:, 2].type(torch.int32)

        return neighbor_list


# class LJLKInterSystemModule(torch.jit.ScriptModule):
class LJLKInterSystemModule:
    def __init__(
        self,
        context_system_ids,
        system_min_block_bondsep,
        system_inter_block_bondsep,
        system_neighbor_list,
        bt_n_atoms,
        bt_n_heavy_atoms,
        bt_n_heavy_atoms_in_tile,
        bt_heavy_atoms_in_tile,
        bt_atom_types,
        bt_heavy_atom_inds,
        bt_n_interblock_bonds,
        bt_atoms_forming_chemical_bonds,
        bt_path_distance,
        type_params,
        global_params,
        lj_lk_weights,
    ):
        super().__init__()

        # torch.set_printoptions(threshold=10000)
        # print("bt_path_distance")
        # print(bt_path_distance)
        # print("bt_atoms_forming_chemical_bonds")
        # print(bt_atoms_forming_chemical_bonds)

        def _p(t):
            return torch.nn.Parameter(t, requires_grad=False)

        def _t(ts):
            return tuple(map(lambda t: t.to(torch.float), ts))

        self.context_system_ids = _p(context_system_ids)
        self.system_min_block_bondsep = _p(system_min_block_bondsep)
        self.system_inter_block_bondsep = _p(system_inter_block_bondsep)
        self.system_neighbor_list = _p(system_neighbor_list)
        self.bt_n_atoms = _p(bt_n_atoms)
        self.bt_n_heavy_atoms = _p(bt_n_heavy_atoms)
        self.bt_n_heavy_atoms_in_tile = _p(bt_n_heavy_atoms_in_tile)
        self.bt_heavy_atoms_in_tile = _p(bt_heavy_atoms_in_tile)
        self.bt_atom_types = _p(bt_atom_types)
        self.bt_heavy_atom_inds = _p(bt_heavy_atom_inds)
        self.bt_n_interblock_bonds = _p(bt_n_interblock_bonds)
        self.bt_atoms_forming_chemical_bonds = _p(bt_atoms_forming_chemical_bonds)
        self.bt_path_distance = _p(bt_path_distance)

        # Pack parameters into dense tensor. Parameter ordering must match
        # struct layout declared in `potentials/params.hh`.
        self.lj_type_params = _p(
            torch.stack(
                _t(
                    [
                        type_params.lj_radius,
                        type_params.lj_wdepth,
                        type_params.is_donor,
                        type_params.is_hydroxyl,
                        type_params.is_polarh,
                        type_params.is_acceptor,
                    ]
                ),
                dim=1,
            )
        )

        # Pack parameters into dense tensor. Parameter ordering must match
        # struct layout declared in `potentials/params.hh`.
        self.lk_type_params = _p(
            torch.stack(
                _t(
                    [
                        type_params.lj_radius,
                        type_params.lk_dgfree,
                        type_params.lk_lambda,
                        type_params.lk_volume,
                        type_params.is_donor,
                        type_params.is_hydroxyl,
                        type_params.is_polarh,
                        type_params.is_acceptor,
                    ]
                ),
                dim=1,
            )
        )

        self.ljlk_type_params = _p(
            torch.stack(
                _t(
                    [
                        type_params.lj_radius,
                        type_params.lj_wdepth,
                        type_params.lk_dgfree,
                        type_params.lk_lambda,
                        type_params.lk_volume,
                        type_params.is_donor,
                        type_params.is_hydroxyl,
                        type_params.is_polarh,
                        type_params.is_acceptor,
                    ]
                ),
                dim=1,
            )
        )

        self.global_params = _p(
            torch.stack(
                _t(
                    [
                        global_params.lj_hbond_dis,
                        global_params.lj_hbond_OH_donor_dis,
                        global_params.lj_hbond_hdis,
                    ]
                ),
                dim=1,
            )
        )
        self.lj_lk_weights = _p(lj_lk_weights)

    def register_with_sim_annealer(
        self,
        context_coords,
        context_block_type,
        alternate_coords,
        alternate_ids,
        output_energies,
        score_event_tensor,
        annealer_event_tensor,
        annealer,
    ):

        register_lj_lk_rotamer_pair_energy_eval(
            context_coords,
            context_block_type,
            alternate_coords,
            alternate_ids,
            self.context_system_ids,
            self.system_min_block_bondsep,
            self.system_inter_block_bondsep,
            self.system_neighbor_list,
            self.bt_n_atoms,
            self.bt_n_heavy_atoms,
            self.bt_n_heavy_atoms_in_tile,
            self.bt_heavy_atoms_in_tile,
            self.bt_atom_types,
            self.bt_heavy_atom_inds,
            self.bt_n_interblock_bonds,
            self.bt_atoms_forming_chemical_bonds,
            self.bt_path_distance,
            self.ljlk_type_params,
            self.global_params,
            self.lj_lk_weights,
            output_energies,
            score_event_tensor,
            annealer_event_tensor,
            annealer,
        )

    # deprecated
    # @torch.jit.script_method
    # def forward(
    def go(self, context_coords, context_block_type, alternate_coords, alternate_ids):
        return score_ljlk_inter_system_scores(
            context_coords,
            context_block_type,
            alternate_coords,
            alternate_ids,
            self.context_system_ids,
            self.system_min_block_bondsep,
            self.system_inter_block_bondsep,
            self.system_neighbor_list,
            self.bt_n_atoms,
            self.bt_n_heavy_atoms,
            self.bt_n_heavy_atoms_in_tile,
            self.bt_heavy_atoms_in_tile,
            self.bt_atom_types,
            self.bt_heavy_atom_inds,
            self.bt_n_interblock_bonds,
            self.bt_atoms_forming_chemical_bonds,
            self.bt_path_distance,
            self.ljlk_type_params,
            self.global_params,
            self.lj_lk_weights,
        )
