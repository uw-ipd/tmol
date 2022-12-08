import torch
import numpy

from ..atom_type_dependent_term import AtomTypeDependentTerm
from ..bond_dependent_term import BondDependentTerm

from tmol.database import ParameterDatabase
from tmol.score.common.stack_condense import tile_subset_indices
from tmol.score.elec.params import ElecParamResolver, ElecGlobalParams
from tmol.score.elec.elec_whole_pose_module import ElecWholePoseScoringModule

from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack
from tmol.types.torch import Tensor

# from tmol.score.ljlk.potentials.compiled import (
#     score_ljlk_inter_system_scores,
#     register_lj_lk_rotamer_pair_energy_eval,
# )


class ElecEnergyTerm(AtomTypeDependentTerm, BondDependentTerm):
    param_resolver: ElecParamResolver
    global_params: ElecGlobalParams

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        param_resolver = ElecParamResolver.from_database(
            param_db.scoring.elec, device=device
        )
        super(ElecEnergyTerm, self).__init__(param_db=param_db, device=device)
        self.param_resolver = param_resolver
        self.global_params = self.param_resolver.global_params

    @classmethod
    def score_types(cls):
        import tmol.score.terms.elec_creator

        return tmol.score.terms.elec_creator.ElecTermCreator.score_types()

    def n_bodies(self):
        return 2

    def setup_block_type(self, block_type: RefinedResidueType):
        super(ElecEnergyTerm, self).setup_block_type(block_type)
        if hasattr(block_type, "elec_inter_repr_path_distance"):
            assert hasattr(block_type, "elec_intra_repr_path_distance")
            assert hasattr(block_type, "elec_partial_charge")
            return
        partial_charge = numpy.zeros((block_type.n_atoms,), dtype=numpy.float32)

        for i, atname in enumerate(block_type.atoms):
            if (block_type.name, atname.name) in self.param_resolver.partial_charges:
                partial_charge[i] = self.param_resolver.partial_charges[
                    (block_type.name, atname.name)
                ]
        # print("partial_charge", block_type.name)
        # print(partial_charge)

        # count pair representative logic:
        # decide whether or not two atoms i and j should have their interaction counted
        # on the basis of the number of chemical bonds that separate their respective
        # representative atoms r(i) and r(j).
        # We will create a tensor to measure the distance of all representative atoms to:
        # all connection atoms.
        # let rpd be the acronym for "representative_path_distance"
        # rpd[a,b] will be the number of chemical bonds between atom a and atom r(b). Then
        # we can answer how far apart the representative atoms are for atoms i and j on
        # residues k and l by computinging:
        # min(ci cj, rpd_i[ci, i] + rpd_j[cj, j] + sep(ci, cj))
        # over all pairs of connection atoms ci and cj on residues i and j
        # NOTE: the connection atoms must be their own representatives for this to work

        representative_mapping = numpy.arange(block_type.n_atoms, dtype=numpy.int32)
        if block_type.name in self.param_resolver.cp_reps_by_res:
            for outer, inner in self.param_resolver.cp_reps_by_res[
                block_type.name
            ].items():
                if (
                    outer not in block_type.atom_to_idx
                    or inner not in block_type.atom_to_idx
                ):
                    continue
                representative_mapping[
                    block_type.atom_to_idx[inner]
                ] = block_type.atom_to_idx[outer]

        # print("representative_mapping")
        # print(representative_mapping)
        # inter_rep_path_dist = numpy.copy(block_type.path_distance)
        # intra_rep_path_dist = numpy.copy(block_type.path_distance)
        # inter_rep_path_dist[:, representative_mapping] = block_type.path_distance
        # intra_rep_path_dist[representative_mapping, :] = inter_rep_path_dist

        inter_rep_path_dist = block_type.path_distance[:, representative_mapping]
        intra_rep_path_dist = inter_rep_path_dist[representative_mapping, :]

        # representative_path_distance1 = block_type.path_distance[:, representative_mapping]
        # representative_path_distance = representative_path_distance1[representative_mapping, :]

        # print("path_distance", block_type.name)
        # print([(i, n.name) for i, n in enumerate(block_type.atoms)])
        # print(block_type.path_distance)
        #
        # print("inter_representative_path_distance", block_type.name)
        # print(inter_rep_path_dist)
        # print("intra_representative_path_distance", block_type.name)
        # print(intra_rep_path_dist)

        setattr(block_type, "elec_partial_charge", partial_charge)
        setattr(block_type, "elec_inter_repr_path_distance", inter_rep_path_dist)
        setattr(block_type, "elec_intra_repr_path_distance", intra_rep_path_dist)

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(ElecEnergyTerm, self).setup_packed_block_types(packed_block_types)
        if hasattr(packed_block_types, "elec_inter_repr_path_distance"):
            assert hasattr(packed_block_types, "elec_intra_repr_path_distance")
            assert hasattr(packed_block_types, "elec_partial_charge")
            return

        def _ti(arr):
            return torch.tensor(arr, dtype=torch.int32, device=self.device)

        def _tf(arr):
            return torch.tensor(arr, dtype=torch.float32, device=self.device)

        pbt = packed_block_types
        elec_partial_charge = torch.zeros(
            (pbt.n_types, pbt.max_n_atoms), dtype=torch.float32, device=self.device
        )
        elec_inter_repr_path_distance = torch.zeros(
            (pbt.n_types, pbt.max_n_atoms, pbt.max_n_atoms),
            dtype=torch.int32,
            device=self.device,
        )
        elec_intra_repr_path_distance = torch.zeros(
            (pbt.n_types, pbt.max_n_atoms, pbt.max_n_atoms),
            dtype=torch.int32,
            device=self.device,
        )

        for i, bt in enumerate(packed_block_types.active_block_types):
            elec_partial_charge[i, : bt.n_atoms] = _tf(bt.elec_partial_charge)
            elec_inter_repr_path_distance[i, : bt.n_atoms, : bt.n_atoms] = _ti(
                bt.elec_inter_repr_path_distance
            )
            elec_intra_repr_path_distance[i, : bt.n_atoms, : bt.n_atoms] = _ti(
                bt.elec_intra_repr_path_distance
            )

        # print("elec_partial_charge")
        # print(elec_partial_charge)
        # print("elec_inter_representative_path_distance")
        # print(elec_inter_repr_path_distance)
        # print("elec_intra_representative_path_distance")
        # print(elec_intra_repr_path_distance)

        setattr(packed_block_types, "elec_partial_charge", elec_partial_charge)
        setattr(
            packed_block_types,
            "elec_inter_repr_path_distance",
            elec_inter_repr_path_distance,
        )
        setattr(
            packed_block_types,
            "elec_intra_repr_path_distance",
            elec_intra_repr_path_distance,
        )

    def setup_poses(self, poses: PoseStack):
        super(ElecEnergyTerm, self).setup_poses(poses)

    def render_whole_pose_scoring_module(self, pose_stack: PoseStack):
        pbt = pose_stack.packed_block_types
        return ElecWholePoseScoringModule(
            pose_stack_block_coord_offset=pose_stack.block_coord_offset,
            pose_stack_block_types=pose_stack.block_type_ind,
            pose_stack_min_block_bondsep=pose_stack.min_block_bondsep,
            pose_stack_inter_block_bondsep=pose_stack.inter_block_bondsep,
            bt_n_atoms=pbt.n_atoms,
            bt_partial_charge=pbt.elec_partial_charge,
            bt_n_interblock_bonds=pbt.n_interblock_bonds,
            bt_atoms_forming_chemical_bonds=pbt.atoms_for_interblock_bonds,
            bt_inter_repr_path_distance=pbt.elec_inter_repr_path_distance,
            bt_intra_repr_path_distance=pbt.elec_intra_repr_path_distance,
            global_params=self.global_params,
        )

    # def render_inter_module(
    #     self,
    #     packed_block_types: PackedBlockTypes,
    #     systems: PoseStack,
    #     context_system_ids: Tensor[int][:, :],
    #     system_bounding_spheres: Tensor[float][:, :, 4],
    #     weights,  # map string->Real
    # ):
    #     system_neighbor_list = self.create_block_neighbor_lists(
    #         systems, system_bounding_spheres
    #     )
    #     lj_lk_weights = torch.zeros((2,), dtype=torch.float32, device=self.device)
    #     lj_lk_weights[0] = weights["lj"] if "lj" in weights else 0
    #     lj_lk_weights[1] = weights["lk"] if "lk" in weights else 0
    #
    #     pbt = packed_block_types
    #     return LJLKInterSystemModule(
    #         context_system_ids=context_system_ids,
    #         system_min_block_bondsep=systems.min_block_bondsep,
    #         system_inter_block_bondsep=systems.inter_block_bondsep,
    #         system_neighbor_list=system_neighbor_list,
    #         bt_n_atoms=pbt.n_atoms,
    #         bt_n_heavy_atoms=pbt.n_heavy_atoms,
    #         bt_n_heavy_atoms_in_tile=pbt.ljlk_n_heavy_atoms_in_tile,
    #         bt_heavy_atoms_in_tile=pbt.ljlk_heavy_atoms_in_tile,
    #         bt_atom_types=pbt.atom_types,
    #         bt_heavy_atom_inds=pbt.heavy_atom_inds,
    #         bt_n_interblock_bonds=pbt.n_interblock_bonds,
    #         bt_atoms_forming_chemical_bonds=pbt.atoms_for_interblock_bonds,
    #         bt_path_distance=pbt.bond_separation,
    #         type_params=self.type_params,
    #         global_params=self.global_params,
    #         lj_lk_weights=lj_lk_weights,
    #     )

    # def create_block_neighbor_lists(
    #     self, systems: PoseStack, system_bounding_spheres: Tensor[float][:, :, 4]
    # ):
    #     # we need to make lists of all block pairs within
    #     # striking distances of each other that we will use to
    #     # decide which atom-pair calculations to perform during
    #     # rotamer substititions
    #     sphere_centers = system_bounding_spheres[:, :, :3].clone().detach()
    #     n_sys = system_bounding_spheres.shape[0]
    #     max_n_blocks = system_bounding_spheres.shape[1]
    #     sphere_centers_1 = sphere_centers.view((n_sys, -1, max_n_blocks, 3))
    #     sphere_centers_2 = sphere_centers.view((n_sys, max_n_blocks, -1, 3))
    #     sphere_dists = torch.norm(sphere_centers_1 - sphere_centers_2, dim=3)
    #     expanded_radii = (
    #         system_bounding_spheres[:, :, 3] + self.global_params.max_dis / 2
    #     )
    #     expanded_radii_1 = expanded_radii.view(n_sys, -1, max_n_blocks)
    #     expanded_radii_2 = expanded_radii.view(n_sys, max_n_blocks, -1)
    #     radii_sum = expanded_radii_1 + expanded_radii_2
    #     spheres_overlap = sphere_dists < radii_sum
    #
    #     # great -- now how tf are we going to condense this into the lists
    #     # of neighbors for each block?
    #
    #     neighbor_counts = torch.sum(spheres_overlap, dim=2)
    #     max_n_neighbors = torch.max(neighbor_counts)
    #
    #     neighbor_list = torch.full(
    #         (n_sys, max_n_blocks, max_n_neighbors),
    #         -1,
    #         dtype=torch.int32,
    #         device=self.device,
    #     )
    #     nz_spheres_overlap = torch.nonzero(spheres_overlap)
    #     inc_inds = (
    #         torch.arange(max_n_neighbors, device=self.device)
    #         .repeat(n_sys * max_n_blocks)
    #         .view(n_sys, max_n_blocks, max_n_neighbors)
    #     )
    #     store_neighbor = inc_inds < neighbor_counts.view(n_sys, max_n_blocks, 1)
    #
    #     neighbor_list[
    #         nz_spheres_overlap[:, 0],
    #         nz_spheres_overlap[:, 1],
    #         inc_inds[store_neighbor].view(-1),
    #     ] = nz_spheres_overlap[:, 2].type(torch.int32)
    #
    #     return neighbor_list


# class LJLKInterSystemModule(torch.jit.ScriptModule):
# class LJLKInterSystemModule:
#     def __init__(
#         self,
#         context_system_ids,
#         system_min_block_bondsep,
#         system_inter_block_bondsep,
#         system_neighbor_list,
#         bt_n_atoms,
#         bt_n_heavy_atoms,
#         bt_n_heavy_atoms_in_tile,
#         bt_heavy_atoms_in_tile,
#         bt_atom_types,
#         bt_heavy_atom_inds,
#         bt_n_interblock_bonds,
#         bt_atoms_forming_chemical_bonds,
#         bt_path_distance,
#         type_params,
#         global_params,
#         lj_lk_weights,
#     ):
#         super().__init__()
#
#         def _p(t):
#             return torch.nn.Parameter(t, requires_grad=False)
#
#         def _t(ts):
#             return tuple(map(lambda t: t.to(torch.float), ts))
#
#         self.context_system_ids = _p(context_system_ids)
#         self.system_min_block_bondsep = _p(system_min_block_bondsep)
#         self.system_inter_block_bondsep = _p(system_inter_block_bondsep)
#         self.system_neighbor_list = _p(system_neighbor_list)
#         self.bt_n_atoms = _p(bt_n_atoms)
#         self.bt_n_heavy_atoms = _p(bt_n_heavy_atoms)
#         self.bt_n_heavy_atoms_in_tile = _p(bt_n_heavy_atoms_in_tile)
#         self.bt_heavy_atoms_in_tile = _p(bt_heavy_atoms_in_tile)
#         self.bt_atom_types = _p(bt_atom_types)
#         self.bt_heavy_atom_inds = _p(bt_heavy_atom_inds)
#         self.bt_n_interblock_bonds = _p(bt_n_interblock_bonds)
#         self.bt_atoms_forming_chemical_bonds = _p(bt_atoms_forming_chemical_bonds)
#         self.bt_path_distance = _p(bt_path_distance)
#
#         # Pack parameters into dense tensor. Parameter ordering must match
#         # struct layout declared in `potentials/params.hh`.
#         self.lj_type_params = _p(
#             torch.stack(
#                 _t(
#                     [
#                         type_params.lj_radius,
#                         type_params.lj_wdepth,
#                         type_params.is_donor,
#                         type_params.is_hydroxyl,
#                         type_params.is_polarh,
#                         type_params.is_acceptor,
#                     ]
#                 ),
#                 dim=1,
#             )
#         )
#
#         # Pack parameters into dense tensor. Parameter ordering must match
#         # struct layout declared in `potentials/params.hh`.
#         self.lk_type_params = _p(
#             torch.stack(
#                 _t(
#                     [
#                         type_params.lj_radius,
#                         type_params.lk_dgfree,
#                         type_params.lk_lambda,
#                         type_params.lk_volume,
#                         type_params.is_donor,
#                         type_params.is_hydroxyl,
#                         type_params.is_polarh,
#                         type_params.is_acceptor,
#                     ]
#                 ),
#                 dim=1,
#             )
#         )
#
#         self.ljlk_type_params = _p(
#             torch.stack(
#                 _t(
#                     [
#                         type_params.lj_radius,
#                         type_params.lj_wdepth,
#                         type_params.lk_dgfree,
#                         type_params.lk_lambda,
#                         type_params.lk_volume,
#                         type_params.is_donor,
#                         type_params.is_hydroxyl,
#                         type_params.is_polarh,
#                         type_params.is_acceptor,
#                     ]
#                 ),
#                 dim=1,
#             )
#         )
#
#         self.global_params = _p(
#             torch.stack(
#                 _t(
#                     [
#                         global_params.lj_hbond_dis,
#                         global_params.lj_hbond_OH_donor_dis,
#                         global_params.lj_hbond_hdis,
#                     ]
#                 ),
#                 dim=1,
#             )
#         )
#         self.lj_lk_weights = _p(lj_lk_weights)
#
#     def register_with_sim_annealer(
#         self,
#         context_coords,
#         context_coord_offsets,
#         context_block_type,
#         alternate_coords,
#         alternate_coord_offsets,
#         alternate_ids,
#         output_energies,
#         score_event_tensor,
#         annealer_event_tensor,
#         annealer,
#     ):
#         register_lj_lk_rotamer_pair_energy_eval(
#             context_coords,
#             context_coord_offsets,
#             context_block_type,
#             alternate_coords,
#             alternate_coord_offsets,
#             alternate_ids,
#             self.context_system_ids,
#             self.system_min_block_bondsep,
#             self.system_inter_block_bondsep,
#             self.system_neighbor_list,
#             self.bt_n_atoms,
#             self.bt_n_heavy_atoms,
#             self.bt_n_heavy_atoms_in_tile,
#             self.bt_heavy_atoms_in_tile,
#             self.bt_atom_types,
#             self.bt_heavy_atom_inds,
#             self.bt_n_interblock_bonds,
#             self.bt_atoms_forming_chemical_bonds,
#             self.bt_path_distance,
#             self.ljlk_type_params,
#             self.global_params,
#             self.lj_lk_weights,
#             output_energies,
#             score_event_tensor,
#             annealer_event_tensor,
#             annealer,
#         )
#
#     # deprecated
#     # @torch.jit.script_method
#     # def forward(
#     def go(
#         self,
#         context_coords,
#         context_coord_offsets,
#         context_block_type,
#         alternate_coords,
#         alternate_coord_offsets,
#         alternate_ids,
#     ):
#         return score_ljlk_inter_system_scores(
#             context_coords,
#             context_coord_offsets,
#             context_block_type,
#             alternate_coords,
#             alternate_coord_offsets,
#             alternate_ids,
#             self.context_system_ids,
#             self.system_min_block_bondsep,
#             self.system_inter_block_bondsep,
#             self.system_neighbor_list,
#             self.bt_n_atoms,
#             self.bt_n_heavy_atoms,
#             self.bt_n_heavy_atoms_in_tile,
#             self.bt_heavy_atoms_in_tile,
#             self.bt_atom_types,
#             self.bt_heavy_atom_inds,
#             self.bt_n_interblock_bonds,
#             self.bt_atoms_forming_chemical_bonds,
#             self.bt_path_distance,
#             self.ljlk_type_params,
#             self.global_params,
#             self.lj_lk_weights,
#         )
