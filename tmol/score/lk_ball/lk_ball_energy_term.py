import attr
import torch

from ..atom_type_dependent_term import AtomTypeDependentTerm
from ..bond_dependent_term import BondDependentTerm
from ..hbond.hbond_dependent_term import HBondDependentTerm
from ..ljlk.params import LJLKGlobalParams, LJLKParamResolver

from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack
from tmol.types.torch import Tensor


@attr.s(auto_attribs=True)
class LKBallEnergy(HBondDependentTerm, AtomTypeDependentTerm, BondDependentTerm):

    ljlk_global_params: LJLKGlobalParams
    ljlk_param_resolver: LJLKParamResolver

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        ljlk_param_resolver = LJLKParamResolver.from_database(
            param_db.chemical, param_db.scoring.ljlk, device=device
        )
        super(LJLKEnergyTerm, self).__init__(param_db=param_db, device=device)
        self.type_params = ljlk_param_resolver.type_params
        self.global_params = ljlk_param_resolver.global_params
        self.tile_size = LJLKEnergyTerm.tile_size

    def setup_block_type(self, block_type: RefinedResidueType):
        super(LKBallEnergy, self).setup_block_type(block_type)

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(LKBallEnergy, self).setup_packed_block_types(packed_block_types)

    def setup_poses(self, pose_stack: PoseStack):
        super(LKBallEnergy, self).setup_poses(pose_stack)

    def inter_module(
        self,
        packed_block_types: PackedBlockTypes,
        pose_stack: PoseStack,
        context_system_ids: Tensor[int][:, :],
        system_bounding_spheres: Tensor[float][:, :, 4],
        weights,  # map string->Real
    ):
        system_neighbor_list = self.create_block_neighbor_lists(system_bounding_spheres)
        lkb_weight = torch.zeros((1,), dtype=torch.float32, device=self.device)
        lkb_weight[0] = weights["lk_ball"] if "lk_ball" in weights else 0

        pbt = packed_block_types
        return LKBallInterSystemModule(
            context_system_ids=context_system_ids,
            system_min_block_bondsep=pose_stack.min_block_bondsep,
            system_inter_block_bondsep=pose_stack.inter_block_bondsep,
            system_neighbor_list=system_neighbor_list,
            bt_n_heavy_atoms=pbt.n_heavy_atoms,
            bt_atom_types=pbt.atom_types,
            bt_heavy_atom_inds=pbt.heavy_atom_inds,
            bt_n_interblock_bonds=pbt.n_interblock_bonds,
            bt_atoms_forming_chemical_bonds=pbt.atoms_for_interblock_bonds,
            bt_path_distance=pbt.bond_separation,
            bt_is_acceptor=pbt.hbpbt_params.is_acceptor,
            bt_acceptor_type=pbt.hbpbt_params.acceptor_type,
            bt_acceptor_hybridization=pbt.hbpbt_params.acceptor_hybridization,
            bt_acceptor_base_inds=pbt.hbpbt_params.acceptor_base_inds,
            bt_is_donor=pbt.hbpbt_params.is_donor,
            bt_donor_type=pbt.hbpbt_params.donor_type,
            bt_donor_attached_hydrogens=pbt.hbpbt_params.donor_attached_hydrogens,
            param_resolver=self.ljlk_param_resolver,
            lkb_weight=lkb_weight,
        )

    def create_block_neighbor_lists(
        self, system_bounding_spheres: Tensor[float][:, :, 4]
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
            system_bounding_spheres[:, :, 3] + self.ljlk_global_params.max_dis / 2
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


class LKBallInterSystemModule(torch.jit.ScriptModule):
    def __init__(
        self,
        context_system_ids,
        system_min_block_bondsep,
        system_inter_block_bondsep,
        system_neighbor_list,
        bt_n_heavy_atoms,
        bt_atom_types,
        bt_heavy_atom_inds,
        bt_n_interblock_bonds,
        bt_atoms_forming_chemical_bonds,
        bt_path_distance,
        bt_is_acceptor,
        bt_acceptor_type,
        bt_acceptor_hybridization,
        bt_acceptor_base_inds,
        bt_is_donor,
        bt_donor_type,
        bt_donor_attached_hydrogens,
        param_resolver: LJLKParamResolver,
        lkb_weight,
    ):
        super().__init__()

        def _p(t):
            return torch.nn.Parameter(t, requires_grad=False)

        def _t(ts):
            return tuple(map(lambda t: t.to(torch.float), ts))

        # TEMP!
        self.context_water_coords = _p(
            torch.zeros(
                (1, 1, 1, 4, 3), dtype=torch.float32, device=context_system_ids.device
            )
        )

        self.context_system_ids = _p(context_system_ids)
        self.system_min_block_bondsep = _p(system_min_block_bondsep)
        self.system_inter_block_bondsep = _p(system_inter_block_bondsep)
        self.system_neighbor_list = _p(system_neighbor_list)

        # self.bt_n_atoms = _p(bt_n_atoms)
        self.bt_n_heavy_atoms = _p(bt_n_heavy_atoms)
        self.bt_atom_types = _p(bt_atom_types)
        self.bt_heavy_atom_inds = _p(bt_heavy_atom_inds)
        self.bt_n_interblock_bonds = _p(bt_n_interblock_bonds)
        self.bt_atoms_forming_chemical_bonds = _p(bt_atoms_forming_chemical_bonds)
        self.bt_path_distance = _p(bt_path_distance)

        self.bt_is_acceptor = _p(bt_is_acceptor)
        self.bt_acceptor_type = _p(bt_acceptor_type)
        self.bt_acceptor_hybridization = _p(bt_acceptor_hybridization)
        self.bt_acceptor_base_inds = _p(bt_acceptor_base_inds)
        self.bt_is_donor = _p(bt_is_donor)
        self.bt_donor_type = _p(bt_donor_type)
        self.bt_donor_attached_hydrogens = _p(bt_donor_attached_hydrogens)

        self.lkball_global_params = _p(
            torch.stack(
                _t(
                    [
                        param_resolver.global_params.lj_hbond_dis,
                        param_resolver.global_params.lj_hbond_OH_donor_dis,
                        param_resolver.global_params.lj_hbond_hdis,
                        param_resolver.global_params.lkb_water_dist,
                    ]
                ),
                dim=1,
            )
        )

        # Pack parameters into dense tensor. Parameter ordering must match
        # struct layout declared in `potentials/params.hh`.
        self.lj_type_params = _p(
            torch.stack(
                _t(
                    [
                        param_resolver.type_params.lj_radius,
                        param_resolver.type_params.lj_wdepth,
                        param_resolver.type_params.is_donor,
                        param_resolver.type_params.is_hydroxyl,
                        param_resolver.type_params.is_polarh,
                        param_resolver.type_params.is_acceptor,
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
                        param_resolver.type_params.lj_radius,
                        param_resolver.type_params.lk_dgfree,
                        param_resolver.type_params.lk_lambda,
                        param_resolver.type_params.lk_volume,
                        param_resolver.type_params.is_donor,
                        param_resolver.type_params.is_hydroxyl,
                        param_resolver.type_params.is_polarh,
                        param_resolver.type_params.is_acceptor,
                    ]
                ),
                dim=1,
            )
        )

        self.watergen_water_tors_sp2 = torch.nn.Parameter(
            param_resolver.global_params.lkb_water_tors_sp2, requires_grad=False
        )
        self.watergen_water_tors_sp3 = torch.nn.Parameter(
            param_resolver.global_params.lkb_water_tors_sp3, requires_grad=False
        )
        self.watergen_water_tors_ring = torch.nn.Parameter(
            param_resolver.global_params.lkb_water_tors_ring, requires_grad=False
        )

        # self.global_params = _p(
        #     torch.stack(
        #         _t(
        #             [
        #                 global_params.lj_hbond_dis,
        #                 global_params.lj_hbond_OH_donor_dis,
        #                 global_params.lj_hbond_hdis,
        #             ]
        #         ),
        #         dim=1,
        #     )
        # )

        self.lkb_weight = _p(lkb_weight)

    @torch.jit.script_method
    def forward(
        self, context_coords, context_block_type, alternate_coords, alternate_ids
    ):
        return torch.ops.tmol.score_lkball_inter_system_scores(
            context_coords,
            context_block_type,
            alternate_coords,
            alternate_ids,
            self.context_water_coords,
            self.context_system_ids,
            self.system_min_block_bondsep,
            self.system_inter_block_bondsep,
            self.system_neighbor_list,
            # self.bt_n_atoms,
            # self.bt_n_heavy_atoms,
            # self.bt_atom_types,
            # self.bt_heavy_atom_inds,
            # self.bt_n_interblock_bonds,
            # self.bt_atoms_forming_chemical_bonds,
            # self.bt_path_distance,
            self.bt_is_acceptor,
            self.bt_acceptor_type,
            self.bt_acceptor_hybridization,
            self.bt_acceptor_base_inds,
            self.bt_is_donor,
            self.bt_donor_type,
            self.bt_donor_attached_hydrogens,
            # self.lj_type_params,
            # self.lk_type_params,
            # self.global_params,
            self.lkball_global_params,
            self.watergen_water_tors_sp2,
            self.watergen_water_tors_sp3,
            self.watergen_water_tors_ring,
            self.lkb_weight,
        )
