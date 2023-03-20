import numpy
import torch
from tmol.types.array import NDArray
from tmol.types.torch import Tensor

from tmol.score.energy_term import EnergyTerm
from .params import (
    RamaBlockTypeParams,
    RamaPackedBlockTypesParams,
    RamaParams,
    RamaParamResolver,
)
from .rama_whole_pose_module import RamaWholePoseScoringModule
from ..atom_type_dependent_term import AtomTypeDependentTerm
from ..hbond.hbond_dependent_term import HBondDependentTerm
from ..ljlk.params import LJLKGlobalParams, LJLKParamResolver
from tmol.database import ParameterDatabase

from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack
from tmol.score.common.stack_condense import arg_tile_subset_indices


class RamaEnergyTerm(EnergyTerm):
    device: torch.device
    param_resolver: RamaParamResolver
    tables: Tensor[torch.float32][:, :, :]
    table_params: Tensor[torch.float32][:, :, :]

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super(RamaEnergyTerm, self).__init__(param_db=param_db, device=device)
        self.device = device

        self.param_resolver = RamaParamResolver.from_database(
            param_db.scoring.rama, device
        )

        self.tables = self.param_resolver.rama_params.tables
        self.table_params = torch.cat(
            (
                self.param_resolver.rama_params.bbstarts.to(torch.float32),
                self.param_resolver.rama_params.bbsteps.to(torch.float32),
            ),
            dim=1,
        )

    @classmethod
    def score_types(cls):
        import tmol.score.terms.rama_creator

        return tmol.score.terms.rama_creator.RamaTermCreator.score_types()

    def n_bodies(self):
        return 2

    def setup_block_type(self, block_type: RefinedResidueType):
        super(RamaEnergyTerm, self).setup_block_type(block_type)
        if hasattr(block_type, "rama_params"):
            return

        rname = block_type.name
        lookups = numpy.array([[rname, "_"], [rname, "PRO"]], dtype=object)
        table_inds = self.param_resolver.rama_lookup.index.get_indexer(lookups)

        rama_torsion_atoms = numpy.full((2, 4, 3), -1, dtype=numpy.int32)

        def uaids_to_np(tor_uaids):
            uaids = numpy.full((4, 3), -1, dtype=numpy.int32)
            for i in range(4):
                uaids[i, :] = numpy.array(tor_uaids[i], dtype=numpy.int32)
            return uaids

        if table_inds[0] != -1 or table_inds[1] != -1:
            for i, tor in enumerate(["phi", "psi"]):
                if tor in block_type.torsion_to_uaids:
                    rama_torsion_atoms[i] = uaids_to_np(
                        block_type.torsion_to_uaids[tor]
                    )
        rama_torsion_atoms = rama_torsion_atoms.reshape(8, 3)

        upper_conn_id = numpy.full((1,), -1, dtype=numpy.int32)
        if "up" in block_type.connection_to_cidx:
            upper_conn_id[0] = block_type.connection_to_cidx["up"]
        is_pro = numpy.full((1,), 0, dtype=numpy.int32)

        # TO DO: Better logic here to handle proline variants
        if block_type.name == "PRO":
            is_pro[0] = 1

        bt_rama_params = RamaBlockTypeParams(
            table_inds=table_inds,
            upper_conn_ind=upper_conn_id,
            is_pro=is_pro,
            rama_torsion_atoms=rama_torsion_atoms,
        )
        setattr(block_type, "rama_params", bt_rama_params)

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(RamaEnergyTerm, self).setup_packed_block_types(packed_block_types)
        if hasattr(packed_block_types, "rama_params"):
            return
        n_types = packed_block_types.n_types

        bt_table = numpy.full((n_types, 2), -1, dtype=numpy.int32)
        bt_upper_conn_ind = numpy.full((n_types,), -1, dtype=numpy.int32)
        bt_is_pro = numpy.full((n_types,), -1, dtype=numpy.int32)
        bt_torsion_atoms = numpy.full((n_types, 8, 3), 0, dtype=numpy.int32)

        for i, bt in enumerate(packed_block_types.active_block_types):
            i_rama_params = bt.rama_params
            bt_table[i] = i_rama_params.table_inds
            bt_upper_conn_ind[i] = i_rama_params.upper_conn_ind
            bt_is_pro[i] = i_rama_params.is_pro
            bt_torsion_atoms[i] = i_rama_params.rama_torsion_atoms

        def _t(t):
            return torch.tensor(t, device=packed_block_types.device)

        rama_params = RamaPackedBlockTypesParams(
            bt_table=_t(bt_table),
            bt_upper_conn_ind=_t(bt_upper_conn_ind),
            bt_is_pro=_t(bt_is_pro),
            bt_torsion_atoms=_t(bt_torsion_atoms),
        )
        setattr(packed_block_types, "rama_params", rama_params)

    def setup_poses(self, pose_stack: PoseStack):
        super(RamaEnergyTerm, self).setup_poses(pose_stack)

    def render_whole_pose_scoring_module(self, pose_stack: PoseStack):
        pbt = pose_stack.packed_block_types
        ljlk_global_params = self.ljlk_param_resolver.global_params

        return RamaWholePoseScoringModule(
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
            bt_tile_don_hvy_inds=pbt.hbpbt_params.tile_donH_hvy_inds,
            bt_tile_which_donH_for_hvy=pbt.hbpbt_params.tile_which_donH_of_donH_hvy,
            bt_tile_acc_inds=pbt.hbpbt_params.tile_acc_inds,
            bt_tile_hybridization=pbt.hbpbt_params.tile_acceptor_hybridization,
            bt_tile_acc_n_attached_H=pbt.hbpbt_params.tile_acceptor_n_attached_H,
            bt_atom_is_hydrogen=pbt.hbpbt_params.is_hydrogen,
            bt_tile_n_polar_atoms=pbt.rama_params.tile_n_polar_atoms,
            bt_tile_n_occluder_atoms=pbt.rama_params.tile_n_occluder_atoms,
            bt_tile_pol_occ_inds=pbt.rama_params.tile_pol_occ_inds,
            bt_tile_rama_params=pbt.rama_params.tile_rama_params,
            bt_path_distance=pbt.bond_separation,
            rama_global_params=self.stack_rama_global_params(),
            water_gen_global_params=self.stack_rama_water_gen_global_params(),
            sp2_water_tors=ljlk_global_params.lkb_water_tors_sp2,
            sp3_water_tors=ljlk_global_params.lkb_water_tors_sp3,
            ring_water_tors=ljlk_global_params.lkb_water_tors_ring,
        )

    def _tfloat(self, ts):
        return tuple(map(lambda t: t.to(torch.float), ts))

    def stack_rama_global_params(self):
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

    def stack_rama_water_gen_global_params(self):
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
