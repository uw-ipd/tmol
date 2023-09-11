import attr

import torch
import numpy

from ..energy_term import EnergyTerm

from tmol.database import ParameterDatabase
from tmol.score.dunbrack.dunbrack_whole_pose_module import (
    DunbrackWholePoseScoringModule,
)
from tmol.score.dunbrack.params import DunbrackParamResolver
from tmol.score.dunbrack.params import ScoringDunbrackDatabaseView

from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack
from tmol.types.torch import Tensor

from itertools import count
from functools import partial
from dataclasses import dataclass
import dataclasses


@dataclass
class DunbrackBlockAttrs:
    n_dihedrals: int
    phi_uaids: numpy.ndarray
    psi_uaids: numpy.ndarray
    chi_uaids: numpy.ndarray
    dih_uaids: numpy.ndarray
    rotamer_table_set: int
    rotameric_index: int
    semirotameric_index: int
    n_chi: int
    n_rotameric_chi: int
    probability_table_offset: int
    mean_table_offset: int
    rotamer_index_to_table_index_offset: int
    semirotameric_tableset_offset: int


class DunbrackEnergyTerm(EnergyTerm):
    device: torch.device  # = attr.ib()

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super(DunbrackEnergyTerm, self).__init__(param_db=param_db, device=device)

        self.global_params = DunbrackParamResolver.from_database(
            param_db.scoring.dun, device
        )
        self.dunbrack_db = [
            getattr(self.global_params.scoring_db, field.name)
            for field in attr.fields(ScoringDunbrackDatabaseView)
        ]
        self.device = device

    @classmethod
    def score_types(cls):
        import tmol.score.terms.dunbrack_creator

        return tmol.score.terms.dunbrack_creator.DunbrackTermCreator.score_types()

    def n_bodies(self):
        return 1

    def setup_block_type(self, block_type: RefinedResidueType):
        super(DunbrackEnergyTerm, self).setup_block_type(block_type)

        if hasattr(block_type, "dunbrack_attrs"):
            return

        inds = self.global_params.all_table_indices.index.get_indexer(
            [block_type.base_name]
        )
        r_inds = self.global_params.rotameric_table_indices.index.get_indexer(
            [block_type.base_name]
        )
        s_inds = self.global_params.semirotameric_table_indices.index.get_indexer(
            [block_type.base_name]
        )

        inds[inds != -1] = self.global_params.all_table_indices.iloc[inds[inds != -1]][
            "dun_table_name"
        ].values
        r_inds[r_inds != -1] = self.global_params.rotameric_table_indices.iloc[
            r_inds[r_inds != -1]
        ]["dun_table_name"].values
        s_inds[s_inds != -1] = self.global_params.semirotameric_table_indices.iloc[
            s_inds[s_inds != -1]
        ]["dun_table_name"].values

        rotamer_table_set = inds[0]
        rotameric_index = r_inds[0]
        semirotameric_index = s_inds[0]
        semirotameric = semirotameric_index != -1

        semirotameric_tableset_offset = (
            numpy.array(-1)
            if not semirotameric
            else self.global_params.scoring_db_aux.semirotameric_tableset_offsets[
                s_inds[s_inds != -1]
            ][0]
        )

        empty_tor = numpy.full((4, 3), -1, dtype=numpy.int32)

        phi_uaids = self.get_torsion("phi", block_type)
        if phi_uaids is None:
            phi_uaids = empty_tor

        psi_uaids = self.get_torsion("psi", block_type)
        if psi_uaids is None:
            psi_uaids = empty_tor

        chis = []
        n = count(1)
        while (t := self.get_torsion("chi" + str(next(n)), block_type)) is not None:
            chis += [t]
        chi_uaids = numpy.array(chis) if len(chis) > 0 else None

        dih_uaids = numpy.array([phi_uaids] + [psi_uaids] + chis)

        n_chi = self.global_params.scoring_db_aux.nchi_for_table_set[rotamer_table_set]
        n_rotameric_chi = n_chi - (1 if semirotameric else 0)
        n_dihedrals = n_chi + 2

        probability_table_offset = (
            self.global_params.scoring_db_aux.rotameric_prob_tableset_offsets[
                rotameric_index
            ]
        )

        mean_table_offset = (
            self.global_params.scoring_db_aux.rotameric_meansdev_tableset_offsets[
                rotamer_table_set
            ]
        )
        rotamer_index_to_table_index_offset = (
            self.global_params.scoring_db_aux.rotameric_chi_ri2ti_offsets[
                rotamer_table_set
            ]
        )

        dunbrack_attrs = DunbrackBlockAttrs(
            n_dihedrals=n_dihedrals,
            phi_uaids=phi_uaids,
            psi_uaids=psi_uaids,
            chi_uaids=chi_uaids,
            dih_uaids=dih_uaids,
            rotamer_table_set=rotamer_table_set,
            rotameric_index=rotameric_index,
            semirotameric_index=semirotameric_index,
            n_chi=n_chi,
            n_rotameric_chi=n_rotameric_chi,
            probability_table_offset=probability_table_offset,
            mean_table_offset=mean_table_offset,
            rotamer_index_to_table_index_offset=rotamer_index_to_table_index_offset,
            semirotameric_tableset_offset=semirotameric_tableset_offset,
        )

        setattr(block_type, "dunbrack_attrs", dunbrack_attrs)

    def get_torsion(self, name, block_type):
        if name in block_type.torsion_to_uaids:
            return numpy.array(block_type.torsion_to_uaids[name], dtype=numpy.int32)
        return None

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(DunbrackEnergyTerm, self).setup_packed_block_types(packed_block_types)

        if hasattr(packed_block_types, "dunbrack_packed_block_data"):
            return

        pack = partial(
            DunbrackEnergyTerm.pack_data_keyed_on_block_type,
            self,
            packed_block_types.active_block_types,
            device=self.device,
        )
        packed_data = [
            pack(lambda f: getattr(f.dunbrack_attrs, field.name))
            for field in dataclasses.fields(DunbrackBlockAttrs)
        ]

        setattr(packed_block_types, "dunbrack_packed_block_data", packed_data)

    def pack_data_keyed_on_block_type(
        self, active_block_types, field_getter, device, default_fill=-1
    ):
        max_size = None
        dtype = None
        for bt in active_block_types:
            bt_data = field_getter(bt)
            if bt_data is None:
                continue
            cur = numpy.shape(bt_data)
            if max_size is None:
                max_size = cur
                dtype = bt_data.dtype
            max_size = numpy.maximum(max_size, cur)

        n_block_types = (len(active_block_types),)
        size = n_block_types + tuple(max_size)

        dtype_conversion = {
            numpy.dtype(numpy.int32): torch.int32,
            numpy.dtype(numpy.int64): torch.int32,
            int: torch.int32,
            torch.int32: torch.int32,
            torch.int64: torch.int32,
        }

        tensor = torch.full(
            size, default_fill, dtype=dtype_conversion[dtype], device=device
        )

        def dim_slices(dim):
            return slice(0, dim)

        for i, bt in enumerate(active_block_types):
            bt_data = field_getter(bt)
            if bt_data is None:
                continue
            slices = [i] + [*map(dim_slices, bt_data.shape)]
            tensor[slices] = torch.tensor(
                bt_data, dtype=dtype_conversion[dtype], device=device
            )

        return tensor

    def setup_poses(self, poses: PoseStack):
        super(DunbrackEnergyTerm, self).setup_poses(poses)

    def render_whole_pose_scoring_module(self, pose_stack: PoseStack):
        pbt = pose_stack.packed_block_types

        return DunbrackWholePoseScoringModule(
            pose_stack_block_coord_offset=pose_stack.block_coord_offset,
            pose_stack_block_types=pose_stack.block_type_ind,
            pose_stack_inter_block_connections=pose_stack.inter_residue_connections,
            bt_atom_downstream_of_conn=pbt.atom_downstream_of_conn,
            global_params=self.dunbrack_db,
            dunbrack_packed_block_data=pbt.dunbrack_packed_block_data,
        )
