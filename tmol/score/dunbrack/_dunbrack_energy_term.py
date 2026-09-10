import attr

import torch
import numpy

from .._energy_term import EnergyTerm

from tmol.database import ParameterDatabase
from tmol.score.dunbrack import (
    DunbrackParamResolver,
    ScoringDunbrackDatabaseView,
)
from tmol.chemical import RefinedResidueType
from tmol.pose import (
    PackedBlockTypes,
    PoseStack,
)
from itertools import count
from functools import partial
from dataclasses import dataclass
import dataclasses


@dataclass
class DunbrackBlockAttrs:
    n_dihedrals: int
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


def _empty_dunbrack_attrs() -> "DunbrackBlockAttrs":
    return DunbrackBlockAttrs(
        n_dihedrals=0,
        dih_uaids=numpy.full((0, 4, 3), -1, dtype=numpy.int32),
        rotamer_table_set=-1,
        rotameric_index=-1,
        semirotameric_index=-1,
        n_chi=0,
        n_rotameric_chi=0,
        probability_table_offset=-1,
        mean_table_offset=-1,
        rotamer_index_to_table_index_offset=-1,
        semirotameric_tableset_offset=numpy.array(-1),
    )


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
        self._table_indices = tuple(
            table["dun_table_name"].to_dict()
            for table in (
                self.global_params.all_table_indices,
                self.global_params.rotameric_table_indices,
                self.global_params.semirotameric_table_indices,
            )
        )
        aux = self.global_params.scoring_db_aux
        self._host_aux = {
            field.name: getattr(aux, field.name).cpu().numpy()
            for field in attr.fields(type(aux))
        }
        self.device = device

    @classmethod
    def class_name(cls):
        return "Dunbrack"

    @classmethod
    def score_types(cls):
        import tmol.score.terms._dunbrack_creator

        return tmol.score.terms._dunbrack_creator.DunbrackTermCreator.score_types()

    def n_bodies(self):
        return 1

    def setup_block_type(self, block_type: RefinedResidueType):
        super(DunbrackEnergyTerm, self).setup_block_type(block_type)

        if hasattr(block_type, "dunbrack_attrs"):
            return

        # Dunbrack rotamer libraries cover alpha amino acids only. For ligands
        # and other non-AA block types, install a sentinel with
        # rotamer_table_set=-1 so the C++ kernel short-circuits the block.
        polymer = block_type.properties.polymer
        if (
            not polymer.is_polymer
            or polymer.polymer_type != "amino_acid"
            or polymer.backbone_type != "alpha"
        ):
            setattr(block_type, "dunbrack_attrs", _empty_dunbrack_attrs())
            return

        rotamer_table_set, rotameric_index, semirotameric_index = (
            table.get(block_type.base_name, -1) for table in self._table_indices
        )
        semirotameric = semirotameric_index != -1

        semirotameric_tableset_offset = (
            numpy.array(-1)
            if not semirotameric
            else numpy.array(
                self._host_aux["semirotameric_tableset_offsets"][semirotameric_index]
            )
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

        dih_uaids = numpy.array([phi_uaids] + [psi_uaids] + chis)

        n_chi = int(self._host_aux["nchi_for_table_set"][rotamer_table_set])
        n_rotameric_chi = n_chi - (1 if semirotameric else 0)
        n_dihedrals = n_chi + 2

        probability_table_offset = int(
            self._host_aux["rotameric_prob_tableset_offsets"][rotameric_index]
        )

        mean_table_offset = int(
            self._host_aux["rotameric_meansdev_tableset_offsets"][rotamer_table_set]
        )
        rotamer_index_to_table_index_offset = int(
            self._host_aux["rotameric_chi_ri2ti_offsets"][rotamer_table_set]
        )

        dunbrack_attrs = DunbrackBlockAttrs(
            n_dihedrals=n_dihedrals,
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
        for bt in active_block_types:
            bt_data = field_getter(bt)
            if bt_data is None:
                continue
            cur = numpy.shape(bt_data)
            if max_size is None:
                max_size = cur
            max_size = numpy.maximum(max_size, cur)

        n_block_types = (len(active_block_types),)
        size = n_block_types + tuple(max_size)

        packed = numpy.full(size, default_fill, dtype=numpy.int32)
        for i, bt in enumerate(active_block_types):
            bt_data = field_getter(bt)
            if bt_data is None:
                continue
            if isinstance(bt_data, (int, numpy.integer)) or (
                isinstance(bt_data, numpy.ndarray) and bt_data.ndim == 0
            ):
                packed[i] = int(bt_data)
            else:
                slices = (i,) + tuple(slice(0, dim) for dim in numpy.shape(bt_data))
                packed[slices] = bt_data

        return torch.as_tensor(packed, dtype=torch.int32, device=device)

    def setup_poses(self, poses: PoseStack):
        super(DunbrackEnergyTerm, self).setup_poses(poses)

    def get_pose_score_term_function(self):
        from tmol.score.dunbrack.potentials import dunbrack_pose_scores

        return dunbrack_pose_scores

    def get_rotamer_score_term_function(self):
        from tmol.score.dunbrack.potentials import dunbrack_rotamer_scores

        return dunbrack_rotamer_scores

    def get_score_term_attributes(self, pose_stack):
        pbt = pose_stack.packed_block_types

        return [
            pose_stack.inter_residue_connections,
            pbt.atom_downstream_of_conn,
            *self.dunbrack_db,
            *pbt.dunbrack_packed_block_data,
        ]
