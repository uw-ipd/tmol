import attr

import torch
import numpy

from .._energy_term import EnergyTerm
from .._annotation_cache import AnnotationKey, cached_annotation, store_annotation

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
    # a d-amino acid reads its l form's library at negated torsions; only the
    #    default used for a terminus needs the sign applied here, since a
    #    torsion measured from coordinates is already mirrored
    is_mirrored: int


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
        is_mirrored=0,
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
        self.device = device
        self._annotation_key = AnnotationKey.from_sources(self.global_params)
        # Small lookup metadata are host-side; do not read device scalars per RT.
        self._table_indices = []
        for name in (
            "all_table_indices",
            "rotameric_table_indices",
            "semirotameric_table_indices",
        ):
            table = getattr(self.global_params, name)
            if not table.index.is_unique:
                raise ValueError("Dunbrack residue lookup names must be unique")
            self._table_indices.append(dict(table["dun_table_name"].items()))
        self._aux = {
            field.name: getattr(self.global_params.scoring_db_aux, field.name)
            .cpu()
            .tolist()
            for field in attr.fields(type(self.global_params.scoring_db_aux))
        }

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

        cached = cached_annotation(
            block_type, "_dunbrack_annotation", self._annotation_key
        )
        if cached is None:
            cached = store_annotation(
                block_type,
                "_dunbrack_annotation",
                self._annotation_key,
                self._create_block_attrs(block_type),
            )
        block_type.dunbrack_attrs = cached
        return cached

    def _create_block_attrs(self, block_type):
        # Dunbrack rotamer libraries cover alpha amino acids only. For ligands
        # and other non-AA block types, install a sentinel with
        # rotamer_table_set=-1 so the C++ kernel short-circuits the block.
        polymer = block_type.properties.polymer
        if (
            not polymer.is_polymer
            or polymer.polymer_type != "amino_acid"
            or polymer.backbone_type != "alpha_aa"
        ):
            return _empty_dunbrack_attrs()

        rotamer_table_set, rotameric_index, semirotameric_index = (
            table.get(block_type.base_name, -1) for table in self._table_indices
        )
        semirotameric = semirotameric_index != -1

        # An amino acid the library does not cover scores no dunbrack term;
        # a borrowed dunbrack_reference is for sampling only.
        if rotamer_table_set < 0:
            return _empty_dunbrack_attrs()

        semirotameric_tableset_offset = (
            self._aux["semirotameric_tableset_offsets"][semirotameric_index]
            if semirotameric
            else -1
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

        n_chi = self._aux["nchi_for_table_set"][rotamer_table_set]
        n_rotameric_chi = n_chi - (1 if semirotameric else 0)
        n_dihedrals = n_chi + 2
        probability_table_offset = self._aux["rotameric_prob_tableset_offsets"][
            rotameric_index
        ]
        mean_table_offset = self._aux["rotameric_meansdev_tableset_offsets"][
            rotamer_table_set
        ]
        rotamer_index_to_table_index_offset = self._aux["rotameric_chi_ri2ti_offsets"][
            rotamer_table_set
        ]

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
            is_mirrored=int(polymer.sidechain_chirality == "d"),
        )

        return dunbrack_attrs

    def get_torsion(self, name, block_type):
        if name in block_type.torsion_to_uaids:
            return numpy.array(block_type.torsion_to_uaids[name], dtype=numpy.int32)
        return None

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(DunbrackEnergyTerm, self).setup_packed_block_types(packed_block_types)

        cached = cached_annotation(
            packed_block_types, "_dunbrack_packed_annotation", self._annotation_key
        )
        if cached is None:
            for block_type in packed_block_types.active_block_types:
                self.setup_block_type(block_type)
            cached = [
                self.pack_data_keyed_on_block_type(
                    packed_block_types.active_block_types,
                    lambda bt: getattr(bt.dunbrack_attrs, field.name),
                    self.device,
                )
                for field in dataclasses.fields(DunbrackBlockAttrs)
            ]
            store_annotation(
                packed_block_types,
                "_dunbrack_packed_annotation",
                self._annotation_key,
                cached,
            )
        packed_block_types.dunbrack_packed_block_data = cached
        return cached

    def pack_data_keyed_on_block_type(
        self, active_block_types, field_getter, device, default_fill=-1
    ):
        values = [field_getter(bt) for bt in active_block_types]
        arrays = [
            None if value is None else numpy.asarray(value, dtype=numpy.int32)
            for value in values
        ]
        shapes = [array.shape for array in arrays if array is not None]
        maximum = tuple(max(dim) for dim in zip(*shapes))
        packed = numpy.full((len(arrays), *maximum), default_fill, dtype=numpy.int32)
        for index, array in enumerate(arrays):
            if array is not None:
                packed[(index, *(slice(0, size) for size in array.shape))] = array
        return torch.tensor(packed, dtype=torch.int32, device=device)

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
        packed_data = self.setup_packed_block_types(pbt)

        return [
            pose_stack.inter_residue_connections,
            pbt.atom_downstream_of_conn,
            *self.dunbrack_db,
            *packed_data,
        ]
