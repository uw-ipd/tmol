import numpy
import torch

import attr

from tmol.types.torch import Tensor
from tmol.types.array import NDArray

from tmol.score.energy_term import EnergyTerm
from .params import BackboneTorsionParamResolver
from .bb_torsion_whole_pose_module import BackboneTorsionWholePoseScoringModule
from tmol.database import ParameterDatabase

from tmol.chemical.restypes import RefinedResidueType, uaid_t
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack


@attr.s(frozen=True, slots=True, auto_attribs=True)
class BackboneTorsionBlockTypeParams:
    rama_table_inds: NDArray[numpy.int32][2]  # non-prepro/prepro
    omega_table_inds: NDArray[numpy.int32][2]  # non-prepro/prepro
    upper_conn_ind: NDArray[numpy.int32][1]
    is_pro: NDArray[numpy.int32][1]
    backbone_torsion_atoms: NDArray[uaid_t][12, 3]


@attr.s(frozen=True, slots=True, auto_attribs=True)
class BackboneTorsionPackedBlockTypesParams:
    bt_rama_table: Tensor[torch.int32][:, 2]  # non-prepro/prepro
    bt_omega_table: Tensor[torch.int32][:, 2]  # non-prepro/prepro
    bt_upper_conn_ind: Tensor[torch.int32][:]
    bt_is_pro: Tensor[torch.int32][:]
    bt_backbone_torsion_atoms: Tensor[torch.int32][:, 12, 3]


class BackboneTorsionEnergyTerm(EnergyTerm):
    device: torch.device
    param_resolver: BackboneTorsionParamResolver
    rama_tables: Tensor[torch.float32][:, :, :]
    rama_table_params: Tensor[torch.float32][:, :, :]
    omega_tables: Tensor[torch.float32][:, :, :, :]
    omega_table_params: Tensor[torch.float32][:, :, :]

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super(BackboneTorsionEnergyTerm, self).__init__(
            param_db=param_db, device=device
        )
        self.device = device

        self.param_resolver = BackboneTorsionParamResolver.from_database(
            param_db.scoring.rama, param_db.scoring.omega_bbdep, device
        )

        self.rama_tables = self.param_resolver.rama_params.tables
        self.rama_table_params = torch.cat(
            (
                self.param_resolver.rama_params.bbstarts.to(torch.float32),
                self.param_resolver.rama_params.bbsteps.to(torch.float32),
            ),
            dim=1,
        )
        self.omega_tables = self.param_resolver.rama_params.tables
        self.omega_table_params = torch.cat(
            (
                self.param_resolver.omega_params.bbstarts.to(torch.float32),
                self.param_resolver.omega_params.bbsteps.to(torch.float32),
            ),
            dim=1,
        )

    @classmethod
    def score_types(cls):
        import tmol.score.terms.rama_omega_creator

        return (
            tmol.score.terms.rama_omega_creator.BackboneTorsionTermCreator.score_types()
        )

    def n_bodies(self):
        return 2

    def setup_block_type(self, block_type: RefinedResidueType):
        def uaids_to_np(tor_uaids):
            uaids = numpy.full((4,), -1, dtype=uaid_t)
            for i in range(4):
                uaids[i] = numpy.array(tor_uaids[i], dtype=uaid_t)
            return uaids

        super(BackboneTorsionEnergyTerm, self).setup_block_type(block_type)
        if hasattr(block_type, "backbone_torsion_params"):
            return

        rname = block_type.name
        lookups = numpy.array([[rname, "_"], [rname, "PRO"]], dtype=object)
        rama_table_inds = self.param_resolver.rama_lookup.index.get_indexer(lookups)
        rama_table_inds = self.param_resolver.rama_lookup.iloc[rama_table_inds, :][
            "table_id"
        ].values
        omega_table_inds = self.param_resolver.omega_lookup.index.get_indexer(lookups)
        omega_table_inds = self.param_resolver.omega_lookup.iloc[omega_table_inds, :][
            "table_id"
        ].values

        backbone_torsion_atoms = numpy.full((3, 4), -1, dtype=uaid_t)
        if rama_table_inds[0] != -1 or rama_table_inds[1] != -1:
            for i, tor in enumerate(["phi", "psi", "omega"]):
                if tor in block_type.torsion_to_uaids:
                    backbone_torsion_atoms[i] = uaids_to_np(
                        block_type.torsion_to_uaids[tor]
                    )
        backbone_torsion_atoms = backbone_torsion_atoms.reshape(-1)

        upper_conn_id = numpy.full((1,), -1, dtype=numpy.int32)
        if "up" in block_type.connection_to_cidx:
            upper_conn_id[0] = block_type.connection_to_cidx["up"]
        is_pro = numpy.full((1,), 0, dtype=numpy.int32)

        # TO DO: Better logic here to handle proline variants
        if block_type.name == "PRO":
            is_pro[0] = 1

        bt_bbtors_params = BackboneTorsionBlockTypeParams(
            rama_table_inds=rama_table_inds,
            omega_table_inds=omega_table_inds,
            upper_conn_ind=upper_conn_id,
            is_pro=is_pro,
            backbone_torsion_atoms=backbone_torsion_atoms,
        )
        setattr(block_type, "backbone_torsion_params", bt_bbtors_params)

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(BackboneTorsionEnergyTerm, self).setup_packed_block_types(
            packed_block_types
        )
        if hasattr(packed_block_types, "backbone_torsion_params"):
            return
        n_types = packed_block_types.n_types

        bt_rama_table = numpy.full((n_types, 2), -1, dtype=numpy.int32)
        bt_omega_table = numpy.full((n_types, 2), -1, dtype=numpy.int32)
        bt_upper_conn_ind = numpy.full((n_types,), -1, dtype=numpy.int32)
        bt_is_pro = numpy.full((n_types,), -1, dtype=numpy.int32)
        bt_backbone_torsion_atoms = numpy.full((n_types, 12, 3), -1, dtype=numpy.int32)

        for i, bt in enumerate(packed_block_types.active_block_types):
            i_bbtors_params = bt.backbone_torsion_params
            bt_rama_table[i] = i_bbtors_params.rama_table_inds
            bt_omega_table[i] = i_bbtors_params.omega_table_inds
            bt_upper_conn_ind[i] = i_bbtors_params.upper_conn_ind
            bt_is_pro[i] = i_bbtors_params.is_pro
            bt_backbone_torsion_atoms[i] = i_bbtors_params.backbone_torsion_atoms.view(
                numpy.int32
            ).reshape(12, 3)

        def _t(t):
            return torch.tensor(t, device=packed_block_types.device)

        backbone_torsion_params = BackboneTorsionPackedBlockTypesParams(
            bt_rama_table=_t(bt_rama_table),
            bt_omega_table=_t(bt_omega_table),
            bt_upper_conn_ind=_t(bt_upper_conn_ind),
            bt_is_pro=_t(bt_is_pro),
            bt_backbone_torsion_atoms=_t(bt_backbone_torsion_atoms),
        )
        setattr(packed_block_types, "backbone_torsion_params", backbone_torsion_params)

    def setup_poses(self, pose_stack: PoseStack):
        super(BackboneTorsionEnergyTerm, self).setup_poses(pose_stack)

    def render_whole_pose_scoring_module(self, pose_stack: PoseStack):
        pbt = pose_stack.packed_block_types

        return BackboneTorsionWholePoseScoringModule(
            pose_stack_block_coord_offset=pose_stack.block_coord_offset,
            pose_stack_block_type=pose_stack.block_type_ind,
            pose_stack_inter_residue_connections=pose_stack.inter_residue_connections,
            bt_atom_downstream_of_conn=pbt.atom_downstream_of_conn,
            bt_rama_table=pbt.backbone_torsion_params.bt_rama_table,
            bt_omega_table=pbt.backbone_torsion_params.bt_omega_table,
            bt_upper_conn_ind=pbt.backbone_torsion_params.bt_upper_conn_ind,
            bt_is_pro=pbt.backbone_torsion_params.bt_is_pro,
            bt_backbone_torsion_atoms=pbt.backbone_torsion_params.bt_backbone_torsion_atoms,
            rama_tables=self.rama_tables,
            rama_table_params=self.rama_table_params,
            omega_tables=self.omega_tables,
            omega_table_params=self.omega_table_params,
        )
