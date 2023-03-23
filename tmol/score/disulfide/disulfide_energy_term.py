import torch
import numpy

from ..energy_term import EnergyTerm

from tmol.database import ParameterDatabase
from tmol.score.disulfide.params import DisulfideGlobalParams
from tmol.score.disulfide.disulfide_whole_pose_module import (
    DisulfideWholePoseScoringModule
)

from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack


class DisulfideEnergyTerm(EnergyTerm):
    device: torch.device  # = attr.ib()

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super(DisulfideEnergyTerm, self).__init__(param_db=param_db, device=device)

        self.global_params = DisulfideGlobalParams.from_database(
            param_db.scoring.disulfide, device
        )
        self.device = device

    @classmethod
    def score_types(cls):
        import tmol.score.terms.disulfide_creator

        return tmol.score.terms.disulfide_creator.DisulfideTermCreator.score_types()

    def n_bodies(self):
        return 1

    def setup_block_type(self, block_type: RefinedResidueType):
        super(DisulfideEnergyTerm, self).setup_block_type(block_type)

        if hasattr(block_type, "disulfide_connections"):
            return

        disulfide_connections = numpy.array([], dtype=numpy.int32)
        if "dslf" in block_type.connection_to_cidx.keys():
            disulfide_connections = numpy.append(
                disulfide_connections, [block_type.connection_to_cidx["dslf"]]
            )

        setattr(block_type, "disulfide_connections", disulfide_connections)

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(DisulfideEnergyTerm, self).setup_packed_block_types(packed_block_types)

        if hasattr(packed_block_types, "disulfide_conns"):
            return

        disulfide_conns = torch.full(
            (packed_block_types.n_types, packed_block_types.max_n_conn),
            0,
            dtype=torch.int32,
            device=self.device,
        )

        for i, bt in enumerate(packed_block_types.active_block_types):
            for conn in bt.disulfide_connections:
                disulfide_conns[i, conn] = 1

        setattr(packed_block_types, "disulfide_conns", disulfide_conns)

    def setup_poses(self, poses: PoseStack):
        super(DisulfideEnergyTerm, self).setup_poses(poses)

    def render_whole_pose_scoring_module(self, pose_stack: PoseStack):
        pbt = pose_stack.packed_block_types

        return DisulfideWholePoseScoringModule(
            pose_stack_block_coord_offset=pose_stack.block_coord_offset,
            pose_stack_block_types=pose_stack.block_type_ind,
            pose_stack_inter_block_connections=pose_stack.inter_residue_connections,
            bt_disulfide_conns=pbt.disulfide_conns,
            bt_atom_downstream_of_conn=pbt.atom_downstream_of_conn,
            global_params=self.global_params,
        )
