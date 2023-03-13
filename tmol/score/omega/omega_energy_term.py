import torch

from ..energy_term import EnergyTerm

from tmol.database import ParameterDatabase
from tmol.score.omega.params import OmegaGlobalParams
from tmol.score.omega.omega_whole_pose_module import OmegaWholePoseScoringModule

from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack


class OmegaEnergyTerm(EnergyTerm):
    device: torch.device  # = attr.ib()

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super(OmegaEnergyTerm, self).__init__(param_db=param_db, device=device)

        self.global_params = OmegaGlobalParams.from_database(
            param_db.scoring.omega, device
        )
        self.device = device

    @classmethod
    def score_types(cls):
        import tmol.score.terms.omega_creator

        return tmol.score.terms.omega_creator.OmegaTermCreator.score_types()

    def n_bodies(self):
        return 2

    def setup_block_type(self, block_type: RefinedResidueType):
        super(OmegaEnergyTerm, self).setup_block_type(block_type)

        if hasattr(block_type, "omega_quad_uaids"):
            return

        omega_quad_uaids = torch.full((4, 3), -1, dtype=torch.int32)
        if "omega" in block_type.torsion_to_uaids:
            omega_quad_uaids = torch.tensor(
                block_type.torsion_to_uaids["omega"], dtype=torch.int32
            )

        setattr(block_type, "omega_quad_uaids", omega_quad_uaids)

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(OmegaEnergyTerm, self).setup_packed_block_types(packed_block_types)

        if hasattr(packed_block_types, "omega_quad_uaids"):
            return

        def _t(arr):
            return torch.tensor(arr, dtype=torch.int32, device=self.device)

        omega_quad_uaids = torch.full(
            (packed_block_types.n_types, 4, 3),
            -1,
            dtype=torch.int32,
            device=self.device,
        )

        for i, bt in enumerate(packed_block_types.active_block_types):
            uaids = bt.omega_quad_uaids
            omega_quad_uaids[i, :] = uaids

        setattr(packed_block_types, "omega_quad_uaids", omega_quad_uaids)

    def setup_poses(self, poses: PoseStack):
        super(OmegaEnergyTerm, self).setup_poses(poses)

    def setup_poses(self, poses: PoseStack):
        super(OmegaEnergyTerm, self).setup_poses(poses)

    def render_whole_pose_scoring_module(self, pose_stack: PoseStack):
        pbt = pose_stack.packed_block_types

        return OmegaWholePoseScoringModule(
            pose_stack_block_coord_offset=pose_stack.block_coord_offset,
            pose_stack_block_types=pose_stack.block_type_ind,
            pose_stack_inter_block_connections=pose_stack.inter_residue_connections,
            bt_omega_quad_uaids=pbt.omega_quad_uaids,
            bt_atom_downstream_of_conn=pbt.atom_downstream_of_conn,
            global_params=self.global_params,
        )
