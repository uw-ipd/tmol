import torch

from ..atom_type_dependent_term import AtomTypeDependentTerm
from ..bond_dependent_term import BondDependentTerm

from tmol.database import ParameterDatabase
from tmol.score.common.stack_condense import tile_subset_indices
from tmol.score.omega.params import OmegaParamResolver
from tmol.score.omega.omega_whole_pose_module import OmegaWholePoseScoringModule

from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack
from tmol.types.torch import Tensor

# from tmol.score.omega.potentials.compiled import (
#    score_ljlk_inter_system_scores,
# )


class OmegaEnergyTerm(AtomTypeDependentTerm, BondDependentTerm):
    tile_size: int = 32

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        omega_param_resolver = OmegaParamResolver.from_database(
            param_db.scoring.omega, device=device
        )
        super(OmegaEnergyTerm, self).__init__(param_db=param_db, device=device)
        self.global_params = omega_param_resolver.global_params
        self.tile_size = OmegaEnergyTerm.tile_size

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

        # or i in range(pbt.omega_quad_uaids.size(dim=0)):
        # print(pbt.omega_quad_uaids[i][2][2].item())

        print("DEVICES")
        print(pbt.omega_quad_uaids.device)

        return OmegaWholePoseScoringModule(
            pose_stack_block_coord_offset=pose_stack.block_coord_offset,
            pose_stack_block_types=pose_stack.block_type_ind,
            pose_stack_inter_block_connections=pose_stack.inter_residue_connections,
            bt_omega_quad_uaids=pbt.omega_quad_uaids,
            global_params=self.global_params,
        )
