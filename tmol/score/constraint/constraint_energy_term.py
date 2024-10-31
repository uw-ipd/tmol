import torch
import math

from ..energy_term import EnergyTerm

from tmol.database import ParameterDatabase
from tmol.score.constraint.constraint_whole_pose_module import (
    ConstraintWholePoseScoringModule,
)
from tmol.score.constraint.potentials.compiled import get_torsion_angle

from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack


class ConstraintEnergyTerm(EnergyTerm):
    device: torch.device  # = attr.ib()

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super(ConstraintEnergyTerm, self).__init__(param_db=param_db, device=device)

        self.device = device

    @classmethod
    def score_types(cls):
        import tmol.score.terms.constraint_creator

        return tmol.score.terms.constraint_creator.ConstraintTermCreator.score_types()

    def n_bodies(self):
        return 1

    @classmethod
    def get_torsion_angle_test(cls, tensor):
        return get_torsion_angle(tensor)

    @classmethod
    def harmonic(cls, atoms, params):
        atoms1 = atoms[:, 0]
        atoms2 = atoms[:, 1]
        dist = torch.linalg.norm(atoms1 - atoms2, dim=-1)
        return (dist - params[:, 0]) ** 2

    @classmethod
    def bounded(cls, atoms, params):
        lb = params[:, 0]
        ub = params[:, 1]
        sd = params[:, 2]
        rswitch = params[:, 3]

        atoms1 = atoms[:, 0]
        atoms2 = atoms[:, 1]
        dist = torch.linalg.norm(atoms1 - atoms2, dim=-1)

        ret = torch.full_like(dist, 0)

        ub2 = ub + 0.5 * sd

        g0 = dist < lb
        # g1 = torch.logical_and(lb <= dist, dist <= ub) # default 0
        g2 = torch.logical_and(ub < dist, dist <= ub2)
        g3 = dist > ub2

        ret[g0] = ((lb[g0] - dist[g0]) / sd[g0]) ** 2
        # ret[g1] = 0
        ret[g2] = ((dist[g2] - ub[g2]) / sd[g2]) ** 2
        ret[g3] = 2 * rswitch[g3] * ((dist[g3] - ub[g3]) / sd[g3]) - rswitch[g3] ** 2

        return ret

    @classmethod
    def circularharmonic(cls, atoms, params):
        x0 = params[:, 0]  # The desired angle
        sd = params[:, 1]
        offset = params[:, 2]

        angles = get_torsion_angle(atoms)

        def round_away_from_zero(val):
            return torch.trunc(val + torch.sign(val) * 0.5)

        def nearest_angle_radians(angle, x0):
            return angle - (
                round_away_from_zero((angle - x0) / (math.pi * 2)) * (math.pi / 2)
            )

        z = (nearest_angle_radians(angles, x0) - x0) / sd

        return z * z + offset

    def setup_block_type(self, block_type: RefinedResidueType):
        super(ConstraintEnergyTerm, self).setup_block_type(block_type)

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(ConstraintEnergyTerm, self).setup_packed_block_types(packed_block_types)

    def setup_poses(self, poses: PoseStack):
        super(ConstraintEnergyTerm, self).setup_poses(poses)

    def render_whole_pose_scoring_module(self, pose_stack: PoseStack):
        pbt = pose_stack.packed_block_types

        return ConstraintWholePoseScoringModule(
            pose_stack_block_coord_offset=pose_stack.block_coord_offset,
            pose_stack_block_types=pose_stack.block_type_ind,
            pose_stack_inter_block_connections=pose_stack.inter_residue_connections,
            bt_atom_downstream_of_conn=pbt.atom_downstream_of_conn,
            constraint_set=pose_stack.constraint_set,
        )
