import torch
import attr

from tmol.pose.pose_stack import PoseStack
from tmol.score.score_function import ScoreFunction
from tmol.kinematics.operations import inverseKin, forwardKin
from tmol.kinematics.datatypes import KinDOF
from tmol.kinematics.fold_forest import FoldForest
from tmol.pose.pose_kinematics import construct_pose_stack_kinforest


class CartesianSfxnNetwork(torch.nn.Module):
    def __init__(
        self, score_function: ScoreFunction, pose_stack: PoseStack, coord_mask=None
    ):
        super(CartesianSfxnNetwork, self).__init__()

        wpsm = score_function.render_whole_pose_scoring_module(pose_stack)
        self.whole_pose_scoring_module = wpsm

        self.full_coords = pose_stack.coords
        if coord_mask is None:
            coord_mask = torch.full(
                self.full_coords.shape[:-1],
                True,
                device=self.full_coords.device,
                dtype=torch.bool,
            )
        self.coord_mask = coord_mask

        self.masked_coords = torch.nn.Parameter(self.full_coords[self.coord_mask])

    def forward(self):
        # apply mask to coords
        self.full_coords = self.full_coords.detach()
        self.full_coords[self.coord_mask] = self.masked_coords

        # score pose
        return self.whole_pose_scoring_module(self.full_coords)


class KinematicSfxnNetwork(torch.nn.Module):
    def __init__(self, score_function, pose_stack, nonideal=False):
        super(KinematicSfxnNetwork, self).__init__()

        wpsm = score_function.render_whole_pose_scoring_module(pose_stack)
        self.whole_pose_scoring_module = wpsm

        # fd There are two issues to address:
        # fd  1) this call only works on CPU (since we use numba)
        # fd  2) 'pose_stack.n_res_per_pose' does not work on multichain poses

        # a) construct the kinforest and dof mask
        fold_forest = FoldForest.polymeric_forest(pose_stack.n_res_per_pose.cpu())
        self.kinforest = construct_pose_stack_kinforest(pose_stack, fold_forest).to(
            pose_stack.coords.device
        )
        self.dof_mask = self.kinforest.default_mask(nonideal=nonideal)

        # b) inverse fold to get torsions from pose
        self.pose_stack_coords = pose_stack.coords
        coords_flat = pose_stack.coords.reshape(-1, 3)
        kincoords = coords_flat[self.kinforest.id.to(torch.long)]
        dofs = inverseKin(self.kinforest, kincoords.to(torch.double))

        # c) apply mask, allocate params
        self.full_dofs = dofs.raw
        self.masked_dofs = torch.nn.Parameter(self.full_dofs[self.dof_mask])

    def forward(self):
        # apply mask to dofs
        self.full_dofs = self.full_dofs.detach()
        self.full_dofs[self.dof_mask] = self.masked_dofs

        # recompute coordinates from dofs
        self.pose_stack_coords = self.pose_stack_coords.detach()
        kincoords = forwardKin(self.kinforest, KinDOF(raw=self.full_dofs)).float()
        self.pose_stack_coords.view(-1, 3)[self.kinforest.id.to(torch.long)] = kincoords

        # score pose
        return self.whole_pose_scoring_module(self.pose_stack_coords)
