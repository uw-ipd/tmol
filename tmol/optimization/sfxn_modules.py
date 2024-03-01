import torch

from tmol.pose.pose_stack import PoseStack
from tmol.optimization.modules import DOFMaskingFunc
from tmol.score.score_function import ScoreFunction


class CartesianSfxnNetwork(torch.nn.Module):
    def __init__(
        self, score_function: ScoreFunction, pose_stack: PoseStack, coord_mask=None
    ):
        super(CartesianSfxnNetwork, self).__init__()

        wpsm = score_function.render_whole_pose_scoring_module(pose_stack)
        self.whole_pose_scoring_module = wpsm

        self.coord_mask = coord_mask

        self.full_coords = pose_stack.coords
        if self.coord_mask is None:
            self.full_coords = torch.nn.Parameter(pose_stack.coords)
        else:
            self.masked_coords = torch.nn.Parameter(pose_stack.coords[self.coord_mask])
        self.count = 0

    def forward(self):
        self.count += 1
        if self.coord_mask is not None:
            self.full_coords = DOFMaskingFunc.apply(
                self.masked_coords, self.coord_mask, self.full_coords
            )
        return self.whole_pose_scoring_module(self.full_coords)


# class KinematicSfxnNetwork(torch.nn.Module):
#     def __init__(self, score_function, pose_stack, dof_mask=None):
#         super(KinematicSfxnNetwork, self).__init__()
#
#         self.whole_pose_scoring_module = (
#             score_function.render_whole_pose_scoring_module(pose_stack)
#         )
#         self.dof_mask = dof_mask
#         # TO DO!!!
