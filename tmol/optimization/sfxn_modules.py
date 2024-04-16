import torch

from tmol.pose.pose_stack import PoseStack
from tmol.score.score_function import ScoreFunction


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
        self.count = 0

    def forward(self):
        self.count += 1
        self.full_coords = self.full_coords.detach()
        self.full_coords[self.coord_mask] = self.masked_coords
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
