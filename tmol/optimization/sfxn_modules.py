import torch
import attrs

from tmol.pose.pose_stack import PoseStack
from tmol.score.score_function import ScoreFunction
from tmol.kinematics.datatypes import NodeType, BondDOFTypes, JumpDOFTypes
from tmol.kinematics.script_modules import PoseStackKinematicsModule

from tmol.kinematics.compiled import inverse_kin


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


class KinForestSfxnNetwork(torch.nn.Module):
    def __init__(
        self,
        score_function: ScoreFunction,
        pose_stack: PoseStack,
        kin_module: PoseStackKinematicsModule,
        dof_mask=None,
    ):

        super(KinForestSfxnNetwork, self).__init__()

        torch_device = pose_stack.device
        self.pose_stack = pose_stack
        wpsm = score_function.render_whole_pose_scoring_module(pose_stack)
        kmd = kin_module.kmd
        self.kin_module = kin_module
        self.whole_pose_scoring_module = wpsm
        self.full_coords = pose_stack.coords.clone().detach()
        self.flat_coords = self.full_coords.view(-1, 3)
        self.orig_coords_shape = pose_stack.coords.shape
        self.id = kmd.forest.id

        kincoords = torch.zeros(
            (kin_module.kmd.forest.id.shape[0], 3),
            dtype=torch.float32,
            device=torch_device,
        )
        kincoords[1:] = pose_stack.coords.view(-1, 3)[kmd.forest.id[1:]]

        raw_dofs = inverse_kin(
            kincoords,
            kmd.forest.parent,
            kmd.forest.frame_x,
            kmd.forest.frame_y,
            kmd.forest.frame_z,
            kmd.forest.doftype,
        )
        self.full_dofs = raw_dofs

        if dof_mask is None:
            # Default behavior:
            #   Enable minimization of phi_c dofs for bonded atoms
            #   Enable minimization of 6 dofs for jump atoms
            #   - RBx, y, z, and
            #   - RBdel_alpha, beta, gamma
            dof_mask = torch.zeros(
                raw_dofs.shape, dtype=torch.bool, device=torch_device
            )
            dof_mask[kmd.forest.doftype == NodeType.bond, BondDOFTypes.phi_c] = True
            dof_mask[
                kmd.forest.doftype == NodeType.jump, : JumpDOFTypes.RBdel_gamma
            ] = True
        self.dof_mask = dof_mask

        self.masked_dofs = torch.nn.Parameter(self.full_dofs[self.dof_mask])
        self.count = 0

    def forward(self):
        self.count += 1

        # get rid of any gradients from the previous iteration
        self.full_dofs = self.full_dofs.detach()
        self.full_coords = self.full_coords.detach()
        self.flat_coords = self.flat_coords.detach()

        # update the full-dofs, calc the coords, and map them
        # to the pose-stack-ordered coords
        self.full_dofs[self.dof_mask] = self.masked_dofs
        kin_coords = self.kin_module(self.full_dofs)
        self.flat_coords[self.id[1:]] = kin_coords[1:]
        self.full_coords = self.flat_coords.view(self.orig_coords_shape)

        # now evaluate the score
        return self.whole_pose_scoring_module(self.full_coords)

    def pose_stack_from_dofs(self):

        full_dofs = self.full_dofs.clone()
        flat_coords = self.flat_coords.detach()
        full_dofs[self.dof_mask] = self.masked_dofs
        kin_coords = self.kin_module(full_dofs)
        flat_coords[self.id[1:]] = kin_coords[1:]
        full_coords = flat_coords.view(self.orig_coords_shape)

        return attrs.evolve(self.pose_stack, coords=full_coords)
