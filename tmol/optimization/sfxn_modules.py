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

        self.pose_stack = pose_stack
        # clone: forward() writes into full_coords in place, which would
        # otherwise overwrite the caller's coordinates
        self.full_coords = pose_stack.coords.clone().detach()
        if coord_mask is None:
            coord_mask = torch.full(
                self.full_coords.shape[:-1],
                True,
                device=self.full_coords.device,
                dtype=torch.bool,
            )
        self.coord_mask = coord_mask

        # Precompute flat integer indices for the boolean mask
        # Flat integer is faster than bool mask
        #   (since i think torch does nonzero each time under the hood)
        self._coord_flat_idx = (
            self.coord_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        )

        self.masked_coords = torch.nn.Parameter(self.full_coords[self.coord_mask])
        self.count = 0

    def dof_pose_assignment(self):
        """Return a 1-D int64 tensor of length n_flat_dofs giving the pose
        index for each DOF in the flat parameter vector (masked_coords.view(-1)).

        Atoms in masked_coords are ordered by (pose, atom) because coord_mask
        is reshaped row-major, so all atoms of pose 0 precede pose 1, etc.
        """
        n_poses = self.full_coords.shape[0]
        device = self.full_coords.device
        atoms_per_pose = self.coord_mask.sum(dim=1)  # [n_poses]
        pose_indices = torch.arange(n_poses, device=device, dtype=torch.int64)
        atom_pose = torch.repeat_interleave(pose_indices, atoms_per_pose)
        return atom_pose.repeat_interleave(3)  # x,y,z for each atom

    def forward(self):
        self.count += 1
        self.full_coords = self.full_coords.detach()
        # self.full_coords[self.coord_mask] = self.masked_coords
        self.full_coords.view(-1, self.full_coords.shape[-1])[
            self._coord_flat_idx
        ] = self.masked_coords
        return self.whole_pose_scoring_module(self.full_coords)

    def pose_stack_from_dofs(self):
        full_coords = self.full_coords.detach().clone()
        full_coords.view(-1, full_coords.shape[-1])[
            self._coord_flat_idx
        ] = self.masked_coords.detach()
        return attrs.evolve(self.pose_stack, coords=full_coords)


class KinForestSfxnNetwork(torch.nn.Module):
    def __init__(
        self,
        score_function: ScoreFunction,
        pose_stack: PoseStack,
        kin_module: PoseStackKinematicsModule,
        dof_mask=None,
        kin_dtype=torch.float32,
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
            dtype=kin_dtype,
            device=torch_device,
        )
        kincoords[1:] = pose_stack.coords.view(-1, 3)[kmd.forest.id[1:]].to(kin_dtype)

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

        # Precompute flat integer indices for the boolean mask
        # Flat integer is faster than bool mask
        self._dof_flat_idx = (
            self.dof_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        )

        self.masked_dofs = torch.nn.Parameter(self.full_dofs[self.dof_mask])
        self.count = 0

    def dof_pose_assignment(self):
        """Return a 1-D int64 tensor of length n_flat_dofs giving the pose
        index for each DOF in the flat parameter vector (masked_dofs.view(-1)).

        kmd.forest.id maps kinematic-atom index -> flat pose-stack atom index
        (into coords.view(-1, 3)).  Index 0 is a virtual root; id[1:] are
        the real atoms.  Pose index = flat_atom_idx // max_n_pose_atoms.
        """
        max_n_pose_atoms = self.pose_stack.max_n_pose_atoms
        kin_atom_to_pose = torch.zeros(
            self.id.shape[0], dtype=torch.int64, device=self.id.device
        )
        kin_atom_to_pose[1:] = self.id[1:].to(torch.int64) // max_n_pose_atoms
        # For each masked DOF, retrieve the pose of its kinematic atom
        masked_kin_atoms = self.dof_mask.nonzero(as_tuple=False)[:, 0]
        return kin_atom_to_pose[masked_kin_atoms]

    def forward(self):
        self.count += 1

        # get rid of any gradients from the previous iteration
        self.full_dofs = self.full_dofs.detach()
        self.full_coords = self.full_coords.detach()
        self.flat_coords = self.flat_coords.detach()

        # update the full-dofs, calc the coords, and map them
        # to the pose-stack-ordered coords
        self.full_dofs.view(-1)[self._dof_flat_idx] = self.masked_dofs
        kin_coords = self.kin_module(self.full_dofs)
        self.flat_coords[self.id[1:]] = kin_coords[1:].to(self.flat_coords.dtype)
        self.full_coords = self.flat_coords.view(self.orig_coords_shape)

        # now evaluate the score
        return self.whole_pose_scoring_module(self.full_coords)

    def pose_stack_from_dofs(self):

        full_dofs = self.full_dofs.clone()
        flat_coords = self.flat_coords.detach()
        full_dofs.view(-1)[self._dof_flat_idx] = self.masked_dofs
        kin_coords = self.kin_module(full_dofs)
        flat_coords[self.id[1:]] = kin_coords[1:].to(flat_coords.dtype)
        full_coords = flat_coords.view(self.orig_coords_shape)

        return attrs.evolve(self.pose_stack, coords=full_coords)
