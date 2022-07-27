# import attr
import torch

from tmol.pack.sim_anneal.compiled.compiled import (
    pick_random_rotamers,
    metropolis_accept_reject,
    # create_sim_annealer,
    # delete_sim_annealer,
    # register_standard_random_rotamer_picker,
    # reigester_standard_metropolis_accept_or_rejector,
    # run_sim_annealing,
)


# class SelectRanRotModule(torch.jit.ScriptModule):
class SelectRanRotModule:
    def __init__(
        self,
        n_traj_per_pose,
        pose_id_for_context,
        n_rots_for_pose,
        rot_offset_for_pose,
        block_type_ind_for_rot,
        block_ind_for_rot,
        rotamer_coords,
        rotamer_coord_offsets,
        alternate_coords,
        alternate_coord_offsets,
        alternate_block_id,
        random_rots,
        block_type_n_atoms,
    ):
        super().__init__()

        dev = rotamer_coords.device

        assert n_rots_for_pose.device == dev
        assert rot_offset_for_pose.device == dev
        assert block_type_ind_for_rot.device == dev
        assert block_ind_for_rot.device == dev
        assert rotamer_coords.device == dev
        assert rotamer_coord_offsets.device == dev
        assert alternate_coords.device == dev
        assert alternate_coord_offsets.device == dev
        assert alternate_block_id.device == dev
        assert random_rots.device == dev
        assert block_type_n_atoms.device == dev

        def _p(t):
            return torch.nn.Parameter(t, requires_grad=False)

        n_traj_per_pose = torch.full(
            (1,), n_traj_per_pose, dtype=torch.int64, device=dev
        )

        self.pose_id_for_context = _p(pose_id_for_context.to(torch.int32))
        self.n_rots_for_pose = _p(n_rots_for_pose.to(torch.int32))
        self.rot_offset_for_pose = _p(rot_offset_for_pose.to(torch.int32))
        self.block_type_ind_for_rot = _p(block_type_ind_for_rot.to(torch.int32))
        self.block_ind_for_rot = _p(block_ind_for_rot.to(torch.int32))
        self.rotamer_coords = _p(rotamer_coords)
        self.rotamer_coord_offsets = _p(rotamer_coord_offsets)
        self.alternate_coords = _p(alternate_coords)
        self.alternate_coord_offsets = _p(alternate_coord_offsets)
        self.alternate_block_id = _p(alternate_block_id)
        self.random_rots = _p(random_rots)
        self.block_type_n_atoms = _p(block_type_n_atoms)
        self.max_n_atoms = int(torch.max(block_type_n_atoms).cpu().item())

    # @torch.jit.script_method
    # def forward(self, context_coords, context_block_type):

    def go(self, context_coords, context_coord_offsets, context_block_type):
        """Select one rotamer for each context as well as the current rotamer at that position"""

        return pick_random_rotamers(
            context_coords,
            context_coord_offsets,
            context_block_type,
            self.pose_id_for_context,
            self.n_rots_for_pose,
            self.rot_offset_for_pose,
            self.block_type_ind_for_rot,
            self.block_ind_for_rot,
            self.rotamer_coords,
            self.rotamer_coord_offsets,
            self.alternate_coords,
            self.alternate_coord_offsets,
            self.alternate_block_id,
            self.random_rots,
            self.block_type_n_atoms,
            self.max_n_atoms,
        )

        # context_block_type = context_block_type.to(torch.int64)
        # n_contexts = self.pose_id_for_context.shape[0]
        # max_n_atoms = self.rotamer_coords.shape[1]
        # dev = self.pose_id_for_context.device
        #
        # alternate_coords = torch.zeros(
        #     (n_contexts * 2, max_n_atoms, 3), dtype=torch.float32, device=dev
        # )
        # alternate_ids = torch.zeros((n_contexts * 2, 3), dtype=torch.int32, device=dev)
        #
        # urand = torch.rand(n_contexts, dtype=torch.float32, device=dev)
        #
        # rand_rot_local = torch.floor(
        #     urand * self.n_rots_for_pose[self.pose_id_for_context].to(torch.float32)
        # ).to(torch.int64)
        # # 1 context per pose
        # rand_rot_global = self.rot_offset_for_pose + rand_rot_local
        #
        # rand_rot_block_ind = self.block_ind_for_rot[rand_rot_global]
        # n_contexts = self.pose_id_for_context.shape[0]
        # context_arange = torch.arange(n_contexts, dtype=torch.int64, device=dev)
        # alternate_coords[2 * context_arange] = context_coords[
        #     self.pose_id_for_context, rand_rot_block_ind
        # ]
        # alternate_ids[2 * context_arange, 0] = context_arange.to(torch.int32)
        # alternate_ids[2 * context_arange, 1] = rand_rot_block_ind.to(torch.int32)
        # alternate_ids[2 * context_arange, 2] = context_block_type[
        #     context_arange, rand_rot_block_ind
        # ].to(torch.int32)
        #
        # alternate_coords[2 * context_arange + 1,] = self.rotamer_coords[rand_rot_global]
        #
        # alternate_ids[2 * context_arange + 1, 0] = context_arange.to(torch.int32)
        # alternate_ids[2 * context_arange + 1, 1] = rand_rot_block_ind.to(torch.int32)
        # alternate_ids[2 * context_arange + 1, 2] = self.block_type_ind_for_rot[
        #     rand_rot_global
        # ].to(torch.int32)
        #
        # return alternate_coords, alternate_ids, rand_rot_global


# class MCAcceptRejectModule(torch.jit.ScriptModule):
class MCAcceptRejectModule:
    def __init__(
        self,
        block_type_n_atoms,
    ):
        super().__init__()

        def _p(t):
            return torch.nn.Parameter(t, requires_grad=False)

        self.block_type_n_atoms = _p(block_type_n_atoms)
        self.max_n_atoms = int(torch.max(block_type_n_atoms).cpu().item())

    # @torch.jit.script_method
    # def forward(

    def go(
        self,
        temperature,
        context_coords,
        context_coord_offsets,
        context_block_type,
        alternate_coords,
        alternate_coord_offsets,
        alternate_ids,
        rotamer_component_energies,  # pre-weighted
        accepted,
    ):
        return metropolis_accept_reject(
            temperature,
            context_coords,
            context_coord_offsets,
            context_block_type,
            alternate_coords,
            alternate_coord_offsets,
            alternate_ids,
            rotamer_component_energies,
            accepted,
            self.block_type_n_atoms,
            self.max_n_atoms,
        )

        # rotamer_energies = torch.sum(rotamer_component_energies, dim=0)
        # n_contexts = context_coords.shape[0]
        # dev = context_coords.device
        # context_arange = torch.arange(n_contexts, dtype=torch.int64, device=dev)
        # deltaE = (
        #     rotamer_energies[2 * context_arange + 1]
        #     - rotamer_energies[2 * context_arange]
        # )
        # uran = torch.rand(n_contexts, dtype=torch.float32, device=dev)
        # prob_accept = torch.exp(-1 * deltaE / temperature)
        # accept_contexts = torch.nonzero((uran < prob_accept) | (deltaE < 0)).flatten()
        # accept_block_ind = alternate_ids[2 * accept_contexts, 1].to(torch.int64)
        #
        # context_coords[accept_contexts, accept_block_ind] = alternate_coords[
        #     2 * accept_contexts + 1
        # ]
        # context_block_type[accept_contexts, accept_block_ind] = alternate_ids[
        #     2 * accept_contexts + 1, 2
        # ]
        #
        # return accept_contexts
