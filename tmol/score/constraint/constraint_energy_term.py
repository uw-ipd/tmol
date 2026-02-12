import torch
import math
import os
import sys

# import torch.nn.functional

from ..energy_term import EnergyTerm

from tmol.database import ParameterDatabase
from tmol.score.constraint.potentials.compiled import get_torsion_angle

from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class ConstraintEnergyTerm(EnergyTerm):
    device: torch.device  # = attr.ib()

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super(ConstraintEnergyTerm, self).__init__(param_db=param_db, device=device)

        self.device = device

    @classmethod
    def class_name(cls):
        return "Constraint"

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
    def harmonic_coordinate(cls, atoms, params):
        atoms1 = atoms[:, 0]
        atoms2 = params[:, 1:4]
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

    def get_pose_score_term_function(self):
        return self.constraint_pose_scores

    def get_rotamer_score_term_function(self):
        return self.constraint_pose_scores

    def get_score_term_attributes(self, pose_stack: PoseStack):
        return [
            pose_stack.get_constraint_set(),
        ]

    def constraint_pose_scores(
        self,
        coords,
        rot_coord_offset,  # rot coord offset
        pose_ind_for_atom,  # pose_ind_for_atom?? unused
        rot_offset_for_block,  # first rot for block
        first_rot_block_type,  # first rot block type
        block_ind_for_rot,
        pose_for_rot,
        block_type_ind_for_rot,
        n_rots_for_pose,
        rot_offset_for_pose,
        n_rots_for_block,
        rot_offset_for_block______2,  # three times?!
        max_n_rots_per_pose,
        constraint_set,
        output_block_pair_energies=False,
    ):
        device = coords.device

        # Early exit if we have no constraints
        if constraint_set is None:
            max_n_rots_per_block = n_rots_for_block.max().item()
            # print("max_n_rots_per_block", max_n_rots_per_block)
            if max_n_rots_per_block > 1:
                return (
                    torch.zeros((1, 0), dtype=coords.dtype, device=device),
                    torch.zeros((3, 0), dtype=torch.int32, device=device),
                )
            else:
                n_poses = first_rot_block_type.shape[0]
                max_n_blocks = first_rot_block_type.shape[1]
                if output_block_pair_energies:
                    return (
                        torch.zeros(
                            (1, n_poses, max_n_blocks, max_n_blocks),
                            dtype=torch.float32,
                            device=device,
                        ),
                        torch.zeros((3, 0), dtype=torch.int32, device=device),
                    )
                else:
                    return (
                        torch.zeros(
                            (
                                1,
                                n_poses,
                            ),
                            dtype=torch.float32,
                            device=device,
                        ),
                        torch.zeros((3, 0), dtype=torch.int32, device=device),
                    )

        unique_blocks = constraint_set.constraint_num_unique_blocks
        constraint_atoms = constraint_set.constraint_atoms
        constraint_fn_inds = constraint_set.constraint_function_inds
        constraint_params = constraint_set.constraint_params

        constraint_poses = constraint_atoms[:, :, 0]
        constraint_blocks = constraint_atoms[:, :, 1]
        constraint_ats = constraint_atoms[:, :, 2]

        rotamerized_atoms = torch.zeros((0, 4), dtype=torch.int32, device=device)
        rotamerized_fn_inds = torch.zeros((0,), dtype=torch.int32, device=device)
        rotamerized_params = torch.zeros(
            (
                0,
                constraint_params.size(1),
            ),
            device=device,
        )

        rotamerized_inds = torch.zeros((0, 3), dtype=torch.int32, device=device)

        def add_onebody_vectorized():
            # first, compute how many copies of each constraint we get
            onebody = (unique_blocks == 1).nonzero().squeeze(-1)
            if onebody.count_nonzero() == 0:
                return

            constraint_poses = constraint_atoms[onebody][:, 0, 0]
            constraint_blocks = constraint_atoms[onebody][:, :, 1]
            constraint_ats = constraint_atoms[onebody][:, :, 2]

            # get the index of the first and second blocks involved in each constraint
            block_inds = constraint_set.constraint_unique_blocks[onebody][:, 1]

            # now compute how many copies of each constraint we'll need
            num_copies_per_constraint = n_rots_for_block[constraint_poses, block_inds]
            num_copies_prev_constraint = num_copies_per_constraint.roll(shifts=1)
            num_copies_prev_constraint[0] = 0

            # use repeat_interleave to create new tensors for atoms, fn_inds, and params
            new_fn_inds = constraint_fn_inds[onebody].repeat_interleave(
                num_copies_per_constraint
            )
            new_params = constraint_params[onebody].repeat_interleave(
                num_copies_per_constraint, dim=0
            )
            # fn_inds and params are good to go, but the atoms need to be updated to point to the right rotamer atoms
            new_atoms = torch.zeros(
                (
                    new_params.size(0),
                    4,
                ),
                dtype=torch.int32,
                device=device,
            )
            # compute the rotamer offsets for each of the two bodies for all copies of the constraint
            block_subrot = torch.ones_like(new_fn_inds, dtype=torch.int32)
            start_inds = num_copies_prev_constraint.cumsum(0)
            block_subrot[start_inds] = -(num_copies_prev_constraint - 1)
            block_subrot[0] = 0
            block_subrot = block_subrot.cumsum(0)

            # first, get the rotamer offset for the first and second unique blocks
            block_rotamer_offset = rot_offset_for_block[
                constraint_poses, block_inds
            ].repeat_interleave(num_copies_per_constraint)

            # compute the first and second rotamers involved
            rot_pose = constraint_poses.repeat_interleave(num_copies_per_constraint)
            rot = block_rotamer_offset + block_subrot

            # get the matching blocks within each constraint
            constraint_blocks_match_first = (
                constraint_blocks == constraint_blocks[:, 0].unsqueeze(-1)
            ).repeat_interleave(num_copies_per_constraint, dim=0)

            atoms_rep = constraint_ats.repeat_interleave(
                num_copies_per_constraint, dim=0
            )

            first_counts_per_row = constraint_blocks_match_first.sum(dim=1)

            new_atoms[constraint_blocks_match_first] = rot_coord_offset[
                rot
            ].repeat_interleave(first_counts_per_row)

            new_atoms += atoms_rep

            # combine them
            nonlocal rotamerized_atoms
            rotamerized_atoms = torch.cat((rotamerized_atoms, new_atoms))

            nonlocal rotamerized_fn_inds
            rotamerized_fn_inds = torch.cat((rotamerized_fn_inds, new_fn_inds))

            nonlocal rotamerized_params
            rotamerized_params = torch.cat((rotamerized_params, new_params))

            nonlocal rotamerized_inds
            rotamerized_indices = torch.stack((rot_pose, rot, rot), dim=-1)
            rotamerized_inds = torch.cat((rotamerized_inds, rotamerized_indices))

        def add_twobody_vectorized():
            # first, compute how many copies of each constraint we get
            twobody = (unique_blocks == 2).nonzero().squeeze(-1)
            if twobody.count_nonzero() == 0:
                return

            constraint_poses = constraint_atoms[twobody][:, 0, 0]
            constraint_blocks = constraint_atoms[twobody][:, :, 1]
            constraint_ats = constraint_atoms[twobody][:, :, 2]

            # get the index of the first and second blocks involved in each constraint
            first_block_inds = constraint_set.constraint_unique_blocks[twobody][:, 1]
            second_block_inds = constraint_set.constraint_unique_blocks[twobody][:, 2]

            n_rots_for_first = n_rots_for_block[constraint_poses, first_block_inds]
            n_rots_for_second = n_rots_for_block[constraint_poses, second_block_inds]

            # now compute how many copies of each constraint we'll need
            num_copies_per_constraint = n_rots_for_first * n_rots_for_second
            num_copies_prev_constraint = num_copies_per_constraint.roll(shifts=1)
            num_copies_prev_constraint[0] = 0

            # use repeat_interleave to create new tensors for atoms, fn_inds, and params
            new_fn_inds = constraint_fn_inds[twobody].repeat_interleave(
                num_copies_per_constraint
            )
            new_params = constraint_params[twobody].repeat_interleave(
                num_copies_per_constraint, dim=0
            )
            # fn_inds and params are good to go, but the atoms need to be updated to point to the right rotamer atoms
            new_atoms = torch.zeros(
                (
                    new_params.size(0),
                    4,
                ),
                dtype=torch.int32,
                device=device,
            )
            # compute the rotamer offsets for each of the two bodies for all copies of the constraint
            cs = torch.ones_like(new_fn_inds, dtype=torch.int32)
            start_inds = num_copies_prev_constraint.cumsum(0)
            cs[start_inds] = -(num_copies_prev_constraint - 1)
            cs[0] = 0
            cs = cs.cumsum(0)

            first_block_subrot = cs % n_rots_for_first.repeat_interleave(
                num_copies_per_constraint
            )
            second_block_subrot = cs // n_rots_for_first.repeat_interleave(
                num_copies_per_constraint
            )

            # first, get the rotamer offset for the first and second unique blocks
            block_first_rotamer_offset = rot_offset_for_block[
                constraint_poses, first_block_inds
            ].repeat_interleave(num_copies_per_constraint)
            block_second_rotamer_offset = rot_offset_for_block[
                constraint_poses, second_block_inds
            ].repeat_interleave(num_copies_per_constraint)

            # compute the first and second rotamers involved
            rot_pose = constraint_poses.repeat_interleave(num_copies_per_constraint)
            first_rot = block_first_rotamer_offset + first_block_subrot
            second_rot = block_second_rotamer_offset + second_block_subrot

            # get the matching blocks within each constraint
            constraint_blocks_match_first = (
                constraint_blocks == constraint_blocks[:, 0].unsqueeze(-1)
            ).repeat_interleave(num_copies_per_constraint, dim=0)
            constraint_blocks_match_second = (
                constraint_blocks != constraint_blocks[:, 0].unsqueeze(-1)
            ).repeat_interleave(num_copies_per_constraint, dim=0)

            atoms_rep = constraint_ats.repeat_interleave(
                num_copies_per_constraint, dim=0
            )

            first_counts_per_row = constraint_blocks_match_first.sum(dim=1)
            second_counts_per_row = constraint_blocks_match_second.sum(dim=1)

            new_atoms[constraint_blocks_match_first] = rot_coord_offset[
                first_rot
            ].repeat_interleave(first_counts_per_row)
            new_atoms[constraint_blocks_match_second] = rot_coord_offset[
                second_rot
            ].repeat_interleave(second_counts_per_row)

            new_atoms += atoms_rep

            # combine them
            nonlocal rotamerized_atoms
            rotamerized_atoms = torch.cat((rotamerized_atoms, new_atoms))

            nonlocal rotamerized_fn_inds
            rotamerized_fn_inds = torch.cat((rotamerized_fn_inds, new_fn_inds))

            nonlocal rotamerized_params
            rotamerized_params = torch.cat((rotamerized_params, new_params))

            nonlocal rotamerized_inds
            rotamerized_indices = torch.stack((rot_pose, first_rot, second_rot), dim=-1)
            rotamerized_inds = torch.cat((rotamerized_inds, rotamerized_indices))

        # convert all pose+block+atom inds into just global atom inds
        def add_non_rotamerized_cnstrs():

            # get the rotamer for every constraint atom
            constraint_rots = rot_offset_for_block[constraint_poses, constraint_blocks]

            # finally, compute the offset into the coordinates tensor
            constraint_atom_inds = rot_coord_offset[constraint_rots] + constraint_ats

            # print('cstr5',constraint_atom_inds)
            nonlocal rotamerized_atoms
            rotamerized_atoms = torch.cat((rotamerized_atoms, constraint_atom_inds))

            nonlocal rotamerized_fn_inds
            rotamerized_fn_inds = torch.cat((rotamerized_fn_inds, constraint_fn_inds))

            nonlocal rotamerized_params
            rotamerized_params = torch.cat((rotamerized_params, constraint_params))

            nonlocal rotamerized_inds
            rotamerized_inds = torch.cat(
                (rotamerized_inds, constraint_set.constraint_unique_blocks)
            )

        has_rotamers = (n_rots_for_block.max() > 1).item()
        if has_rotamers:
            add_twobody_vectorized()
            add_onebody_vectorized()
        else:
            add_non_rotamerized_cnstrs()

        def score_cnstrs(functions, types, atom_coords, params):
            cnstr_scores = torch.full(
                (len(types),), 0, dtype=torch.float32, device=coords.device
            )

            for ind, fn in enumerate(functions):

                # get the constraints that match this constraint type
                c_inds = torch.nonzero(types == ind)

                # compute the scores for this function
                fn_coords = atom_coords[c_inds].squeeze(1)
                result = fn(fn_coords, params[c_inds].squeeze(1))
                result = result.unsqueeze(1)

                # add the result to the output tensor
                cnstr_scores[c_inds] += result

            return cnstr_scores

        # extract coordinates from the computed global atom indices
        atom_coords = coords[rotamerized_atoms.flatten()].view(
            rotamerized_atoms.size(0), rotamerized_atoms.size(1), 3
        )

        # score the constraints
        scores = score_cnstrs(
            constraint_set.constraint_functions,
            rotamerized_fn_inds,
            atom_coords,
            rotamerized_params,
        )

        if not has_rotamers:
            max_n_blocks = first_rot_block_type.shape[1]
            n_poses = first_rot_block_type.shape[0]
            sparse_scores = torch.sparse_coo_tensor(
                rotamerized_inds.transpose(0, 1),
                scores,
                size=(n_poses, max_n_blocks, max_n_blocks),
            )

            scores = sparse_scores.to_dense()
            if not output_block_pair_energies:
                scores = scores.sum(dim=(1, 2))

        return scores.unsqueeze(0), rotamerized_inds.transpose(0, 1)
