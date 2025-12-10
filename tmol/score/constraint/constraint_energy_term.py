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


import os, sys


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

    # def render_whole_pose_scoring_module(self, pose_stack: PoseStack):
    # pbt = pose_stack.packed_block_types

    # return ConstraintWholePoseScoringModule(
    # pose_stack_block_coord_offset=pose_stack.block_coord_offset,
    # pose_stack_block_types=pose_stack.block_type_ind,
    # pose_stack_inter_block_connections=pose_stack.inter_residue_connections,
    # bt_atom_downstream_of_conn=pbt.atom_downstream_of_conn,
    # constraint_set=pose_stack.get_constraint_set(),
    # )

    def get_score_term_function(self):
        return self.constraint_pose_scores

    def get_score_term_attributes(self, pose_stack: PoseStack):
        return [
            pose_stack.get_constraint_set(),
        ]

    def constraint_pose_scores(
        self,
        coords,
        rot_coord_offset,  # rot coord offset
        pose_ind_for_atom,  # pose_ind_for_atom?? unused:w
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
        device = constraint_set.constraint_atoms.device
        unique_blocks = constraint_set.constraint_num_unique_blocks
        constraint_atoms = constraint_set.constraint_atoms
        constraint_fn_inds = constraint_set.constraint_function_inds
        constraint_params = constraint_set.constraint_params

        constraint_poses = constraint_atoms[:, :, 0]
        constraint_blocks = constraint_atoms[:, :, 1]
        constraint_ats = constraint_atoms[:, :, 2]

        onebody = (unique_blocks == 1).nonzero()
        twobody = (unique_blocks == 2).nonzero().squeeze(-1)
        twobody_old = (unique_blocks == 2).nonzero()
        multibody = (unique_blocks > 2).nonzero()

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

        # n_cnstrs_for_rot_pairs = first_atom_rot_count * second_atom_rot_count

        def add_onebody():
            obs = constraint_atoms[onebody]
            if obs.size(0) == 0:
                return
            for index in range(constraint_atoms[onebody].size(0)):
                constraint = constraint_atoms[onebody][index]
                constraint_fn = constraint_fn_inds[onebody][index]
                constraint_param = constraint_params[onebody][index]
                pose = constraint[0, 0, 0]  # TODO: maybe make this better
                block = constraint[0, 0, 1]
                block_rot_count = n_rots_for_block[pose, block]
                first_rot = rot_offset_for_block[pose, block]
                rot_constraints = torch.zeros(
                    (block_rot_count, 4), dtype=torch.int32, device=device
                )
                nonlocal rotamerized_inds
                for ind1 in range(block_rot_count):
                    rot_constraint = rot_constraints[ind1]
                    rot_constraint[:] = rot_coord_offset[first_rot + ind1]
                    rot_constraint += constraint[0, :, 2]
                    rotamerized_inds = torch.cat(
                        (
                            rotamerized_inds,
                            torch.tensor(
                                [[pose, block, block]], dtype=torch.int32, device=device
                            ),
                        )
                    )

                nonlocal rotamerized_atoms
                nonlocal rotamerized_fn_inds
                nonlocal rotamerized_params
                rotamerized_atoms = torch.cat((rotamerized_atoms, rot_constraints))
                rotamerized_fn_inds = torch.cat(
                    (rotamerized_fn_inds, constraint_fn.repeat(block_rot_count))
                )
                rotamerized_params = torch.cat(
                    (rotamerized_params, constraint_param.repeat(block_rot_count, 1))
                )

        def add_twobody_vectorized():
            with HiddenPrints():
                print("twobody vectorized start")
                # first, compute how many copies of each constraint we get
                if twobody.count_nonzero() == 0:
                    return

                print(constraint_atoms)
                # first, compute the index of the first occurance of the second block, per constraint
                constraint_poses = constraint_atoms[twobody][:, 0, 0]
                constraint_blocks = constraint_atoms[twobody][:, :, 1]
                constraint_ats = constraint_atoms[twobody][:, :, 2]

                print(constraint_atoms[twobody])
                print(constraint_blocks)
                constraint_blocks_rolled = constraint_blocks.roll(shifts=1, dims=-1)
                constraint_blocks_changed = (
                    constraint_blocks != constraint_blocks_rolled
                ).to(torch.int32)
                constraint_blocks_changed[:, 0] = 0
                constraint_block_first_change = torch.argmax(
                    constraint_blocks_changed, dim=1
                ).unsqueeze(-1)

                first_block_inds = constraint_blocks[:, 0]
                second_block_inds = constraint_blocks.gather(
                    1, constraint_block_first_change
                ).squeeze(-1)

                n_rots_for_first = n_rots_for_block[constraint_poses, first_block_inds]
                n_rots_for_second = n_rots_for_block[
                    constraint_poses, second_block_inds
                ]

                # now compute how many copies of each constraint we'll need
                num_copies_per_constraint = n_rots_for_first * n_rots_for_second
                num_copies_prev_constraint = num_copies_per_constraint.roll(shifts=1)
                num_copies_prev_constraint[0] = 0

                # then use repeat_interleave to create new tensors for atoms, fn_inds, and params
                new_fn_inds = constraint_fn_inds[twobody].repeat_interleave(
                    num_copies_per_constraint
                )
                new_params = constraint_params[twobody].repeat_interleave(
                    num_copies_per_constraint, dim=0
                )
                new_atoms = torch.zeros(
                    (
                        new_params.size(0),
                        4,
                    ),
                    dtype=torch.int32,
                    device=device,
                )
                # fn_inds and params are good to go, but the atoms need to be updated to point to the right rotamer atoms

                # 01111-4111-3
                cs = torch.ones_like(new_fn_inds, dtype=torch.int32)
                start_inds = num_copies_prev_constraint.cumsum(0)
                cs[start_inds] = -(num_copies_prev_constraint - 1)
                print("num_cnstrs", num_copies_per_constraint)
                print("num_prev_cnstrs", num_copies_prev_constraint)
                print("PRE-CSUM", cs)
                cs[0] = 0
                cs = cs.cumsum(0)
                print(num_copies_per_constraint)
                print("CSUM", cs)

                first_nrot_rep = n_rots_for_first.repeat_interleave(
                    num_copies_per_constraint
                )
                second_nrot_rep = n_rots_for_second.repeat_interleave(
                    num_copies_per_constraint
                )

                first_block_subrot = cs % n_rots_for_first.repeat_interleave(
                    num_copies_per_constraint
                )
                second_block_subrot = cs // n_rots_for_first.repeat_interleave(
                    num_copies_per_constraint
                )

                print("n_rots:", first_nrot_rep, second_nrot_rep)
                print("subrots:", first_block_subrot, second_block_subrot)

                # first, get the rotamer offset for the first and second unique blocks
                block_first_rotamer_offset = rot_offset_for_block[
                    constraint_poses, first_block_inds
                ].repeat_interleave(num_copies_per_constraint)
                block_second_rotamer_offset = rot_offset_for_block[
                    constraint_poses, second_block_inds
                ].repeat_interleave(num_copies_per_constraint)

                # compute the rotamer offset from the first rotamer
                print(
                    "first_coord_offset:",
                    rot_coord_offset[block_first_rotamer_offset + first_block_subrot],
                )
                print(
                    "second_coord_offset:",
                    rot_coord_offset[block_second_rotamer_offset + second_block_subrot],
                )

                # get the matching blocks within each constraint
                constraint_blocks_match_first = (
                    constraint_blocks == constraint_blocks[:, 0].unsqueeze(-1)
                ).repeat_interleave(num_copies_per_constraint, dim=0)
                constraint_blocks_match_second = (
                    constraint_blocks != constraint_blocks[:, 0].unsqueeze(-1)
                ).repeat_interleave(num_copies_per_constraint, dim=0)

                print(constraint_blocks_match_first, constraint_blocks_match_second)
                print(constraint_ats)
                atoms_rep = constraint_ats.repeat_interleave(
                    num_copies_per_constraint, dim=0
                )

                print(new_atoms[constraint_blocks_match_second])
                first_counts_per_row = constraint_blocks_match_first.sum(dim=1)
                second_counts_per_row = constraint_blocks_match_second.sum(dim=1)
                new_atoms[constraint_blocks_match_first] = rot_coord_offset[
                    block_first_rotamer_offset + first_block_subrot
                ].repeat_interleave(first_counts_per_row)
                new_atoms[constraint_blocks_match_second] = rot_coord_offset[
                    block_second_rotamer_offset + second_block_subrot
                ].repeat_interleave(second_counts_per_row)

                new_atoms += atoms_rep

                print("FINAL ATOMS", new_atoms)

                # combine them

        def add_twobody():

            tbs = constraint_atoms[twobody_old]
            if tbs.size(0) == 0:
                return
            for index in range(constraint_atoms[twobody_old].size(0)):

                constraint = constraint_atoms[twobody_old][index]
                constraint_fn = constraint_fn_inds[twobody_old][index]
                constraint_param = constraint_params[twobody_old][index]

                pose = constraint[0, 0, 0]  # TODO: maybe make this better
                unique_blocks = torch.unique(constraint[:, :, 1])
                unique_blocks = unique_blocks[unique_blocks != -1]

                first_unique_block = unique_blocks[0]
                first_unique_block_inds = constraint[0, :, 1] == first_unique_block
                first_unique_block_rot_count = n_rots_for_block[
                    pose, first_unique_block
                ]
                first_unique_block_first_rot = rot_offset_for_block[
                    pose, first_unique_block
                ]

                second_unique_block = unique_blocks[1]
                second_unique_block_inds = constraint[0, :, 1] == second_unique_block
                second_unique_block_rot_count = n_rots_for_block[
                    pose, second_unique_block
                ]
                second_unique_block_first_rot = rot_offset_for_block[
                    pose, second_unique_block
                ]

                rot_combinations = torch.cartesian_prod(
                    torch.arange(first_unique_block_rot_count),
                    torch.arange(second_unique_block_rot_count),
                )
                # print(rot_combinations)
                rot_constraints = torch.zeros(
                    (rot_combinations.size(0), 4), dtype=torch.int32, device=device
                )

                nonlocal rotamerized_inds
                # now generate all possible rotamer combinations for the constraint, setting the final atom offsets
                for ind1, rot_combo in enumerate(rot_combinations):
                    # set the rot_coord_offsets
                    rot_constraint = rot_constraints[ind1]
                    rot1 = first_unique_block_first_rot + rot_combo[0]
                    rot2 = second_unique_block_first_rot + rot_combo[1]
                    rot_constraint[first_unique_block_inds] = rot_coord_offset[rot1]
                    rot_constraint[second_unique_block_inds] = rot_coord_offset[rot2]
                    # add the atom offset
                    rot_constraint += constraint[0, :, 2]
                    rotamerized_inds = torch.cat(
                        (
                            rotamerized_inds,
                            torch.tensor(
                                [[pose, rot1, rot2]], dtype=torch.int32, device=device
                            ),
                        )
                    )

                # TODO: now add the constraints to the final pool
                nonlocal rotamerized_atoms
                nonlocal rotamerized_fn_inds
                nonlocal rotamerized_params
                rotamerized_atoms = torch.cat((rotamerized_atoms, rot_constraints))
                print(rotamerized_atoms)
                rotamerized_fn_inds = torch.cat(
                    (
                        rotamerized_fn_inds,
                        constraint_fn.repeat(rot_combinations.size(0)),
                    )
                )
                rotamerized_params = torch.cat(
                    (
                        rotamerized_params,
                        constraint_param.repeat(rot_combinations.size(0), 1),
                    )
                )

        # convert all pose+block+atom inds into just global atom inds
        def add_non_rotamerized_cnstrs():

            # get the rotamer for every constraint atom
            constraint_rots = rot_offset_for_block[constraint_poses, constraint_blocks]

            # finally, compute the offset into the coordinates tensor
            constraint_atom_inds = rot_coord_offset[constraint_rots] + constraint_ats

            # print('cstr5',constraint_atom_inds)
            nonlocal rotamerized_atoms
            rotamerized_atoms = torch.cat((rotamerized_atoms, constraint_atom_inds))

        def add_multibody():
            if torch.max(n_rots_for_block) > 1:
                return
            pass

        # add_twobody_vectorized()
        # add_twobody()
        # exit()

        # add_onebody()

        add_non_rotamerized_cnstrs()
        # print(rotamerized_atoms)
        # print(rotamerized_fn_inds)
        # print(rotamerized_params)

        # new_constraint_atoms = torch.repeat_interleave(constraint_atoms, n_cnstrs_for_rot_pairs, dim=0)
        # first_rot_for_cnstr = torch.cat([torch.arange])

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
            constraint_fn_inds,
            atom_coords,
            constraint_params,
        )

        if output_block_pair_energies:
            pass
        else:
            # for each pose, sum up the block scores
            print(constraint_poses.shape)
            scores = torch.bincount(constraint_poses[:, 0].squeeze(0), weights=scores)

        print(scores)

        return scores.unsqueeze(0), constraint_set.constraint_unique_blocks.transpose(
            0, 1
        )
