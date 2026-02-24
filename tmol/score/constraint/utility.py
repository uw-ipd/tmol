import attrs
import torch

from tmol.pose.constraint_set import ConstraintSet
from tmol.pose.pose_stack import PoseStack
from tmol.score.constraint.constraint_energy_term import ConstraintEnergyTerm


def constrain_all_ca(pose_stack: PoseStack) -> PoseStack:
    constraint_set = pose_stack.constraint_set

    cnstr_atoms = torch.full((0, 1, 3), 0, dtype=torch.int32, device=pose_stack.device)
    cnstr_params = torch.full((0, 3), 0, dtype=torch.float32, device=pose_stack.device)

    for pose_ind in range(pose_stack.n_poses):
        for block_ind in range(pose_stack.max_n_blocks):
            if pose_stack.is_real_block(pose_ind, block_ind):
                block_type = pose_stack.block_type(pose_ind, block_ind)

                ca_ind = block_type.atom_to_idx["CA"]
                ca_coords = pose_stack.coords[pose_ind][pose_stack.block_coord_offset[pose_ind, block_ind] + ca_ind]

                cnstr_atoms = torch.cat(
                    [
                        cnstr_atoms,
                        torch.tensor(
                            [[[pose_ind, block_ind, ca_ind]]],
                            dtype=torch.int32,
                            device=pose_stack.device,
                        ),
                    ]
                )
                cnstr_params = torch.cat([cnstr_params, ca_coords.unsqueeze(0)])
    if constraint_set is None:
        constraint_set = ConstraintSet.create_empty(pose_stack.device, pose_stack.n_poses)
    constraint_set = constraint_set.add_constraints(ConstraintEnergyTerm.harmonic_coordinate, cnstr_atoms, cnstr_params)

    return attrs.evolve(
        pose_stack,
        constraint_set=constraint_set,
    )
