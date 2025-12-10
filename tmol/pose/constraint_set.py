import torch

from tmol.types.torch import Tensor


class ConstraintSet:
    """ """

    MAX_N_ATOMS = 4

    device: torch.device
    # constraint_index x (type, pose_index)
    constraint_function_inds: Tensor[torch.int][:]
    constraint_atom: Tensor[torch.int][:, 4, 3]
    constraint_params: Tensor[torch.float32][:, :]
    constraint_num_unique_blocks: Tensor[torch.int][:]
    constraint_unique_blocks: Tensor[torch.int][:, :]

    def __init__(self, device):
        self.constraint_function_inds = torch.full(
            (0,), 0, dtype=torch.int32, device=device
        )
        self.constraint_atoms = torch.full(
            (0, self.MAX_N_ATOMS, 3), 0, dtype=torch.int32, device=device
        )
        self.constraint_params = torch.full(
            (0, 1), 0, dtype=torch.float32, device=device
        )
        self.constraint_num_unique_blocks = torch.full(
            (0,), 0, dtype=torch.int32, device=device
        )
        self.constraint_unique_blocks = torch.full(
            (0, 3), 0, dtype=torch.int32, device=device
        )
        self.device = device
        self.constraint_functions = []

    #################### PROPERTIES #####################

    def count_unique_blocks(self, atom_indices):
        # sorted_blocks_per_constraint, _ = atom_indices[:,:,1]
        # now shift by 1 and count the differences
        # diffs = sorted_blocks_per_constraint[:, 1:] != sorted_blocks_per_constraint[:, :-1]
        constraint_blocks = atom_indices[:, :, 1]
        # now shift by 1 and count the differences
        diffs = constraint_blocks[:, 1:] != constraint_blocks[:, :-1]

        temp = diffs.sum(dim=1) + 1
        # print(temp)
        return temp

    def add_constraints_to_all_poses(self, fn, atom_indices, params=None):
        nposes = self.pose_stack.n_poses
        self.add_constraints(fn, atom_indices, params, nposes=nposes)

    def constrain_all_ca(self):
        ps = self.pose_stack
        cnstr_atoms = torch.full((0, 1, 3), 0, dtype=torch.int32, device=ps.device)
        cnstr_params = torch.full((0, 3), 0, dtype=torch.float32, device=ps.device)

        for pose_ind in range(ps.n_poses):
            for block_ind in range(ps.max_n_blocks):
                if ps.is_real_block(pose_ind, block_ind):
                    block_type = ps.block_type(pose_ind, block_ind)

                    # C vs CA? check if has C?
                    ca_ind = block_type.atom_to_idx["C"]
                    ca_coords = ps.coords[pose_ind][
                        ps.block_coord_offset[pose_ind, block_ind] + ca_ind
                    ]

                    # print('atom_inds',torch.tensor([pose_ind, block_ind, ca_ind]))
                    # print('ca_coords',ca_coords)

                    cnstr_atoms = torch.cat(
                        [
                            cnstr_atoms,
                            torch.tensor(
                                [[[pose_ind, block_ind, ca_ind]]],
                                dtype=torch.int32,
                                device=ps.device,
                            ),
                        ]
                    )
                    cnstr_params = torch.cat([cnstr_params, ca_coords.unsqueeze(0)])

        # print(cnstr_atoms)
        # print(cnstr_params)

        def harmonic_coordinate(atoms, params):
            atoms1 = atoms[:, 0]
            atoms2 = params[:, :]
            dist = torch.linalg.norm(atoms1 - atoms2, dim=-1)
            return (dist) ** 2

        self.add_constraints(harmonic_coordinate, cnstr_atoms, cnstr_params)

    def add_constraints(self, fn, atom_indices, params=None, nposes=0):
        def find_or_insert(value, lst):
            if value in lst:
                return lst.index(value)
            lst.append(value)
            return lst.index(value)

        fn_index = find_or_insert(fn, self.constraint_functions)

        if (
            atom_indices.size(2) == 2
        ):  # they did not input pose indices, copy to all poses
            filled_atom_indices = torch.zeros(
                (atom_indices.size(0), atom_indices.size(1), 3),
                dtype=torch.float32,
                device=self.device,
            )
            filled_atom_indices[:, :, 1:3] = atom_indices
            atom_indices, params = self.replicate_constraints(
                nposes, filled_atom_indices, params
            )

        # constraints
        num_to_add = atom_indices.size(0)

        # Make sure no one is mixing atoms from multiple poses in a single constraint
        # flatten
        flat = atom_indices[:, :, 0].view(-1)
        # find the sizes of consecutive occurences of the pose index
        uniq_cnt = torch.unique_consecutive(flat, return_counts=True)[1]
        # make sure those sizes are all divisible by the # of atoms
        if (uniq_cnt % atom_indices.size(1)).any():
            raise Exception(
                "One or more constraints contains atoms from multiple poses"
            )

        constraint_poses = atom_indices[:, 0, 0]
        constraint_blocks = atom_indices[:, :, 1]
        constraint_blocks_rolled = constraint_blocks.roll(shifts=1, dims=-1)
        constraint_blocks_changed = (constraint_blocks != constraint_blocks_rolled).to(
            torch.int32
        )
        constraint_blocks_changed[:, 0] = 0
        constraint_block_first_change = torch.argmax(
            constraint_blocks_changed, dim=1
        ).unsqueeze(-1)
        first_block_inds = constraint_blocks[:, 0]
        second_block_inds = constraint_blocks.gather(
            1, constraint_block_first_change
        ).squeeze(-1)

        # print("POSES", constraint_poses)
        # print("BLOCK1", first_block_inds)
        # print("BLOCK2", second_block_inds)
        self.constraint_unique_blocks = torch.cat(
            [
                self.constraint_unique_blocks,
                torch.stack(
                    [constraint_poses, first_block_inds, second_block_inds], dim=1
                ),
            ]
        )
        # print(self.constraint_unique_blocks)

        new_constraint_function_inds = torch.full(
            (num_to_add,), 0, dtype=torch.int32, device=self.device
        )
        new_constraint_function_inds[:] = fn_index
        self.constraint_function_inds = torch.cat(
            (self.constraint_function_inds, new_constraint_function_inds)
        )

        num_unique_blocks_per_constraint = self.count_unique_blocks(atom_indices)
        self.constraint_num_unique_blocks = torch.cat(
            (self.constraint_num_unique_blocks, num_unique_blocks_per_constraint)
        )

        new_atom_indices = torch.full(
            (num_to_add, self.MAX_N_ATOMS, 3), -1, dtype=torch.int32, device=self.device
        )
        new_atom_indices[:, 0 : atom_indices.size(1), :] = atom_indices
        # now copy the last real atom into the final atom slot so that we can attribute score correctly later
        new_atom_indices[:, self.MAX_N_ATOMS - 1, :] = atom_indices[:, -1, :]
        self.constraint_atoms = torch.cat((self.constraint_atoms, new_atom_indices))

        new_params = torch.full(
            (num_to_add, 0), 0.0, dtype=torch.float32, device=self.device
        )
        if params is not None:
            new_params = params
        max_params = max(new_params.size(1), self.constraint_params.size(1))
        t1 = torch.zeros(
            (self.constraint_params.size(0), max_params),
            dtype=torch.float32,
            device=self.device,
        )
        t1[:, 0 : self.constraint_params.size(1)] = self.constraint_params
        t2 = torch.zeros(
            (new_params.size(0), max_params), dtype=torch.float32, device=self.device
        )
        t2[:, 0 : new_params.size(1)] = new_params
        self.constraint_params = torch.cat((t1, t2))

    def replicate_constraints(self, nposes, c_atms, c_params):
        ncnstr = c_atms.size(0)
        natoms = c_atms.size(1)

        atoms = c_atms.repeat(nposes, 1, 1)
        params = c_params.repeat(nposes, 1)
        poses = torch.arange(0, nposes).repeat_interleave(natoms * ncnstr)
        atoms[:, :, 0] = poses.view(nposes * ncnstr, natoms)

        return atoms, params
