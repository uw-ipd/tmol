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
        self.device = device
        self.constraint_functions = []

    #################### PROPERTIES #####################

    def add_constraints_to_all_poses(self, fn, atom_indices, params=None):
        nposes = self.pose_stack.n_poses
        self.add_constraints(fn, atom_indices, params, nposes=nposes)

    def add_constraints(self, fn, atom_indices, params=None, nposes=0):
        """Add multiple constraints all using the same functional form.

        fn: the constraint function to use. It should take two arguments
            1. the set of atom coordinates as an [n_csts x n_atoms_per_cst x 3] tensor
            2. the set of parameters as an [n_csts x n_param_vals_per_cst] tensor
        atom_indices: [n_csts x n_atoms_per_cst x 3]
           Atom indices should be given as tuples of (pose_index, block_index, atom_within_block_index)
           All of the indices for a single constraint must live in the same pose.
        params: [n_csts x n_param_vals_per_cst]
           Parameters for each of the constraints; e.g. x0 and k for a harmonic constraint.
        """

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

        new_constraint_function_inds = torch.full(
            (num_to_add,), 0, dtype=torch.int32, device=self.device
        )
        new_constraint_function_inds[:] = fn_index
        self.constraint_function_inds = torch.cat(
            (self.constraint_function_inds, new_constraint_function_inds)
        )

        new_atom_indices = torch.full(
            (num_to_add, self.MAX_N_ATOMS, 3), 0, dtype=torch.int32, device=self.device
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
