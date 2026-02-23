import torch
import attr

from typing import Optional, Tuple
from tmol.types.torch import Tensor
from tmol.utility.tensor.common_operations import exclusive_cumsum1d


@attr.s(frozen=True, slots=True, auto_attribs=True)
class ConstraintSet:
    """ """

    MAX_N_ATOMS = 4

    device: torch.device
    n_poses: int
    constraint_function_inds: Tensor[torch.int][:]
    constraint_atoms: Tensor[torch.int][:, 4, 3]
    constraint_params: Tensor[torch.float32][:, :]
    constraint_num_unique_blocks: Tensor[torch.int][:]
    constraint_unique_blocks: Tensor[torch.int][:, :]
    constraint_functions: Tuple

    @classmethod
    def create_empty(cls, device: torch.device, n_poses: int) -> "ConstraintSet":
        return ConstraintSet(
            device=device,
            n_poses=n_poses,
            constraint_function_inds=torch.full(
                (0,), 0, dtype=torch.int32, device=device
            ),
            constraint_atoms=torch.full(
                (0, cls.MAX_N_ATOMS, 3), 0, dtype=torch.int32, device=device
            ),
            constraint_params=torch.full((0, 1), 0, dtype=torch.float32, device=device),
            constraint_num_unique_blocks=torch.full(
                (0,), 0, dtype=torch.int32, device=device
            ),
            constraint_unique_blocks=torch.full(
                (0, 3), 0, dtype=torch.int32, device=device
            ),
            constraint_functions=tuple(),
        )

    @classmethod
    def concatenate(
        cls,
        constraint_sets: Tuple[Optional["ConstraintSet"], ...],
        from_multiple_pose_stacks: bool = True,
        n_poses: Optional[int] = None,
        ps_offset: Optional[Tensor[torch.int64][:]] = None,
    ) -> Optional["ConstraintSet"]:
        """Concatenate multiple ConstraintSets into a single ConstraintSet.

        This function is particularly useful if you're creating a PoseStack from multiple
        PoseStacks, each of which has its own ConstraintSet. In that case, n_poses
        and ps_offset will be readily available. In this use case, "from_multiple_pose_stacks"
        should be set to True.

        The other use case is in creating multiple types of constraints for a single
        PoseStack and then combining them in a single go. This will be more efficient
        than repeatedly invoking add_constraints() as it skips the N^2 copy operations.
        In this use case, "from_multiple_pose_stacks" should be set to False.
        """

        device = None
        for cs in constraint_sets:
            if cs is not None:
                if device is None:
                    device = cs.device
                else:
                    assert (
                        device == cs.device
                    ), "All ConstraintSets must be on the same device"

        if device is None:
            return None

        # now set up n_poses and ps_offset if not provided based on wether these
        # constraint sets are coming from multiple pose stacks or all from the
        # same pose stack
        if n_poses is None:
            if from_multiple_pose_stacks:
                n_poses = sum(cs.n_poses for cs in constraint_sets if cs is not None)
            else:
                n_poses = next(cs for cs in constraint_sets if cs is not None).n_poses
        if ps_offset is None:
            if from_multiple_pose_stacks:
                ps_offset = exclusive_cumsum1d(
                    torch.tensor(
                        [cs.n_poses if cs is not None else 0 for cs in constraint_sets],
                        dtype=torch.int64,
                        device=device,
                    )
                )
            else:
                ps_offset = torch.zeros(
                    (len(constraint_sets),), dtype=torch.int64, device=device
                )

        cs_offset = exclusive_cumsum1d(
            torch.tensor(
                [
                    cs.constraint_atoms.shape[0] if cs is not None else 0
                    for cs in constraint_sets
                ],
                dtype=torch.int64,
            )
        )

        constraint_functions_list = []
        constraint_function_inds = []
        for i, cs in enumerate(constraint_sets):
            if cs is not None:
                constraint_function_inds.append([])
                for j, func in enumerate(cs.constraint_functions):
                    found_existing = False
                    for k, func_existing in enumerate(constraint_functions_list):
                        if func_existing == func:
                            constraint_function_inds[-1].append(k)
                            found_existing = True
                            break
                    if not found_existing:
                        constraint_functions_list.append(func)
                        constraint_function_inds[-1].append(
                            len(constraint_functions_list) - 1
                        )

        new_constraint_function_inds = torch.tensor(
            [
                constraint_function_inds[i][j]
                for i, cs in enumerate(constraint_sets)
                if cs is not None
                for j, func in enumerate(cs.constraint_functions)
            ],
            device=device,
            dtype=torch.int32,
        )
        n_constraints = new_constraint_function_inds.size(0)
        new_constraint_atoms = torch.full(
            (n_constraints, cls.MAX_N_ATOMS, 3), -1, dtype=torch.int32, device=device
        )
        max_n_params = (
            max(
                cs.constraint_params.size(1) for cs in constraint_sets if cs is not None
            )
            if n_constraints > 0
            else 0
        )
        new_constraint_params = torch.full(
            (n_constraints, max_n_params), 0.0, dtype=torch.float32, device=device
        )
        new_constraint_num_unique_blocks = torch.full(
            (n_constraints,), 0, dtype=torch.int32, device=device
        )
        new_constraint_unique_blocks = torch.full(
            (n_constraints, 3), 0, dtype=torch.int32, device=device
        )
        for i, cs in enumerate(constraint_sets):
            if cs is not None:
                n_cs_constraints = cs.constraint_function_inds.size(0)
                constraint_atoms_shifted = cs.constraint_atoms.detach().clone()
                constraint_atoms_pose = constraint_atoms_shifted[:, :, 0]
                is_real_pose = constraint_atoms_pose[:, :] != -1
                constraint_atoms_pose[is_real_pose] += ps_offset[i]
                constraint_atoms_shifted[:, :, 0] = constraint_atoms_pose
                new_constraint_atoms[
                    cs_offset[i] : cs_offset[i] + n_cs_constraints, :, :
                ] = constraint_atoms_shifted
                new_constraint_params[
                    cs_offset[i] : cs_offset[i] + n_cs_constraints,
                    0 : cs.constraint_params.size(1),
                ] = cs.constraint_params
                new_constraint_num_unique_blocks[
                    cs_offset[i] : cs_offset[i] + n_cs_constraints
                ] = cs.constraint_num_unique_blocks
                new_constraint_unique_blocks[
                    cs_offset[i] : cs_offset[i] + n_cs_constraints, :
                ] = cs.constraint_unique_blocks
        return ConstraintSet(
            device=device,
            n_poses=n_poses,
            constraint_function_inds=new_constraint_function_inds,
            constraint_atoms=new_constraint_atoms,
            constraint_params=new_constraint_params,
            constraint_num_unique_blocks=new_constraint_num_unique_blocks,
            constraint_unique_blocks=new_constraint_unique_blocks,
            constraint_functions=tuple(constraint_functions_list),
        )

    def clone(self) -> "ConstraintSet":
        return attr.evolve(
            self,
            constraint_function_inds=self.constraint_function_inds.clone(),
            constraint_atoms=self.constraint_atoms.clone(),
            constraint_params=self.constraint_params.clone(),
            constraint_num_unique_blocks=self.constraint_num_unique_blocks.clone(),
            constraint_unique_blocks=self.constraint_unique_blocks.clone(),
        )

    def to(self, device: torch.device) -> "ConstraintSet":
        return attr.evolve(
            self,
            device=device,
            constraint_function_inds=self.constraint_function_inds.to(device),
            constraint_atoms=self.constraint_atoms.to(device),
            constraint_params=self.constraint_params.to(device),
            constraint_num_unique_blocks=self.constraint_num_unique_blocks.to(device),
            constraint_unique_blocks=self.constraint_unique_blocks.to(device),
        )

    #################### PROPERTIES #####################

    def count_unique_blocks(self, atom_indices):
        # sorted_blocks_per_constraint, _ = atom_indices[:,:,1]
        # now shift by 1 and count the differences
        # diffs = sorted_blocks_per_constraint[:, 1:] != sorted_blocks_per_constraint[:, :-1]
        constraint_blocks = atom_indices[:, :, 1]
        # now shift by 1 and count the differences
        diffs = constraint_blocks[:, 1:] != constraint_blocks[:, :-1]

        temp = diffs.sum(dim=1) + 1
        return temp

    def add_constraints_to_all_poses(
        self, fn, atom_indices, params=None
    ) -> "ConstraintSet":
        """If all Poses in the PoseStack should be constrained in the same way, then
        this convenience function will take a list of atom indices for a single Pose
        and replicate them across all the Poses in the PoseStack."""
        if atom_indices.size(2) == 3:
            # if we just drop the "which pose is it from dimension", then
            # the normal call to add_constraints will apply it to all poses
            atom_indices = atom_indices[:, :, 1:3]
        return self.add_constraints(fn, atom_indices, params)

    def add_constraints(self, fn, atom_indices, params=None) -> "ConstraintSet":
        """
        Create a new ConstraintSet that includes all the old constraints plus the new ones.

        atom_indices: either (n_constraints, n_atoms, 3) or (n_constraints, n_atoms, 2)
                      If the latter, the constraint will be applied to all poses
        """
        empty_at_start = len(self.constraint_functions) == 0

        def find_or_insert(value, lst):
            if value in lst:
                return lst.index(value)
            lst.append(value)
            return lst.index(value)

        constraint_functions_list = list(self.constraint_functions)
        fn_index = find_or_insert(fn, constraint_functions_list)

        if (
            atom_indices.size(2) == 2
        ):  # The user did not input pose indices, copy to all poses
            filled_atom_indices = torch.zeros(
                (atom_indices.size(0), atom_indices.size(1), 3),
                dtype=torch.float32,
                device=self.device,
            )
            filled_atom_indices[:, :, 1:3] = atom_indices
            atom_indices, params = self.replicate_constraints(
                self.n_poses, filled_atom_indices, params
            )

        # constraints
        num_to_add = atom_indices.size(0)

        # Make sure the users does not mix atoms from multiple poses into a single constraint
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

        if not empty_at_start:
            new_constraint_unique_blocks = torch.cat(
                [
                    self.constraint_unique_blocks,
                    torch.stack(
                        [constraint_poses, first_block_inds, second_block_inds], dim=1
                    ),
                ]
            )
        else:
            new_constraint_unique_blocks = torch.stack(
                [constraint_poses, first_block_inds, second_block_inds], dim=1
            )

        constraint_function_inds = torch.full(
            (num_to_add,), 0, dtype=torch.int32, device=self.device
        )
        constraint_function_inds[:] = fn_index
        if not empty_at_start:
            new_constraint_function_inds = torch.cat(
                (self.constraint_function_inds, constraint_function_inds)
            )
        else:
            new_constraint_function_inds = constraint_function_inds

        num_unique_blocks_per_constraint = self.count_unique_blocks(atom_indices)
        if not empty_at_start:
            new_constraint_num_unique_blocks = torch.cat(
                (self.constraint_num_unique_blocks, num_unique_blocks_per_constraint)
            )
        else:
            new_constraint_num_unique_blocks = num_unique_blocks_per_constraint

        new_atom_indices = torch.full(
            (num_to_add, self.MAX_N_ATOMS, 3), -1, dtype=torch.int32, device=self.device
        )
        new_atom_indices[:, 0 : atom_indices.size(1), :] = atom_indices
        # now copy the last real atom into the final atom slot so that we can attribute score correctly later
        new_atom_indices[:, self.MAX_N_ATOMS - 1, :] = atom_indices[:, -1, :]
        if not empty_at_start:
            new_constraint_atoms = torch.cat((self.constraint_atoms, new_atom_indices))
        else:
            new_constraint_atoms = new_atom_indices

        new_params = torch.full(
            (num_to_add, 0), 0.0, dtype=torch.float32, device=self.device
        )
        if params is not None:
            new_params = params
        max_params = max(new_params.size(1), self.constraint_params.size(1))
        if not empty_at_start:
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
        if not empty_at_start:
            new_constraint_params = torch.cat((t1, t2))
        else:
            new_constraint_params = t2

        return attr.evolve(
            self,
            constraint_function_inds=new_constraint_function_inds,
            constraint_atoms=new_constraint_atoms,
            constraint_params=new_constraint_params,
            constraint_num_unique_blocks=new_constraint_num_unique_blocks,
            constraint_unique_blocks=new_constraint_unique_blocks,
            constraint_functions=tuple(constraint_functions_list),
        )

    @classmethod
    def replicate_constraints(cls, n_poses, c_atms, c_params):
        ncnstr = c_atms.size(0)
        natoms = c_atms.size(1)

        atoms = c_atms.repeat(n_poses, 1, 1)
        params = c_params.repeat(n_poses, 1)
        poses = torch.arange(0, n_poses).repeat_interleave(natoms * ncnstr)
        atoms[:, :, 0] = poses.view(n_poses * ncnstr, natoms)

        return atoms, params
