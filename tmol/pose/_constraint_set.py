from collections.abc import Callable, Sequence

import attr
import torch

from tmol.types import Tensor
from tmol.utility.tensor import exclusive_cumsum1d

ConstraintFunction = Callable[
    [Tensor[torch.float32][:, :, 3], Tensor[torch.float32][:, :]],
    Tensor[torch.float32][:],
]


@attr.s(frozen=True, slots=True, auto_attribs=True)
class ConstraintSet:
    """ """

    MAX_N_ATOMS = 4

    device: torch.device
    n_poses: int
    constraint_function_inds: Tensor[torch.int32][:]
    constraint_atoms: Tensor[torch.int32][:, 4, 3]
    constraint_params: Tensor[torch.float32][:, :]
    constraint_num_unique_blocks: Tensor[torch.int32][:]
    constraint_unique_blocks: Tensor[torch.int32][:, 3]
    constraint_functions: tuple[ConstraintFunction, ...]

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
    def concatenate(  # noqa: C901
        cls,
        constraint_sets: Sequence["ConstraintSet | None"],
        from_multiple_pose_stacks: bool = True,
        n_poses: int | None = None,
        ps_offset: Tensor[torch.int64][:] | None = None,
    ) -> "ConstraintSet | None":
        """Combine constraint sets while deduplicating their scoring functions.

        Args:
            constraint_sets: Constraint sets to combine; ``None`` entries are
                retained when calculating pose offsets.
            from_multiple_pose_stacks: Shift pose indices between inputs when
                true. When false, all inputs describe the same pose batch.
            n_poses: Pose count for the result, inferred when omitted.
            ps_offset: Per-input pose offsets, inferred when omitted.

        Returns:
            The combined constraint set, or ``None`` when every input is
            ``None``.
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

        constraint_functions_list: list[ConstraintFunction] = []
        remapped_function_inds: list[Tensor[torch.int32][:]] = []
        for cs in constraint_sets:
            if cs is None:
                continue
            # Remap each source set at once on-device. Indexing this Python
            # mapping with CUDA scalars would synchronize once per constraint.
            function_remap: list[int] = []
            for function in cs.constraint_functions:
                try:
                    function_index = constraint_functions_list.index(function)
                except ValueError:
                    constraint_functions_list.append(function)
                    function_index = len(constraint_functions_list) - 1
                function_remap.append(function_index)
            function_remap_tensor = torch.tensor(
                function_remap, dtype=torch.int32, device=device
            )
            remapped_function_inds.append(
                function_remap_tensor[cs.constraint_function_inds]
            )

        new_constraint_function_inds = torch.cat(remapped_function_inds)
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
                new_constraint_unique_blocks[
                    cs_offset[i] : cs_offset[i] + n_cs_constraints, 0
                ] += ps_offset[i]
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
        """Return a copy with independent tensor storage."""
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

    def split(self, index: int) -> "ConstraintSet":
        """Split out a single pose's worth of constraints from a batch."""
        # find the constraints that apply to this pose
        is_constraint_for_pose = self.constraint_atoms[:, :, 0] == index
        constraint_inds = torch.where(is_constraint_for_pose.any(dim=1))[0]

        # note: constraint_functions is shallow-copied; this might seem dangerous
        # because we might worry about the original constraint set modifying
        # its constraint functions, but this is actually safe because 1. Tuples
        # are immutable, and 2. the original ConstraintSet only ever changes
        # by creating a new ConstraintSet, so really, it also is immutable.
        new_constraint_atoms = self.constraint_atoms[constraint_inds].clone()
        real_pose_for_atoms = new_constraint_atoms[:, :, 0] != -1
        nz_real_pose_for_atoms = torch.nonzero(real_pose_for_atoms, as_tuple=False)
        new_constraint_atoms[
            nz_real_pose_for_atoms[:, 0], nz_real_pose_for_atoms[:, 1], 0
        ] = 0  # reset the pose index to 0 since we're splitting out a single pose
        new_constraint_unique_blocks = self.constraint_unique_blocks[
            constraint_inds
        ].clone()
        new_constraint_unique_blocks[:, 0] = (
            0  # reset the pose index to 0 since we're splitting out a single pose
        )
        return attr.evolve(
            self,
            n_poses=1,
            constraint_function_inds=self.constraint_function_inds[
                constraint_inds
            ].clone(),
            constraint_atoms=new_constraint_atoms,
            constraint_params=self.constraint_params[constraint_inds].clone(),
            constraint_num_unique_blocks=self.constraint_num_unique_blocks[
                constraint_inds
            ].clone(),
            constraint_unique_blocks=new_constraint_unique_blocks,
        )

    #################### PROPERTIES #####################

    @staticmethod
    def count_unique_blocks(
        atom_indices: Tensor[torch.int32][:, :, 3],
    ) -> Tensor[torch.int32][:]:
        """Count distinct consecutive blocks referenced by each constraint."""
        constraint_blocks = atom_indices[:, :, 1]
        diffs = constraint_blocks[:, 1:] != constraint_blocks[:, :-1]
        return diffs.sum(dim=1, dtype=torch.int32) + 1

    def add_constraints_to_all_poses(
        self,
        fn: ConstraintFunction,
        atom_indices: Tensor[torch.int32][:, :, :],
        params: Tensor[torch.float32][:, :] | None = None,
    ) -> "ConstraintSet":
        """Add the same constraints to every pose in the batch.

        Args:
            fn: Function that scores coordinates and per-constraint parameters.
            atom_indices: Integer ``[constraint, atom, block/atom]`` indices, or
                full ``[constraint, atom, pose/block/atom]`` indices.
            params: Optional float ``[constraint, parameter]`` values.

        Returns:
            A new constraint set containing the replicated constraints.
        """
        if atom_indices.ndim == 3 and atom_indices.size(2) == 3:
            # if we just drop the "which-pose-is-it-from? dimension", then
            # the normal call to add_constraints will apply it to all poses
            atom_indices = atom_indices[:, :, 1:3]
        return self.add_constraints(fn, atom_indices, params)

    def add_constraints(
        self,
        fn: ConstraintFunction,
        atom_indices: Tensor[torch.int32][:, :, :],
        params: Tensor[torch.float32][:, :] | None = None,
    ) -> "ConstraintSet":  # noqa: C901
        """Return a constraint set containing the existing and new constraints.

        Args:
            fn: Function that scores coordinates and per-constraint parameters.
            atom_indices: Integer ``[constraint, atom, pose/block/atom]`` indices.
                Omitting the pose column applies each constraint to every pose.
            params: Optional float ``[constraint, parameter]`` values.

        Returns:
            A new constraint set containing the added constraints.
        """
        if atom_indices.ndim != 3 or atom_indices.size(2) not in (2, 3):
            raise ValueError("atom_indices must have shape [constraint, atom, 2 or 3]")
        if not 0 < atom_indices.size(1) <= self.MAX_N_ATOMS:
            raise ValueError(
                f"constraints must contain between 1 and {self.MAX_N_ATOMS} atoms"
            )
        if atom_indices.dtype not in (torch.int32, torch.int64):
            raise TypeError("atom_indices must contain 32- or 64-bit integers")

        atom_indices = atom_indices.to(device=self.device, dtype=torch.int32)
        if params is not None:
            if params.ndim != 2 or params.size(0) != atom_indices.size(0):
                raise ValueError(
                    "params must have shape [constraint, parameter] with one row "
                    "per constraint"
                )
            params = params.to(device=self.device, dtype=torch.float32)

        empty_at_start = len(self.constraint_functions) == 0

        constraint_functions_list = list(self.constraint_functions)
        try:
            fn_index = constraint_functions_list.index(fn)
        except ValueError:
            constraint_functions_list.append(fn)
            fn_index = len(constraint_functions_list) - 1

        if (
            atom_indices.size(2) == 2
        ):  # The user did not input pose indices, copy to all poses
            filled_atom_indices = torch.zeros(
                (atom_indices.size(0), atom_indices.size(1), 3),
                dtype=torch.int32,
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
        max_params = (
            new_params.size(1)
            if empty_at_start
            else max(new_params.size(1), self.constraint_params.size(1))
        )
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

    @staticmethod
    def replicate_constraints(
        n_poses: int,
        c_atms: Tensor[torch.int32][:, :, 3],
        c_params: Tensor[torch.float32][:, :] | None,
    ) -> tuple[Tensor[torch.int32][:, :, 3], Tensor[torch.float32][:, :] | None]:
        """Replicate pose-independent constraints and optional parameters."""
        ncnstr = c_atms.size(0)

        atoms = c_atms.repeat(n_poses, 1, 1)
        params = None if c_params is None else c_params.repeat(n_poses, 1)
        poses = torch.arange(n_poses, dtype=atoms.dtype, device=atoms.device)
        atoms[:, :, 0] = poses.repeat_interleave(ncnstr).unsqueeze(1)

        return atoms, params
