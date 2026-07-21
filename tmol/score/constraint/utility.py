import torch
import attrs

from tmol.types.torch import Tensor
from tmol.pose.packed_block_types import PackedBlockTypes
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
                ca_coords = pose_stack.coords[pose_ind][
                    pose_stack.block_coord_offset[pose_ind, block_ind] + ca_ind
                ]

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
        constraint_set = ConstraintSet.create_empty(
            pose_stack.device, pose_stack.n_poses
        )
    constraint_set = constraint_set.add_constraints(
        ConstraintEnergyTerm.harmonic_coordinate, cnstr_atoms, cnstr_params
    )

    return attrs.evolve(
        pose_stack,
        constraint_set=constraint_set,
    )


@attrs.define
class MCAtomIndices:
    max_n_mainchain_atoms: int
    n_mainchain_atoms: Tensor[torch.int64][:]
    mainchain_atoms: Tensor[torch.int64][:, :]
    is_real_mainchain_atom: Tensor[torch.bool][:, :]


def _annotate_mainchain_atom_indices(packed_block_types: PackedBlockTypes) -> None:
    """Get the list of mainchain atoms for each block type and annotate their indices."""
    if hasattr(packed_block_types, "mainchain_atom_indices"):
        return
    mainchain_atom_indices = []
    for block_type in packed_block_types.active_block_types:
        mainchain_atom_indices.append(
            [
                block_type.atom_to_idx[atom_name]
                for atom_name in block_type.properties.polymer.mainchain_atoms
            ]
        )
    n_mainchain_atoms = [len(mc_at_inds) for mc_at_inds in mainchain_atom_indices]

    max_n_mainchain_atoms = max(n_mainchain_atoms)
    mainchain_atom_indices_padded = torch.full(
        (packed_block_types.n_types, max_n_mainchain_atoms),
        -1,
        dtype=torch.int64,
        device=torch.device("cpu"),
    )
    is_real_mainchain_atom = torch.zeros(
        (packed_block_types.n_types, max_n_mainchain_atoms),
        dtype=torch.bool,
        device=torch.device("cpu"),
    )
    for i, mc_at_inds in enumerate(mainchain_atom_indices):
        mainchain_atom_indices_padded[i, : len(mc_at_inds)] = torch.tensor(
            mc_at_inds, dtype=torch.int64, device=torch.device("cpu")
        )
        is_real_mainchain_atom[i, : len(mc_at_inds)] = True
    mainchain_atom_indices_padded = mainchain_atom_indices_padded.to(
        packed_block_types.device
    )
    is_real_mainchain_atom = is_real_mainchain_atom.to(packed_block_types.device)
    n_mainchain_atoms = torch.tensor(
        n_mainchain_atoms, dtype=torch.int64, device=packed_block_types.device
    )
    mcatominds = MCAtomIndices(
        max_n_mainchain_atoms=max_n_mainchain_atoms,
        n_mainchain_atoms=n_mainchain_atoms,
        mainchain_atoms=mainchain_atom_indices_padded,
        is_real_mainchain_atom=is_real_mainchain_atom,
    )
    setattr(packed_block_types, "mainchain_atom_indices", mcatominds)


def create_mainchain_coordinate_constraints(pose_stack: PoseStack) -> PoseStack:
    pbt = pose_stack.packed_block_types
    _annotate_mainchain_atom_indices(pbt)
    mc_at_inds = pbt.mainchain_atom_indices
    constraint_set = pose_stack.constraint_set

    is_real_bt = pose_stack.block_type_ind64 >= 0
    real_bt = pose_stack.block_type_ind64[is_real_bt]
    # n_mainchain_ats_for_bt = pbt.mainchain_atom_indices.n_mainchain_atoms[real_bt]
    block_ind_for_mc_atom = (
        torch.arange(pose_stack.max_n_blocks, device=pbt.device)
        .unsqueeze(0)
        .unsqueeze(-1)
        .expand(pose_stack.n_poses, -1, mc_at_inds.max_n_mainchain_atoms)
    )
    block_ind_for_mc_atom_for_real_block = block_ind_for_mc_atom[is_real_bt]
    pose_ind_for_mc_atom = (
        torch.arange(pose_stack.n_poses, device=pbt.device)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .expand(-1, pose_stack.max_n_blocks, mc_at_inds.max_n_mainchain_atoms)
    )
    pose_ind_for_mc_atom_for_real_block = pose_ind_for_mc_atom[is_real_bt]
    is_real_mainchain_at = mc_at_inds.is_real_mainchain_atom[real_bt]
    atom_ind_for_mc_atom = mc_at_inds.mainchain_atoms[real_bt]

    block_for_real_mc_at = block_ind_for_mc_atom_for_real_block[is_real_mainchain_at]
    pose_ind_for_real_mc_at = pose_ind_for_mc_atom_for_real_block[is_real_mainchain_at]
    local_atom_ind_for_real_mc_at = atom_ind_for_mc_atom[is_real_mainchain_at]

    n_mc_ats = block_for_real_mc_at.shape[0]

    atom_offset_for_real_mc_at = pose_stack.block_coord_offset64[
        pose_ind_for_real_mc_at, block_for_real_mc_at
    ]
    atom_ind_for_real_mc_at = local_atom_ind_for_real_mc_at + atom_offset_for_real_mc_at

    # cnstr_atoms = torch.full((n_mc_ats, 1, 3), 0, dtype=torch.int32, device=pose_stack.device)
    cnstr_params = torch.full(
        (n_mc_ats, 5), 0, dtype=torch.float32, device=pose_stack.device
    )

    print("pose_ind_for_real_mc_at", pose_ind_for_real_mc_at.shape)
    print("block_for_real_mc_at", block_for_real_mc_at.shape)
    print("local_atom_ind_for_real_mc_at", local_atom_ind_for_real_mc_at.shape)
    cnstr_atoms = torch.stack(
        [pose_ind_for_real_mc_at, block_for_real_mc_at, local_atom_ind_for_real_mc_at],
        dim=-1,
    ).unsqueeze(1)
    print("constr_atoms", cnstr_atoms.shape)
    cnstr_params[:, 1:4] = pose_stack.coords[
        pose_ind_for_real_mc_at, atom_ind_for_real_mc_at
    ]
    # Set standard deviation to 0.5A to match R3 FastRelax
    cnstr_params[:, 4] = 0.5

    if constraint_set is None:
        constraint_set = ConstraintSet.create_empty(
            pose_stack.device, pose_stack.n_poses
        )
    constraint_set = constraint_set.add_constraints(
        ConstraintEnergyTerm.harmonic_coordinate, cnstr_atoms, cnstr_params
    )

    return attrs.evolve(
        pose_stack,
        constraint_set=constraint_set,
    )
