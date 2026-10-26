import torch
import attrs

from tmol.types import Tensor
from tmol.pose._packed_block_types import PackedBlockTypes
from tmol.pose._pose_stack import PoseStack
from tmol.score._annotation_cache import (
    AnnotationKey,
    cached_annotation,
    store_annotation,
)


def constrain_closest_interchain_ca_pair(
    pose_stack: PoseStack, chain_a: str, chain_b: str, *, std_dev: float = 0.5
) -> PoseStack:
    """Restrain the closest CA-CA pair between two chains to its current distance.

    For each pose independently, finds the residue of ``chain_a`` and the
    residue of ``chain_b`` whose CA atoms are closest together, and adds a
    harmonic distance constraint restraining that pair to its current
    distance -- a common way to hold a protein-protein interface together
    while the rest of a structure is sampled or minimized.

    Raises ValueError for any pose where either chain has no residue with a
    CA atom.
    """
    from tmol.ops import get_chain_mask
    from tmol.pose._constraint_set import ConstraintSet
    from tmol.score.constraint._constraint_energy_term import ConstraintEnergyTerm

    constraint_set = pose_stack.constraint_set
    device = pose_stack.device

    chain_mask_a = get_chain_mask(pose_stack, chain_a)
    chain_mask_b = get_chain_mask(pose_stack, chain_b)

    cnstr_atoms = torch.zeros(
        (pose_stack.n_poses, 2, 3), dtype=torch.int32, device=device
    )
    cnstr_params = torch.zeros(
        (pose_stack.n_poses, 2), dtype=torch.float32, device=device
    )

    for pose_ind in range(pose_stack.n_poses):

        def blocks_with_ca(chain_mask):
            return [
                block
                for block in torch.nonzero(chain_mask[pose_ind]).flatten().tolist()
                if "CA" in pose_stack.block_type(pose_ind, block).atom_to_idx
            ]

        blocks_a = blocks_with_ca(chain_mask_a)
        blocks_b = blocks_with_ca(chain_mask_b)
        if not blocks_a or not blocks_b:
            raise ValueError(
                f"pose {pose_ind}: chain {chain_a!r} or {chain_b!r} has no "
                "residue with a CA atom"
            )

        def ca_coord(block):
            block_type = pose_stack.block_type(pose_ind, block)
            offset = int(pose_stack.block_coord_offset[pose_ind, block])
            return pose_stack.coords[pose_ind, offset + block_type.atom_to_idx["CA"]]

        ca_coords_a = torch.stack([ca_coord(block) for block in blocks_a])
        ca_coords_b = torch.stack([ca_coord(block) for block in blocks_b])
        pairwise_distance = torch.cdist(ca_coords_a, ca_coords_b)
        closest = int(torch.argmin(pairwise_distance))
        closest_a = blocks_a[closest // len(blocks_b)]
        closest_b = blocks_b[closest % len(blocks_b)]

        type_a = pose_stack.block_type(pose_ind, closest_a)
        type_b = pose_stack.block_type(pose_ind, closest_b)
        cnstr_atoms[pose_ind, 0] = torch.tensor(
            [pose_ind, closest_a, type_a.atom_to_idx["CA"]]
        )
        cnstr_atoms[pose_ind, 1] = torch.tensor(
            [pose_ind, closest_b, type_b.atom_to_idx["CA"]]
        )
        cnstr_params[pose_ind, 0] = float(pairwise_distance.min())
        cnstr_params[pose_ind, 1] = std_dev

    if constraint_set is None:
        constraint_set = ConstraintSet.create_empty(device, pose_stack.n_poses)
    constraint_set = constraint_set.add_constraints(
        ConstraintEnergyTerm.harmonic, cnstr_atoms, cnstr_params
    )

    return attrs.evolve(pose_stack, constraint_set=constraint_set)


def constrain_all_ca(pose_stack: PoseStack) -> PoseStack:
    """Add coordinate constraints at every available protein CA atom."""
    from tmol.pose._constraint_set import ConstraintSet
    from tmol.score.constraint._constraint_energy_term import ConstraintEnergyTerm

    constraint_set = pose_stack.constraint_set

    cnstr_atoms = torch.full((0, 1, 3), 0, dtype=torch.int32, device=pose_stack.device)
    cnstr_params = torch.full((0, 5), 0, dtype=torch.float32, device=pose_stack.device)

    for pose_ind in range(pose_stack.n_poses):
        for block_ind in range(pose_stack.max_n_blocks):
            if pose_stack.is_real_block(pose_ind, block_ind):
                block_type = pose_stack.block_type(pose_ind, block_ind)
                if "CA" not in block_type.atom_to_idx:
                    continue

                ca_ind = block_type.atom_to_idx["CA"]
                ca_params = torch.full(
                    (1, 5), 0.0, dtype=torch.float32, device=pose_stack.device
                )
                ca_params[0, 1:4] = pose_stack.coords[pose_ind][
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
                cnstr_params = torch.cat([cnstr_params, ca_params])
    cnstr_params[:, 4] = 0.5  # Set standard deviation to 0.5A to match R3 FastRelax
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


@attrs.define(frozen=True)
class MCAtomIndices:
    """Padded main-chain atom indices for every packed block type."""

    max_n_mainchain_atoms: int
    n_mainchain_atoms: Tensor[torch.int64][:]
    mainchain_atoms: Tensor[torch.int64][:, :]
    is_real_mainchain_atom: Tensor[torch.bool][:, :]


def _annotate_mainchain_atom_indices(
    packed_block_types: PackedBlockTypes,
) -> MCAtomIndices:
    """Get the list of mainchain atoms for each block type and annotate their indices."""
    key = AnnotationKey.from_sources(settings=(packed_block_types.device,))
    cached = cached_annotation(
        packed_block_types, "_mainchain_atom_indices_annotation", key
    )
    if cached is not None:
        return cached
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
    return store_annotation(
        packed_block_types,
        "_mainchain_atom_indices_annotation",
        key,
        mcatominds,
        fields=("mainchain_atom_indices",),
    )


def create_mainchain_coordinate_constraints(pose_stack: PoseStack) -> PoseStack:
    """Add coordinate constraints for every declared polymer main-chain atom."""
    from tmol.pose._constraint_set import ConstraintSet
    from tmol.score.constraint._constraint_energy_term import ConstraintEnergyTerm

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

    cnstr_atoms = torch.stack(
        [pose_ind_for_real_mc_at, block_for_real_mc_at, local_atom_ind_for_real_mc_at],
        dim=-1,
    ).unsqueeze(1)
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
