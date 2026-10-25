"""Selection of the movable ligand/sidechain partition of a docking complex.

The partition is a compact PoseStack: the ligand is block 0, followed by the
flexible sidechains and their immediate peptide neighbors in source order.
"""

from typing import NamedTuple

import attr
import torch

from tmol.pose import PoseStack, select_blocks


def has_nonpolymer_connection(
    pose_stack: PoseStack,
    block: int,
    block_type_index: int,
) -> bool:
    """Return True if the block has a realized inter-residue connection that is
    not one of its polymer (down/up) connections -- e.g. a disulfide.

    Flexible-sidechain docking moves chi through the fold forest, which ignores
    non-polymer covalent bonds (``tmol/kinematics/fold_forest.py``), and the chi
    controller is built only from the block's intra-block bonds. A disulfide sulfur
    would therefore rotate independently of its partner, stretching the S-S bond,
    and the compact-pose partition would either drop the beta2016 disulfide term or
    still fail to constrain the bond. Such residues must not be freed.
    """

    pbt = pose_stack.packed_block_types
    n_conn = int(pbt.n_conn[block_type_index])
    if n_conn == 0:
        return False
    down_conn = int(pbt.down_conn_inds[block_type_index])
    up_conn = int(pbt.up_conn_inds[block_type_index])
    partners = pose_stack.inter_residue_connections64[:, block, :n_conn, 0]
    for conn in range(n_conn):
        if conn == down_conn or conn == up_conn:
            continue
        if bool(torch.any(partners[:, conn] >= 0)):
            return True
    return False


def resolve_ligand_block(pose_stack: PoseStack, ligand: str | int) -> int:
    """Block index of a ligand given by residue name (``"ATP"``) or block index."""
    types = pose_stack.packed_block_types.active_block_types
    names = [
        "" if index < 0 else types[index].name3
        for index in pose_stack.block_type_ind64[0].tolist()
    ]
    if isinstance(ligand, str):
        matches = [block for block, name in enumerate(names) if name == ligand]
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one ligand block named {ligand!r}, "
                f"found {len(matches)}"
            )
        return matches[0]
    if not 0 <= ligand < len(names) or not names[ligand]:
        raise ValueError(f"ligand block {ligand} is not present in the pose")
    return ligand


def validated_sidechain_blocks(
    pose_stack: PoseStack,
    ligand_block: int,
    sidechain_blocks,
) -> tuple[int, ...]:
    """Normalize and validate a deterministic set of polymer blocks."""

    if isinstance(sidechain_blocks, torch.Tensor):
        values = sidechain_blocks.detach().cpu().tolist()
    else:
        try:
            values = list(sidechain_blocks)
        except TypeError as exc:
            raise TypeError("sidechain_blocks must be an iterable of integers") from exc

    blocks = []
    for block in values:
        if block < 0 or block >= pose_stack.max_n_blocks:
            raise ValueError(f"sidechain block {block} is out of range")
        if block == ligand_block:
            raise ValueError("the ligand block cannot be a sidechain block")
        blocks.append(block)
    if len(set(blocks)) != len(blocks):
        raise ValueError("sidechain_blocks must not contain duplicates")

    normalized = tuple(sorted(blocks))
    for block in normalized:
        block_type_indices = pose_stack.block_type_ind64[:, block]
        if bool(torch.any(block_type_indices < 0)):
            raise ValueError(
                f"sidechain block {block} is absent from at least one pose"
            )
        if bool(torch.any(block_type_indices != block_type_indices[0])):
            raise ValueError(
                f"all poses must use the same type at sidechain block {block}"
            )
        block_type = pose_stack.packed_block_types.active_block_types[
            int(block_type_indices[0])
        ]
        if not block_type.properties.polymer.is_polymer:
            raise ValueError(
                f"sidechain block {block} ({block_type.name}) is not polymeric"
            )
        if has_nonpolymer_connection(pose_stack, block, int(block_type_indices[0])):
            raise ValueError(
                f"sidechain block {block} ({block_type.name}) has a non-polymer "
                "covalent connection (e.g. a disulfide); flexible-docking kinematics "
                "cannot preserve that bond, so such residues must not be selected as "
                "flexible"
            )
        if int(block_type.n_sc_torsions) == 0:
            raise ValueError(
                f"sidechain block {block} ({block_type.name}) has no sidechain chi "
                "torsions and cannot be made flexible (nothing to move); do not "
                "select chi-less residues such as ALA or GLY"
            )
    return normalized


def sidechain_context_blocks(
    pose_stack: PoseStack,
    ligand_block: int,
    sidechain_blocks=(),
) -> tuple[int, ...]:
    """Return unique immediate polymer neighbors of selected sidechains.

    Only the residue type's declared lower/upper polymer connections are
    followed; non-polymeric connections such as disulfides are not peptide
    context. Selected sidechains and the ligand are never returned.
    """

    sidechain_blocks = validated_sidechain_blocks(
        pose_stack, ligand_block, sidechain_blocks
    )
    if not sidechain_blocks:
        return ()

    pbt = pose_stack.packed_block_types
    excluded = {ligand_block, *sidechain_blocks}
    context: set[int] = set()
    for block in sidechain_blocks:
        block_type_index = int(pose_stack.block_type_ind64[0, block])
        connection_indices = (
            int(pbt.down_conn_inds[block_type_index]),
            int(pbt.up_conn_inds[block_type_index]),
        )
        for connection in connection_indices:
            if connection < 0:
                continue
            partners = pose_stack.inter_residue_connections64[:, block, connection, 0]
            partner_connections = pose_stack.inter_residue_connections64[
                :, block, connection, 1
            ]
            if bool(torch.any(partners != partners[0])):
                raise ValueError(
                    "movable grid docking requires identical peptide context "
                    "across poses"
                )
            partner = int(partners[0])
            if partner < 0 or partner in excluded:
                continue
            partner_type_indices = pose_stack.block_type_ind64[:, partner]
            if bool(torch.any(partner_type_indices < 0)):
                raise ValueError(
                    "a sidechain peptide neighbor is absent from at least one pose"
                )
            if bool(torch.any(partner_type_indices != partner_type_indices[0])):
                raise ValueError(
                    "all poses must use the same peptide-neighbor block type"
                )
            partner_type = pbt.active_block_types[int(partner_type_indices[0])]
            if not partner_type.properties.polymer.is_polymer:
                raise ValueError(
                    "a declared polymer connection points to a non-polymer block"
                )
            expected_partner_connection = (
                partner_type.up_connection_ind
                if connection == int(pbt.down_conn_inds[block_type_index])
                else partner_type.down_connection_ind
            )
            if bool(torch.any(partner_connections != expected_partner_connection)):
                raise ValueError(
                    "a declared polymer connection is not reciprocally peptide-linked"
                )
            context.add(partner)
    return tuple(sorted(context))


def movable_block_mapping(
    pose_stack: PoseStack,
    ligand_block: int,
    sidechain_blocks=(),
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Return compact source order and flexible/context compact indices."""

    sidechain_blocks = validated_sidechain_blocks(
        pose_stack, ligand_block, sidechain_blocks
    )
    context_blocks = sidechain_context_blocks(
        pose_stack, ligand_block, sidechain_blocks
    )
    receptor_blocks = tuple(sorted(sidechain_blocks + context_blocks))
    source_blocks = (ligand_block,) + receptor_blocks
    source_to_compact = {
        source_block: compact_block
        for compact_block, source_block in enumerate(source_blocks)
    }
    compact_sidechains = tuple(
        source_to_compact[source_block] for source_block in sidechain_blocks
    )
    compact_context = tuple(
        source_to_compact[source_block] for source_block in context_blocks
    )
    return source_blocks, compact_sidechains, compact_context


def movable_pose_stack(
    pose_stack: PoseStack, ligand_block: int, sidechain_blocks=()
) -> PoseStack:
    """Compact PoseStack of the ligand (block 0), sidechains and peptide context.

    The disconnected ligand receives its own chain id, so it is always an
    independent kinematic root. With no sidechains this is the ligand alone.
    """

    sidechain_blocks = validated_sidechain_blocks(
        pose_stack, ligand_block, sidechain_blocks
    )
    if bool(
        torch.any(pose_stack.inter_residue_connections64[:, ligand_block, :, 0] >= 0)
    ):
        raise ValueError("grid docking requires a disconnected ligand block")
    source_blocks, _, _ = movable_block_mapping(
        pose_stack, ligand_block, sidechain_blocks
    )
    compact = select_blocks(pose_stack, source_blocks)
    chains = compact.chain_id64.clone()
    # One past the retained receptor maximum cannot collide with any chain.
    chains[:, 0] = chains[:, 1:].amax(dim=1) + 1 if chains.shape[1] > 1 else 0
    return attr.evolve(compact, chain_id=chains.to(torch.int32), chain_id64=chains)


class MovableAtoms(NamedTuple):
    """Where the moving blocks' atoms sit in the compact docking pose."""

    source_blocks: tuple[int, ...]
    compact_offsets: tuple[int, ...]
    atom_map: torch.Tensor  # [pose atom] -> compact atom index, -1 for fixed atoms
    n_atoms: int


def movable_atoms(
    pose_stack: PoseStack, ligand_block: int, sidechain_blocks=()
) -> MovableAtoms:
    """Compact-pose index of every atom of the ligand, sidechains and context."""

    source_blocks, _, _ = movable_block_mapping(
        pose_stack, ligand_block, sidechain_blocks
    )
    compact = movable_pose_stack(pose_stack, ligand_block, sidechain_blocks)
    atom_map = torch.full((pose_stack.coords.shape[1],), -1, dtype=torch.int64)
    n_block_atoms = pose_stack.packed_block_types.n_atoms.to(torch.int64)
    offsets = []
    for compact_block, block in enumerate(source_blocks):
        offset = int(compact.block_coord_offset64[0, compact_block])
        n_atoms = int(n_block_atoms[pose_stack.block_type_ind64[0, block]])
        start = int(pose_stack.block_coord_offset64[0, block])
        atom_map[start : start + n_atoms] = torch.arange(n_atoms) + offset
        offsets.append(offset)
    return MovableAtoms(
        source_blocks, tuple(offsets), atom_map, compact.max_n_pose_atoms
    )


def pose_atom_table(pose_stack: PoseStack) -> tuple[torch.Tensor, ...]:
    """``(atom, block, block type, block-local atom)`` of each real atom of pose 0."""

    block_types = pose_stack.block_type_ind64[0]
    blocks = torch.nonzero(block_types >= 0).squeeze(1)
    counts = pose_stack.packed_block_types.n_atoms.to(torch.int64)[block_types[blocks]]
    block = blocks.repeat_interleave(counts)
    first = torch.cumsum(counts, 0) - counts
    local = torch.arange(int(counts.sum())) - first.repeat_interleave(counts)
    atom = pose_stack.block_coord_offset64[0, block] + local
    return atom, block, block_types[block], local
