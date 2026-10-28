"""Select a subset of blocks of a PoseStack into a compact PoseStack."""

from typing import Sequence

import attr
import numpy
import torch

from tmol.chemical._constants import MAX_SIG_BOND_SEPARATION

from ._inter_block_bondsep import InterBlockBondsep
from ._pdb_info import PDBInfo
from ._pose_stack import PoseStack
from ._constraint_set import ConstraintSet


def _select_inter_block_bondsep(
    pose_stack: PoseStack, blocks: torch.Tensor
) -> InterBlockBondsep:
    """Restrict the sparse block-pair bond separations to ``blocks``."""
    dense = pose_stack.inter_block_bondsep.to_dense()[
        :, blocks[:, None], blocks[None, :]
    ]
    n_poses, n_blocks, _, n_conn, _ = dense.shape
    if dense.numel() == 0:
        return InterBlockBondsep.empty(n_poses, n_blocks, n_conn, pose_stack.device, 1)
    minimum = dense.amin(dim=(-2, -1))
    nearby = minimum < MAX_SIG_BOND_SEPARATION
    slots = nearby.cumsum(dim=2) - 1
    result = InterBlockBondsep.empty(
        n_poses,
        n_blocks,
        n_conn,
        pose_stack.device,
        int(nearby.sum(dim=2).max()) + 1,
    )
    pose, block1, block2 = torch.nonzero(nearby, as_tuple=True)
    slot = slots[pose, block1, block2]
    result.near_blocks[pose, block1, slot, 0] = block2.to(torch.int32)
    result.near_blocks[pose, block1, slot, 1] = minimum[pose, block1, block2].to(
        torch.int32
    )
    result.bondsep[pose, block1, slot] = dense[pose, block1, block2]
    return result


def select_blocks(pose_stack: PoseStack, blocks: Sequence[int]) -> PoseStack:
    """Compact PoseStack holding ``blocks`` (in the given order) of every pose.

    All poses must share one block layout. Connections to omitted blocks are
    cleared, and connections between retained blocks are remapped. Chain ids,
    PDB metadata and bond separations are carried over; constraints and split
    block mappings are dropped.
    """

    if pose_stack.split_block_mapping:
        raise ValueError("selecting blocks does not support split-block metadata")
    device = pose_stack.device
    keep = torch.as_tensor(list(blocks), dtype=torch.int64, device=device)
    block_types = pose_stack.block_type_ind64
    if keep.numel() == 0 or bool(torch.any(block_types[:, keep] < 0)):
        raise ValueError("blocks must be a non-empty set present in every pose")
    selected_types = block_types[:, keep]
    if not bool(torch.all(selected_types == selected_types[0:1])):
        raise ValueError("selecting blocks requires a homogeneous block layout")

    old_to_new = torch.full(
        (pose_stack.max_n_blocks,), -1, dtype=torch.int64, device=device
    )
    old_to_new[keep] = torch.arange(keep.numel(), dtype=torch.int64, device=device)
    connections = pose_stack.inter_residue_connections64[:, keep].clone()
    partners = connections[..., 0]
    real = partners >= 0
    mapped = torch.full_like(partners, -1)
    mapped[real] = old_to_new[partners[real]]
    connections[real & (mapped < 0)] = -1
    partners[mapped >= 0] = mapped[mapped >= 0]

    new_types = block_types[:, keep].clone()
    pbt = pose_stack.packed_block_types
    n_atoms = pbt.n_atoms[new_types[0]].to(torch.int64)
    offsets = torch.zeros_like(new_types)
    offsets[:, 1:] = torch.cumsum(n_atoms[:-1], dim=0)[None, :]
    source_atoms = torch.cat(
        [
            pose_stack.block_coord_offset64[0, b]
            + torch.arange(int(n), dtype=torch.int64, device=device)
            for b, n in zip(keep, n_atoms)
        ]
    )

    keep_np, atoms_np = keep.cpu().numpy(), source_atoms.cpu().numpy()
    pdb = pose_stack.pdb_info
    by_residue = {
        name: getattr(pdb, name)[:, keep_np].copy()
        for name in (
            "residue_labels",
            "residue_insertion_codes",
            "chain_labels",
            "metal_origins",
            "residue_annotations",
        )
        if getattr(pdb, name) is not None
    }
    pdb_info = PDBInfo(
        atom_occupancy=pdb.atom_occupancy[:, atoms_np].copy(),
        atom_b_factor=pdb.atom_b_factor[:, atoms_np].copy(),
        **by_residue,
    )
    chains = pose_stack.chain_id64[:, keep].clone()
    return PoseStack(
        packed_block_types=pbt,
        coords=pose_stack.coords[:, source_atoms].clone(),
        block_coord_offset=offsets.to(torch.int32),
        block_coord_offset64=offsets,
        inter_residue_connections=connections.to(torch.int32),
        inter_residue_connections64=connections,
        inter_block_bondsep=_select_inter_block_bondsep(pose_stack, keep),
        block_type_ind=new_types.to(torch.int32),
        block_type_ind64=new_types,
        chain_id=chains.to(torch.int32),
        chain_id64=chains,
        pdb_info=pdb_info,
        constraint_set=None,
        device=device,
    )


def repeat_pose_stack(pose_stack: PoseStack, n_poses: int) -> PoseStack:
    """A PoseStack of ``n_poses`` independent copies of a single pose.

    Unlike :meth:`PoseStackBuilder.from_poses` this keeps the coordinate dtype
    and the constraint set.
    """
    if pose_stack.n_poses != 1:
        raise ValueError("only a single reference pose can be replicated")
    if pose_stack.split_block_mapping:
        raise ValueError("replicating split-block metadata is not supported")

    def tile(value: torch.Tensor) -> torch.Tensor:
        return value.repeat((n_poses,) + (1,) * (value.ndim - 1))

    constraints = ConstraintSet.concatenate(
        [pose_stack.constraint_set] * n_poses,
        from_multiple_pose_stacks=True,
        n_poses=n_poses,
        ps_offset=torch.arange(n_poses, dtype=torch.int64),
    )
    pdb_info = PDBInfo(
        **{
            name: None if value is None else numpy.repeat(value, n_poses, axis=0)
            for name, value in attr.asdict(pose_stack.pdb_info, recurse=False).items()
        }
    )
    tensors = (
        "coords",
        "block_coord_offset",
        "block_coord_offset64",
        "inter_residue_connections",
        "inter_residue_connections64",
        "block_type_ind",
        "block_type_ind64",
        "chain_id",
        "chain_id64",
    )
    bondsep = pose_stack.inter_block_bondsep
    return attr.evolve(
        pose_stack,
        **{name: tile(getattr(pose_stack, name)) for name in tensors},
        inter_block_bondsep=InterBlockBondsep.concatenate(
            [bondsep] * n_poses, bondsep.max_n_blocks, bondsep.max_n_conn
        ),
        pdb_info=pdb_info,
        constraint_set=constraints,
    )
