"""Read-only queries on a PoseStack: sequence and named torsions.

Setting torsions (which requires inverse kinematics) lives in
``tmol.kinematics._pose_stack_kinematics`` so that this module carries no
kinematics dependency.
"""

from typing import List

import numpy
import torch

from tmol.chemical import three2one
from tmol.numeric import coord_dihedrals
from tmol.pose import PoseStack


def get_sequence(pose_stack: PoseStack, pose: int = 0) -> str:
    """One-letter sequence of one pose's real (non-padding) blocks.

    Blocks whose three-letter code has no one-letter equivalent (ligands,
    ions, and other non-canonical residues) appear as ``"X"``.
    """
    pbt = pose_stack.packed_block_types
    real = pose_stack.block_type_ind64[pose] >= 0
    type_inds = pose_stack.block_type_ind64[pose, real].tolist()
    block_types = [pbt.active_block_types[i] for i in type_inds]
    return "".join(three2one(bt.name3) or "X" for bt in block_types)


def get_ligand_mask(pose_stack: PoseStack) -> torch.Tensor:
    """Boolean mask shaped ``[n_poses, max_n_blocks]``, true where a block is a
    ligand (or any other non-polymer residue, such as an ion). Padding blocks
    are false.
    """
    pbt = pose_stack.packed_block_types
    is_polymer_by_type = torch.tensor(
        [bt.properties.polymer.is_polymer for bt in pbt.active_block_types],
        dtype=torch.bool,
        device=pose_stack.device,
    )
    block_type_ind64 = pose_stack.block_type_ind64
    real = block_type_ind64 >= 0
    safe_ind = torch.where(real, block_type_ind64, torch.zeros_like(block_type_ind64))
    return real & ~is_polymer_by_type[safe_ind]


def get_polymer_type_mask(pose_stack: PoseStack, polymer_type) -> torch.Tensor:
    """Boolean mask shaped ``[n_poses, max_n_blocks]``, true where a block's
    ``properties.polymer.polymer_type`` equals ``polymer_type`` (one of
    ``"amino_acid"``, ``"nucleic_acid"``, or ``None`` for a non-polymer like a
    ligand or ion). Padding blocks are false.
    """
    pbt = pose_stack.packed_block_types
    matches_by_type = torch.tensor(
        [
            bt.properties.polymer.polymer_type == polymer_type
            for bt in pbt.active_block_types
        ],
        dtype=torch.bool,
        device=pose_stack.device,
    )
    block_type_ind64 = pose_stack.block_type_ind64
    real = block_type_ind64 >= 0
    safe_ind = torch.where(real, block_type_ind64, torch.zeros_like(block_type_ind64))
    return real & matches_by_type[safe_ind]


def metal_blocks(pose_stack: PoseStack, pose: int) -> List[int]:
    """Indices of the blocks in one pose that carry metal sites."""
    return [
        block
        for block in range(pose_stack.max_n_blocks)
        if int(pose_stack.block_type_ind64[pose, block]) >= 0
        and pose_stack.block_type(pose, block).metal_sites
    ]


def residue_label(pose_stack: PoseStack, pose: int, block: int) -> str:
    """ "chain resnum name3" for one block."""
    info = pose_stack.pdb_info
    name3 = pose_stack.block_type(pose, block).name3
    return f"{info.chain_labels[pose, block]} {int(info.residue_labels[pose, block])} {name3}"


def atom_label(pose_stack: PoseStack, pose: int, atom: int) -> str:
    """ "chain resnum name3 atom" for a per-pose atom index."""
    real = pose_stack.block_type_ind64[pose] >= 0
    offsets = pose_stack.block_coord_offset64[pose]
    block = int(((offsets <= atom) & real).nonzero()[-1])
    local = atom - int(offsets[block])
    name = pose_stack.block_type(pose, block).atoms[local].name
    return f"{residue_label(pose_stack, pose, block)} {name}"


def block_for_residue_label(
    pose_stack: PoseStack, pose_index: int, chain: str, residue_number: int
) -> int:
    """Block index of one (chain, author residue number) label.

    Raises ValueError if no block in the pose matches.
    """
    chain_labels = pose_stack.pdb_info.chain_labels[pose_index]
    residue_labels = pose_stack.pdb_info.residue_labels[pose_index]
    matches = numpy.flatnonzero(
        (chain_labels == chain) & (residue_labels == residue_number)
    )
    if len(matches) == 0:
        raise ValueError(
            f"no block found for chain {chain!r} residue {residue_number} "
            f"in pose {pose_index}"
        )
    return int(matches[0])


def blocks_for_residue_labels(
    pose_stack: PoseStack,
    pose_index: int,
    chain: str,
    residue_numbers: torch.Tensor,
) -> torch.Tensor:
    """Block index of each (chain, author residue number) label in a batch.

    ``residue_numbers`` may have any shape; the result has the same shape.
    Entries with no matching block resolve to ``-1``.
    """
    chain_labels = pose_stack.pdb_info.chain_labels[pose_index]
    residue_labels = pose_stack.pdb_info.residue_labels[pose_index]
    chain_mask = chain_labels == chain

    queries = residue_numbers.cpu().numpy()
    matches = (residue_labels == queries[..., None]) & chain_mask
    block = numpy.where(matches.any(axis=-1), matches.argmax(axis=-1), -1)
    return torch.as_tensor(block, dtype=torch.int64, device=pose_stack.device)


def get_torsion_names(pose_stack: PoseStack, pose: int, block: int) -> List[str]:
    """Names of the torsions defined on a block's type, in database order."""
    bt_ind = int(pose_stack.block_type_ind64[pose, block])
    if bt_ind == -1:
        raise ValueError(f"block {block} of pose {pose} is not a real block")
    bt = pose_stack.packed_block_types.active_block_types[bt_ind]
    return [tor.name for tor in bt.torsions]


def get_named_torsions(
    pose_stack: PoseStack,
    poses=None,
    blocks=None,
    names=None,
    degrees: bool = True,
):
    """Measure named torsions.

    Returns a single float when ``poses``, ``blocks`` and ``names`` are all
    scalars, a {name: value} dict when only ``names`` is left open, and
    otherwise a list-of-lists of such dicts indexed [pose][block]. Torsions that
    reach an absent neighboring residue measure as nan.
    """
    requests = _torsion_requests(pose_stack, poses, blocks, names)
    values = _measure_torsions(pose_stack, requests, degrees)

    if _is_scalar(poses) and _is_scalar(blocks):
        if _is_scalar(names):
            return float(values[0])
        return {
            name: value for (_, _, _, name), value in zip(requests, values.tolist())
        }

    out = [
        [{} for _ in range(pose_stack.max_n_blocks)] for _ in range(pose_stack.n_poses)
    ]
    for (pose, block, _, name), value in zip(requests, values.tolist()):
        out[pose][block][name] = value
    return out


def _is_scalar(x):
    return isinstance(x, (int, numpy.integer, str))


def _as_list(x, default):
    if x is None:
        return list(default)
    return [x] if _is_scalar(x) else list(x)


def _torsion_requests(pose_stack, poses, blocks, names):  # noqa: C901
    """Expand a selection into (pose, block, torsion index, name) tuples."""
    pbt = pose_stack.packed_block_types
    bti = pose_stack.block_type_ind64

    pose_list = _as_list(poses, range(pose_stack.n_poses))
    block_list = _as_list(blocks, range(pose_stack.max_n_blocks))
    name_list = None if names is None else _as_list(names, ())

    paired = poses is not None and blocks is not None
    if paired and not _is_scalar(poses) and not _is_scalar(blocks):
        if len(pose_list) != len(block_list):
            raise ValueError("poses and blocks must be the same length")
        pairs = list(zip(pose_list, block_list))
    else:
        pairs = [(p, b) for p in pose_list for b in block_list]

    if name_list is not None and len(name_list) == len(pairs) and len(pairs) > 1:
        paired_names = name_list
    else:
        paired_names = None

    requests = []
    for i, (pose, block) in enumerate(pairs):
        bt_ind = int(bti[pose, block])
        if bt_ind == -1:
            if blocks is None:
                continue
            raise ValueError(f"block {block} of pose {pose} is not a real block")
        bt = pbt.active_block_types[bt_ind]
        tor_inds = {tor.name: j for j, tor in enumerate(bt.torsions)}
        if paired_names is not None:
            wanted = [paired_names[i]]
        elif name_list is not None:
            wanted = name_list
        else:
            wanted = list(tor_inds)
        for name in wanted:
            if name not in tor_inds:
                if name_list is None or blocks is None:
                    continue
                raise ValueError(f"block type {bt.name} has no torsion {name!r}")
            requests.append((pose, block, tor_inds[name], name))
    return requests


def _resolve_uaid(pose_stack, pose, block, uaid):
    """Flat (pose, atom) coordinate index for a uaid, or -1 if unresolvable."""
    atom, conn, sep = int(uaid[0]), int(uaid[1]), int(uaid[2])
    pbt = pose_stack.packed_block_types
    if conn == -1:
        if atom == -1:
            return -1
        target_block, local = block, atom
    else:
        partner = pose_stack.inter_residue_connections64[pose, block, conn]
        target_block, target_conn = int(partner[0]), int(partner[1])
        if target_block == -1:
            return -1
        bt_ind = int(pose_stack.block_type_ind64[pose, target_block])
        local = int(pbt.atom_downstream_of_conn[bt_ind, target_conn, sep])
        if local == -1:
            return -1
    offset = int(pose_stack.block_coord_offset64[pose, target_block])
    return pose * pose_stack.max_n_pose_atoms + offset + local


def _measure_torsions(pose_stack, requests, degrees):
    pbt = pose_stack.packed_block_types
    inds = numpy.full((len(requests), 4), -1, dtype=numpy.int64)
    for i, (pose, block, tor_ind, _) in enumerate(requests):
        bt_ind = int(pose_stack.block_type_ind64[pose, block])
        uaids = pbt.active_block_types[bt_ind].ordered_torsions[tor_ind]
        for j in range(4):
            inds[i, j] = _resolve_uaid(pose_stack, pose, block, uaids[j])

    values = numpy.full(len(requests), numpy.nan)
    ok = (inds != -1).all(axis=1)
    if ok.any():
        flat = pose_stack.coords.reshape(-1, 3).double()
        sel = torch.from_numpy(inds[ok]).to(pose_stack.device)
        measured = coord_dihedrals(
            flat[sel[:, 0]], flat[sel[:, 1]], flat[sel[:, 2]], flat[sel[:, 3]]
        )
        values[ok] = measured.cpu().numpy().astype(numpy.float64)
    return numpy.degrees(values) if degrees else values
