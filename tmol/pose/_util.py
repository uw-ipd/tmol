"""Measuring named torsions on a PoseStack.

Setting torsions (which requires inverse kinematics) lives in
``tmol.kinematics._pose_stack_kinematics`` so that this module carries no
kinematics dependency.
"""

from typing import List

import numpy
import torch

from tmol.numeric import coord_dihedrals
from tmol.pose import PoseStack


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
