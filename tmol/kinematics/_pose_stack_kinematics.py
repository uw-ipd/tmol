"""Kinematic operations on a PoseStack: DOF setting and torsion manipulation.

Functions here sit above both tmol.pose (which holds the PoseStack data
structure) and tmol.kinematics (which holds the KinForest and inverse-kin
machinery).  They are placed in tmol.kinematics rather than tmol.pose so that
tmol.pose itself carries no kinematics dependency.
"""

from __future__ import annotations

import attr
import numpy as np
import torch

from tmol.kinematics.compiled import inverse_kin
from tmol.kinematics import (
    BondDOFTypes,
    NodeType,
    FoldForest,
    PoseStackKinematicsModule,
)
from tmol.pose._pose_stack import PoseStack
from tmol.pose._util import _measure_torsions, _resolve_uaid, _torsion_requests


def _controlling_kfo_node(pose_stack, kmd, kfo_for_atom, pose, block, tor_ind):
    """Kinforest node whose phi_c controls a named torsion, or -1 if unresolvable.

    The torsion tracks that phi_c with slope +1: phi_c turns about
    parent→child, so the reversed axis and the reversed end of the dihedral
    cancel.
    """
    pbt = pose_stack.packed_block_types
    gssps = pbt.gen_seg_scan_path_segs
    bt_ind = int(pose_stack.block_type_ind64[pose, block])
    in_conn = int(kmd.block_in_and_first_out[pose, block, 0])
    uaid = gssps.uaid_for_torsion_by_inconn[bt_ind, in_conn, tor_ind]
    flat_atom = _resolve_uaid(pose_stack, pose, block, uaid)
    if flat_atom == -1:
        return -1
    return int(kfo_for_atom[flat_atom])


def _apply_torsion_deltas(pose_stack, requests, deltas, fold_forest=None):
    """Apply a set of torsion deltas to a PoseStack using inverse kinematics."""
    if fold_forest is None:
        fold_forest = FoldForest.reasonable_fold_forest(pose_stack)
    kin_module = PoseStackKinematicsModule(pose_stack, fold_forest)
    kmd = kin_module.kmd

    kincoords = torch.zeros(
        (kmd.forest.id.shape[0], 3), dtype=torch.float32, device=pose_stack.device
    )
    kincoords[1:] = pose_stack.coords.view(-1, 3)[kmd.forest.id[1:].to(torch.int64)]
    dofs = inverse_kin(
        kincoords,
        kmd.forest.parent,
        kmd.forest.frame_x,
        kmd.forest.frame_y,
        kmd.forest.frame_z,
        kmd.forest.doftype,
    )

    kfo_for_atom = torch.full(
        (pose_stack.n_poses * pose_stack.max_n_pose_atoms,),
        -1,
        dtype=torch.int64,
        device=pose_stack.device,
    )
    ids = kmd.forest.id[1:].to(torch.int64)
    kfo_for_atom[ids] = torch.arange(
        1, ids.shape[0] + 1, dtype=torch.int64, device=pose_stack.device
    )

    for (pose, block, tor_ind, name), delta in zip(requests, deltas):
        node = _controlling_kfo_node(
            pose_stack, kmd, kfo_for_atom, pose, block, tor_ind
        )
        if node == -1 or kmd.forest.doftype[node] != NodeType.bond:
            raise ValueError(
                f"torsion {name!r} on block {block} of pose {pose} is not a bonded "
                "degree of freedom under this fold forest"
            )
        dofs[node, BondDOFTypes.phi_c] += delta

    kin_coords = kin_module(dofs)
    flat = pose_stack.coords.reshape(-1, 3).clone()
    flat[ids] = kin_coords[1:].to(flat.dtype)
    return attr.evolve(pose_stack, coords=flat.view(pose_stack.coords.shape))


def set_named_torsions(
    pose_stack: PoseStack,
    poses,
    blocks,
    names,
    values,
    degrees: bool = True,
    fold_forest=None,
) -> PoseStack:
    """Set named torsions, returning a new PoseStack.

    ``poses``, ``blocks``, ``names`` and ``values`` are either all scalars or
    equal-length sequences; a batch is applied in one kinematic pass. Movement
    follows the fold forest, so which side of each bond stays fixed depends on
    it; ``reasonable_fold_forest`` roots each chain at its first residue.
    """
    requests = _torsion_requests(pose_stack, poses, blocks, names)
    targets = np.atleast_1d(np.asarray(values, dtype=np.float64))
    if len(targets) == 1 and len(requests) > 1:
        targets = np.repeat(targets, len(requests))
    if len(targets) != len(requests):
        raise ValueError("values must be a scalar or match the number of torsions")
    if degrees:
        targets = np.radians(targets)

    current = _measure_torsions(pose_stack, requests, degrees=False)
    if np.isnan(current).any():
        bad = requests[int(np.argmax(np.isnan(current)))]
        raise ValueError(
            f"torsion {bad[3]!r} on block {bad[1]} of pose {bad[0]} is undefined; "
            "it reaches a residue that is not present"
        )
    return _apply_torsion_deltas(
        pose_stack, requests, targets - current, fold_forest=fold_forest
    )
