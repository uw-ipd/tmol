"""Building poses from sequences and reading/writing named torsions."""

from typing import Dict, List, Optional

import attr
import numpy
import torch

from tmol.numeric.dihedrals import coord_dihedrals
from tmol.pose.pose_stack import PoseStack
from tmol.pose.pose_stack_builder import PoseStackBuilder

# Backbone torsion values to 'build extended'
#    beta strand for protein
#    B form DNA
#    A form RNA
EXTENDED_BACKBONE_TORSIONS = {
    "alpha": {"phi": -135.0, "psi": 135.0, "omega": 180.0},
    "dna": {
        "alpha": -30.0,
        "beta": 136.0,
        "gamma": 31.0,
        "epsilon": -141.0,
        "zeta": -161.0,
        "chi1": -98.0,
    },
    "rna": {
        "alpha": -68.0,
        "beta": 178.0,
        "gamma": 54.0,
        "epsilon": -153.0,
        "zeta": -71.0,
        "chi1": -158.0,
    },
}


def extended_pose_stack_from_sequences(
    seqs,  # str | Sequence[str]
    device: Optional[torch.device] = None,
    param_db=None,
    termini: bool = True,
    context=None,
    return_context: bool = False,
):
    """Build a PoseStack from sequences with ideal geometry and extended
    backbone torsions.

    See tmol.pose.sequence for the sequence grammar.
    """
    pose_stack, build_context = PoseStackBuilder.from_sequences(
        seqs,
        device=device,
        param_db=param_db,
        termini=termini,
        context=context,
        return_context=True,
    )
    pose_stack = attr.evolve(pose_stack, coords=_ideal_chained_coords(pose_stack))
    pose_stack = _set_ideal_backbone_torsions(pose_stack)
    pose_stack = _separate_pose_stack_chains(pose_stack)
    return (pose_stack, build_context) if return_context else pose_stack


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
    targets = numpy.atleast_1d(numpy.asarray(values, dtype=numpy.float64))
    if len(targets) == 1 and len(requests) > 1:
        targets = numpy.repeat(targets, len(requests))
    if len(targets) != len(requests):
        raise ValueError("values must be a scalar or match the number of torsions")
    if degrees:
        targets = numpy.radians(targets)

    current = _measure_torsions(pose_stack, requests, degrees=False)
    if numpy.isnan(current).any():
        bad = requests[int(numpy.argmax(numpy.isnan(current)))]
        raise ValueError(
            f"torsion {bad[3]!r} on block {bad[1]} of pose {bad[0]} is undefined; "
            "it reaches a residue that is not present"
        )
    return _apply_torsion_deltas(
        pose_stack, requests, targets - current, fold_forest=fold_forest
    )


def _is_scalar(x):
    return isinstance(x, (int, numpy.integer, str))


def _as_list(x, default):
    if x is None:
        return list(default)
    return [x] if _is_scalar(x) else list(x)


def _torsion_requests(pose_stack, poses, blocks, names):
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


def _controlling_kfo_node(pose_stack, kmd, kfo_for_atom, pose, block, tor_ind):
    """Kinforest node whose phi_c holds a torsion, or -1 if unresolvable.

    The torsion tracks that phi_c with slope +1 whichever of the torsion's two
    middle atoms is the child: phi_c turns about parent->child, so the reversed
    axis and the reversed end of the dihedral cancel.
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
    from tmol.kinematics.compiled import inverse_kin
    from tmol.kinematics.datatypes import BondDOFTypes, NodeType
    from tmol.kinematics.fold_forest import FoldForest
    from tmol.kinematics.script_modules import PoseStackKinematicsModule

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


def _set_ideal_backbone_torsions(pose_stack: PoseStack) -> PoseStack:
    pbt = pose_stack.packed_block_types
    requests, values = [], []
    for pose in range(pose_stack.n_poses):
        for block in range(pose_stack.max_n_blocks):
            bt_ind = int(pose_stack.block_type_ind64[pose, block])
            if bt_ind == -1:
                continue
            bt = pbt.active_block_types[bt_ind]
            targets = EXTENDED_BACKBONE_TORSIONS.get(
                bt.properties.polymer.backbone_type
            )
            if targets is None:
                continue
            for j, tor in enumerate(bt.torsions):
                if tor.name in targets:
                    requests.append((pose, block, j, tor.name))
                    values.append(targets[tor.name])
    if not requests:
        return pose_stack

    current = _measure_torsions(pose_stack, requests, degrees=False)
    deltas = numpy.radians(numpy.array(values)) - current
    defined = ~numpy.isnan(deltas)
    requests = [r for r, keep in zip(requests, defined) if keep]
    return _apply_torsion_deltas(pose_stack, requests, deltas[defined])


def _ideal_chained_coords(pose_stack: PoseStack) -> torch.Tensor:
    pbt = pose_stack.packed_block_types
    coords = numpy.zeros(pose_stack.coords.shape, dtype=numpy.float32)
    for pose in range(pose_stack.n_poses):
        placed: Dict[int, numpy.ndarray] = {}
        for block in range(pose_stack.max_n_blocks):
            bt_ind = int(pose_stack.block_type_ind64[pose, block])
            if bt_ind == -1:
                continue
            bt = pbt.active_block_types[bt_ind]
            local = numpy.asarray(bt.ideal_coords, dtype=numpy.float64)

            prev = _bonded_predecessor(pose_stack, pose, block)
            if prev is None:
                # a chain with no bonded predecessor starts in its own frame;
                # _separate_pose_stack_chains spreads the chains out later
                placed[block] = local
            else:
                rot, trans = _junction_transform(
                    pose_stack, pose, block, bt, local, prev, placed
                )
                placed[block] = local @ rot.T + trans

        for block, block_coords in placed.items():
            bt_ind = int(pose_stack.block_type_ind64[pose, block])
            bt = pbt.active_block_types[bt_ind]
            offset = int(pose_stack.block_coord_offset64[pose, block])
            coords[pose, offset : offset + bt.n_atoms] = block_coords[
                bt.at_to_icoor_ind
            ]
    return torch.tensor(coords, dtype=torch.float32, device=pose_stack.device)


def _separate_pose_stack_chains(pose_stack: PoseStack) -> PoseStack:
    """Slide each chain along x so no two chains overlap.

    Chains have no chemistry tying them together, so their relative placement is
    arbitrary; this only keeps them from sitting on top of each other.
    """
    separation = 6.0  # gap between chain bounding boxes
    pbt = pose_stack.packed_block_types
    coords = pose_stack.coords.clone()
    for pose in range(pose_stack.n_poses):
        atoms_for_chain: Dict[int, List[numpy.ndarray]] = {}
        for block in range(pose_stack.max_n_blocks):
            bt_ind = int(pose_stack.block_type_ind64[pose, block])
            if bt_ind == -1:
                continue
            offset = int(pose_stack.block_coord_offset64[pose, block])
            n_atoms = int(pbt.n_atoms[bt_ind])
            chain = int(pose_stack.chain_id[pose, block])
            atoms_for_chain.setdefault(chain, []).append(
                numpy.arange(offset, offset + n_atoms)
            )
        if len(atoms_for_chain) < 2:
            continue

        next_x = 0.0
        for chain in sorted(atoms_for_chain):
            inds = torch.from_numpy(numpy.concatenate(atoms_for_chain[chain])).to(
                pose_stack.device
            )
            xs = coords[pose, inds, 0]
            shift = next_x - float(xs.min())
            coords[pose, inds, 0] += shift
            next_x = float(xs.max()) + shift + separation
    return attr.evolve(pose_stack, coords=coords)


def _bonded_predecessor(pose_stack, pose, block):
    """(block, connection) of the residue this block's down connection joins."""
    bt_ind = int(pose_stack.block_type_ind64[pose, block])
    bt = pose_stack.packed_block_types.active_block_types[bt_ind]
    if bt.down_connection_ind == -1:
        return None
    partner = pose_stack.inter_residue_connections64[
        pose, block, bt.down_connection_ind
    ]
    prev_block, prev_conn = int(partner[0]), int(partner[1])
    if prev_block == -1 or prev_block >= block:
        return None
    prev_bt_ind = int(pose_stack.block_type_ind64[pose, prev_block])
    prev_bt = pose_stack.packed_block_types.active_block_types[prev_bt_ind]
    if prev_conn != prev_bt.up_connection_ind:
        return None
    return prev_block, prev_conn


def _junction_transform(pose_stack, pose, block, bt, local, prev, placed):
    pbt = pose_stack.packed_block_types
    prev_block, prev_conn = prev
    prev_bt = pbt.active_block_types[int(pose_stack.block_type_ind64[pose, prev_block])]
    prev_coords = placed[prev_block]

    prev_conn_name = prev_bt.connections[prev_conn].name
    prev_mainchain = prev_bt.properties.polymer.mainchain_atoms
    conn_atom = prev_bt.connections[prev_conn].atom
    ref_atom = prev_mainchain[prev_mainchain.index(conn_atom) - 1]

    ref = prev_coords[prev_bt.icoors_index[ref_atom]]
    anchor = prev_coords[prev_bt.icoors_index[conn_atom]]
    hinge = prev_coords[prev_bt.icoors_index[prev_conn_name]]

    mainchain = bt.properties.polymer.mainchain_atoms
    src = numpy.array(
        [
            local[bt.icoors_index[bt.connections[bt.down_connection_ind].name]],
            local[bt.icoors_index[mainchain[0]]],
            local[bt.icoors_index[mainchain[1]]],
        ]
    )
    dist = numpy.linalg.norm(src[2] - src[1])
    angle = _angle(src[0], src[1], src[2])
    torsion = numpy.radians(_junction_torsion_value(prev_bt, prev_conn))

    dst = numpy.array(
        [anchor, hinge, _place_atom(ref, anchor, hinge, dist, angle, torsion)]
    )
    return _rigid_transform(src, dst)


def _junction_torsion_value(prev_bt, prev_conn):
    """Target dihedral about the bond leaving prev_bt through prev_conn."""
    targets = EXTENDED_BACKBONE_TORSIONS.get(
        prev_bt.properties.polymer.backbone_type, {}
    )
    for tor in prev_bt.torsions:
        uaids = prev_bt.torsion_to_uaids[tor.name]
        spans = (
            uaids[0][1] == -1
            and uaids[1][1] == -1
            and uaids[2][1] == prev_conn
            and uaids[3][1] == prev_conn
        )
        if spans:
            return targets.get(tor.name, 180.0)
    return 180.0


def _angle(a, b, c):
    ba = a - b
    bc = c - b
    cos = numpy.dot(ba, bc) / (numpy.linalg.norm(ba) * numpy.linalg.norm(bc))
    return numpy.arccos(numpy.clip(cos, -1.0, 1.0))


def _place_atom(a, b, c, dist, angle, torsion):
    """Position d with |cd| = dist, angle(b,c,d) = angle, dihedral(a,b,c,d) = torsion."""
    bc = c - b
    bc /= numpy.linalg.norm(bc)
    n = numpy.cross(bc, a - b)
    n /= numpy.linalg.norm(n)
    m = numpy.cross(n, bc)
    d2 = numpy.array(
        [
            -dist * numpy.cos(angle),
            dist * numpy.sin(angle) * numpy.cos(torsion),
            dist * numpy.sin(angle) * numpy.sin(torsion),
        ]
    )
    return c + d2[0] * bc + d2[1] * m + d2[2] * n


def _rigid_transform(src, dst):
    """Rotation and translation mapping src onto dst (Kabsch)."""
    src_center = src.mean(axis=0)
    dst_center = dst.mean(axis=0)
    h = (src - src_center).T @ (dst - dst_center)
    u, _, vt = numpy.linalg.svd(h)
    d = numpy.sign(numpy.linalg.det(vt.T @ u.T))
    rot = vt.T @ numpy.diag([1.0, 1.0, d]) @ u.T
    return rot, dst_center - rot @ src_center
