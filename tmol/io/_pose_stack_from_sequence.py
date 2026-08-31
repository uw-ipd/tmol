from typing import Dict, List, Optional

import attr
import numpy
import torch

from tmol.chemical import ResidueTypeSet
from tmol.database import ParameterDatabase
from tmol.io._build_context import PoseBuildContext
from tmol.io._canonical_ordering import CanonicalOrdering, default_packed_block_types
from tmol.pose._packed_block_types import PackedBlockTypes
from tmol.pose._pose_stack import PoseStack
from tmol.pose._pose_stack_builder import PoseStackBuilder
from tmol.pose._sequence import (
    resolve_block_type_names,
    smiles_in_tokens,
    tokenize_sequences,
)
from tmol.utility import resolve_device


def create_pose_stack_from_sequences(
    seqs,  # str | Sequence[str]
    packed_block_types: Optional[PackedBlockTypes] = None,
    device: Optional[torch.device] = None,
    param_db=None,
    termini: bool = True,
    context: Optional[PoseBuildContext] = None,
    return_context: bool = False,
):
    """Construct a PoseStack with zero coordinates from sequence strings.

    See tmol.pose._sequence for the grammar. Returns
    (PoseStack, PoseBuildContext) when return_context is set; the context
    carries the database extended with any ligands the sequence names.
    """
    device = resolve_device(device if device is not None else torch.device("cpu"))
    if context is not None and param_db is not None:
        raise ValueError("pass either context= or param_db=, not both")

    tokens, chain_lengths = tokenize_sequences(seqs)

    if context is not None:
        if context.packed_block_types.device.type != device.type:
            raise ValueError(
                f"context was built for device "
                f"{context.packed_block_types.device} but device is {device}"
            )
        param_db = context.parameter_database
        restype_set = context.restype_set
        ligand_names = context.ligand_names
        canonical_ordering = context.canonical_ordering
    else:
        smiles = smiles_in_tokens(tokens)
        if smiles:
            from tmol.ligand import prepare_ligands_from_smiles

            param_db, ligand_names = prepare_ligands_from_smiles(
                smiles, param_db=param_db
            )
        else:
            param_db = param_db or ParameterDatabase.get_default()
            ligand_names = {}
        restype_set = ResidueTypeSet.from_database(param_db.chemical)
        canonical_ordering = None

    names, chain_lengths = resolve_block_type_names(
        tokens, chain_lengths, restype_set, ligand_names, termini
    )

    if packed_block_types is None:
        if restype_set.chem_db is ParameterDatabase.get_default().chemical:
            packed_block_types = default_packed_block_types(device)
        else:
            packed_block_types = PackedBlockTypes.from_restype_list(
                restype_set.chem_db, restype_set, restype_set.residue_types, device
            )

    pose_stack = PoseStackBuilder.from_block_type_names(
        packed_block_types, names, chain_lengths
    )
    if not return_context:
        return pose_stack

    if canonical_ordering is None:
        canonical_ordering = CanonicalOrdering.from_chemdb(param_db.chemical)
    return pose_stack, PoseBuildContext(
        canonical_ordering=canonical_ordering,
        packed_block_types=packed_block_types,
        parameter_database=param_db,
        restype_set=restype_set,
        ligand_names=ligand_names,
    )


# ---------------------------------------------------------------------------
# Backbone torsion values to 'build extended'
#    beta strand for protein / B-form DNA / A-form RNA
# ---------------------------------------------------------------------------

EXTENDED_BACKBONE_TORSIONS = {
    "alpha_aa": {"phi": -135.0, "psi": 135.0, "omega": 180.0},
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

    See tmol.pose._sequence for the grammar. Returns
    (PoseStack, PoseBuildContext) when return_context is set.
    """
    pose_stack, build_context = create_pose_stack_from_sequences(
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


# ---------------------------------------------------------------------------
# Helpers used only by extended_pose_stack_from_sequences
# ---------------------------------------------------------------------------


def _set_ideal_backbone_torsions(pose_stack: PoseStack) -> PoseStack:
    from tmol.pose._util import _measure_torsions
    from tmol.kinematics._pose_stack_kinematics import _apply_torsion_deltas

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
    """Slide each chain along x so no two chains overlap."""
    separation = 6.0
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
    src_center = src.mean(axis=0)
    dst_center = dst.mean(axis=0)
    h = (src - src_center).T @ (dst - dst_center)
    u, _, vt = numpy.linalg.svd(h)
    d = numpy.sign(numpy.linalg.det(vt.T @ u.T))
    rot = vt.T @ numpy.diag([1.0, 1.0, d]) @ u.T
    return rot, dst_center - rot @ src_center
