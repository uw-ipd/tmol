"""Opt-in protonation alternatives in packing.

A pose built with ``pose_stack_from_biotite(..., protonation_alternatives=True)``
records, per residue, AtomWorks' protonation alternatives: block-type base names
(such as HIS, HIS_D, HIS_POS, CYS_DEP and LYS_DEP) and their free-energy offsets
relative to the assigned state, ``1.364 * (pH - pKa)`` kcal/mol for a protonated form. A
``PackerPalette(protonation_alternatives=True)`` lets each such residue take the
block types of its alternatives across protonation states, and the packer adds
each block type's offset to its rotamers' one-body energies.
"""

from __future__ import annotations

import warnings

import attr
import numpy
import torch

from tmol.chemical import RefinedResidueType
from tmol.io._protonation import (
    PROTONATION_ALTERNATIVES,
)
from tmol.io._protonation_alternatives import parse_protonation_alternatives
from tmol.pack._packer_task import (
    PackerTask,
    SetPackerTask,
    _backbone_signatures,
    _exchangeable,
)
from tmol.pack.rotamer import RotamerSet
from tmol.pose import PoseStack


def block_alternatives(
    pose_stack: PoseStack,
) -> dict[tuple[int, int], dict[str, float]] | None:
    """``{(pose, block): {base name: offset}}`` of every block recording alternatives.

    ``None`` when the pose records no alternatives at all.
    """
    annotations = pose_stack.pdb_info.residue_annotations
    if annotations is None or PROTONATION_ALTERNATIVES not in (
        annotations.dtype.names or ()
    ):
        return None
    values = annotations[PROTONATION_ALTERNATIVES]
    real = pose_stack.block_type_ind64.cpu().numpy() >= 0
    out = {}
    for pose, block in zip(*numpy.nonzero(real[:, : values.shape[1]])):
        value = values[pose, block]
        parsed = parse_protonation_alternatives(value) if isinstance(value, str) else {}
        if len(parsed) > 1:
            out[(int(pose), int(block))] = {
                name: offset for name, (offset, _) in parsed.items()
            }
    return out


def add_protonation_alternatives(task: PackerTask, pose_stack: PoseStack) -> None:
    """Widen ``task``'s considered block types by each block's protonation alternatives.

    A block whose pose records alternatives, and whose type holds no metal
    site, also considers every block type named by an alternative that the
    default palette would exchange it with were their protonation states
    equal, with the same side-chain chirality. Its considered block types
    each get the offset of the alternative naming them (0 for others) in
    ``task.per_block_considered_block_type_offset``.
    """
    alternatives = block_alternatives(pose_stack)
    if alternatives is None:
        warnings.warn(
            "PackerPalette(protonation_alternatives=True): the pose records no "
            "protonation alternatives; build it with pose_stack_from_biotite(..., "
            "protonation_alternatives=True) from a structure whose titratable "
            "residues lack hydrogens.",
            stacklevel=3,
        )
        return
    if not alternatives:
        return
    pbt = pose_stack.packed_block_types
    types = pbt.active_block_types
    backbones = _backbone_signatures(pbt)
    by_base_name = {}
    for j, bt in enumerate(types):
        by_base_name.setdefault(bt.base_name, []).append(j)

    considered = task.per_block_considered_block_types.cpu().numpy()
    original = pose_stack.block_type_ind64.cpu().numpy()
    n_considered = task.per_block_n_considered_block_types.cpu().numpy().copy()
    extra = {}
    for (pose, block), offsets in alternatives.items():
        orig = int(original[pose, block])
        orig_bt = types[orig]
        if orig_bt.metal_sites:
            continue
        present = set(considered[pose, block, : n_considered[pose, block]].tolist())
        added = [
            j
            for name in offsets
            for j in by_base_name.get(name, ())
            if j not in present
            and not types[j].metal_sites
            and types[j].properties.polymer.sidechain_chirality
            == orig_bt.properties.polymer.sidechain_chirality
            and _exchangeable(orig_bt, types[j], backbones[orig], backbones[j])
        ]
        if added:
            extra[(pose, block)] = added

    n_new = n_considered.copy()
    for (pose, block), added in extra.items():
        n_new[pose, block] += len(added)
    width = max(considered.shape[2], int(n_new.max(initial=0)))
    device = task.per_block_considered_block_types.device

    def widened(tensor, fill):
        pad = width - tensor.shape[2]
        if pad == 0:
            return tensor.clone()
        filler = torch.full(
            (*tensor.shape[:2], pad), fill, dtype=tensor.dtype, device=device
        )
        return torch.cat([tensor, filler], dim=2)

    new_considered = widened(task.per_block_considered_block_types, -1)
    new_is_orig = widened(task.per_block_considered_block_types_is_orig, False)
    new_rtr = widened(task.restrict_to_repacking_masks, False)
    for (pose, block), added in extra.items():
        first = int(n_considered[pose, block])
        columns = slice(first, first + len(added))
        new_considered[pose, block, columns] = torch.tensor(added, device=device)
        name3 = types[int(original[pose, block])].name3
        new_rtr[pose, block, columns] = torch.tensor(
            [types[j].name3 == name3 for j in added], device=device
        )

    offset = torch.zeros(new_considered.shape, dtype=torch.float32)
    base_names = [bt.base_name for bt in types]
    considered_now = new_considered.cpu().numpy()
    for (pose, block), offsets in alternatives.items():
        for k, j in enumerate(considered_now[pose, block].tolist()):
            if j >= 0 and base_names[j] in offsets:
                offset[pose, block, k] = offsets[base_names[j]]

    task.per_block_considered_block_types = new_considered
    task.per_block_considered_block_types_is_orig = new_is_orig
    task.restrict_to_repacking_masks = new_rtr
    task.per_block_n_considered_block_types = torch.tensor(
        n_new, dtype=task.per_block_n_considered_block_types.dtype, device=device
    )
    task.per_block_considered_block_type_offset = offset.to(device)


def rotamer_offsets(
    offset: torch.Tensor, task: SetPackerTask, rotamer_set: RotamerSet
) -> torch.Tensor:
    """``[n_rotamers]`` offset of each rotamer's block type at its block."""
    pose = rotamer_set.pose_for_rot.to(torch.int64)
    block = rotamer_set.block_ind_for_rot.to(torch.int64)
    block_type = rotamer_set.block_type_ind_for_rot.to(torch.int64)
    same = task.per_block_considered_block_types[pose, block] == block_type[:, None]
    return (offset[pose, block] * same).sum(dim=1)


def protonation_state_energy(pose_stack: PoseStack) -> torch.Tensor:
    """Per-pose pH offsets for the current block types, in kcal/mol."""
    energies = torch.zeros(pose_stack.n_poses, dtype=pose_stack.coords.dtype)
    types = pose_stack.packed_block_types.active_block_types
    block_types = pose_stack.block_type_ind64.cpu().numpy()
    for (pose, block), offsets in (block_alternatives(pose_stack) or {}).items():
        energies[pose] += offsets.get(types[block_types[pose, block]].base_name, 0.0)
    return energies.to(pose_stack.device)


@attr.define(frozen=True)
class ProtonationChoice:
    """The protonation variant a packed block took.

    Attributes:
        pose: Pose index.
        block: Block index.
        chain: Input chain label.
        res_label: Input residue number.
        label: Base name of the chosen block type (``"HIS_POS"``, ...).
        offset: Its offset, kcal/mol relative to AtomWorks' assigned state.
        charge: AtomWorks' formal charge on the titrating side-chain atoms.
        hydrogens: Hydrogens on each heavy atom whose count differs among the
            alternatives' block types.
    """

    pose: int
    block: int
    chain: str
    res_label: int
    label: str
    offset: float
    charge: int
    hydrogens: dict[str, int]


def _hydrogen_counts(
    block_type: RefinedResidueType, is_hydrogen_type: dict[str, bool]
) -> dict[str, int]:
    heavy = {a.name for a in block_type.atoms if not is_hydrogen_type[a.atom_type]}
    counts = dict.fromkeys(heavy, 0)
    names = {a.name: a.atom_type for a in block_type.atoms}
    for a, b, *_ in block_type.bonds:
        for parent, child in ((a, b), (b, a)):
            if parent in heavy and child in names and is_hydrogen_type[names[child]]:
                counts[parent] += 1
    return counts


def chosen_protonation_variants(pose_stack: PoseStack) -> list[ProtonationChoice]:
    """The variant each block recording protonation alternatives holds.

    Empty when the pose records none. Mutations to other residue types are omitted.
    """
    alternatives = block_alternatives(pose_stack) or {}
    pbt = pose_stack.packed_block_types
    types = pbt.active_block_types
    is_hydrogen_type = {
        at.name: at.element.upper() in ("H", "D") for at in pbt.chem_db.atom_types
    }
    backbones = _backbone_signatures(pbt)
    by_base_name = {}
    for j, bt in enumerate(types):
        by_base_name.setdefault(bt.base_name, []).append(j)
    hydrogen_counts = [_hydrogen_counts(bt, is_hydrogen_type) for bt in types]
    block_types = pose_stack.block_type_ind64.cpu().numpy()
    out = []
    for (pose, block), offsets in sorted(alternatives.items()):
        index = int(block_types[pose, block])
        bt = types[index]
        if bt.base_name not in offsets:
            continue
        counts = [
            hydrogen_counts[j]
            for name in offsets
            for j in by_base_name.get(name, ())
            if _exchangeable(bt, types[j], backbones[index], backbones[j])
        ]
        mine = hydrogen_counts[index]
        varying = sorted(
            name for name in mine if len({c.get(name, -1) for c in counts}) > 1
        )
        choices = parse_protonation_alternatives(
            pose_stack.pdb_info.residue_annotations[PROTONATION_ALTERNATIVES][
                pose, block
            ]
        )
        out.append(
            ProtonationChoice(
                pose=pose,
                block=block,
                chain=str(pose_stack.pdb_info.chain_labels[pose, block]),
                res_label=int(pose_stack.pdb_info.residue_labels[pose, block]),
                label=bt.base_name,
                offset=offsets.get(bt.base_name, float("nan")),
                charge=choices[bt.base_name][1],
                hydrogens={name: mine[name] for name in varying},
            )
        )
    return out
