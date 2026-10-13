"""Invert a :class:`PoseStack` back into a canonical-form dict.

Upstream reference: ``tmol/io/pose_stack_deconstruction.py``. The
upstream torch port annotates the ``PackedBlockTypes`` with a
canonical-ordering map, threads the ``pose_stack`` coords back into
``[n_poses, max_n_res, max_n_can_atoms, 3]`` layout, and walks the
inter-residue connection graph to reconstruct ``res_not_connected``
flags and disulfide pairings.

The JAX port delivers a dict rather than the upstream ``CanonicalForm``
attrs dataclass. Keys:

    "coords":          float32[n_poses, max_n_res, max_n_can_atoms, 3]
    "atom_is_present": bool[n_poses, max_n_res, max_n_can_atoms]
    "res_name":        list[list[str]] 3-letter codes (padded with "")
    "res_num":         int32[n_poses, max_n_res]
    "chain_id":        list[list[str]]
    "res_types":       int32[n_poses, max_n_res]  -- canonical indices
                       (only populated when a CanonicalOrdering is passed)

``max_n_can_atoms`` is taken from the caller-supplied
``canonical_ordering`` when present, else derived from
``pbt.max_n_atoms``. NaN-filled slots are marked absent in
``atom_is_present``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def canonical_form_from_pose_stack(
    pose_stack: Any,
    canonical_ordering: Optional[Any] = None,
) -> Dict[str, Any]:
    """Deconstruct a :class:`PoseStack` into a canonical-form dict.

    ``canonical_ordering`` is optional. When supplied (as a
    :class:`tmol.io.canonical_ordering.CanonicalOrdering`-like object
    with ``restype_io_equiv_classes`` / ``max_n_canonical_atoms``), we
    use its layout; otherwise we fall back to using the packed block
    type's own atom count as the canonical size and leave ``res_types``
    at ``-1``.
    """

    pbt = pose_stack.packed_block_types

    bti = np.asarray(pose_stack.block_type_ind)
    n_poses, max_n_res = bti.shape
    n_atoms_per_bt = np.asarray(pbt.n_atoms, dtype=np.int32)

    if canonical_ordering is not None:
        max_n_can_atoms = int(canonical_ordering.max_n_canonical_atoms)
        restype_classes = list(canonical_ordering.restype_io_equiv_classes)
    else:
        max_n_can_atoms = int(getattr(pbt, "max_n_atoms", 0) or 0)
        restype_classes = []

    # --------------------------------------------------------------
    # Scatter the flat [max_n_atoms_per_pose, 3] coords slab back into
    # [n_poses, max_n_res, max_n_can_atoms, 3].
    # --------------------------------------------------------------
    coords_flat = np.asarray(pose_stack.coords, dtype=np.float32)
    block_coord_offset = np.asarray(pose_stack.block_coord_offset, dtype=np.int32)

    cf_coords = np.full(
        (n_poses, max_n_res, max_n_can_atoms, 3), np.nan, dtype=np.float32
    )
    atom_is_present = np.zeros((n_poses, max_n_res, max_n_can_atoms), dtype=bool)

    res_name_per_pose: List[List[str]] = []
    chain_id_per_pose: List[List[str]] = []
    res_num_arr = np.full((n_poses, max_n_res), -1, dtype=np.int32)
    res_types_arr = np.full((n_poses, max_n_res), -1, dtype=np.int32)

    pdb_info = getattr(pose_stack, "pdb_info", None)
    pdb_chain_id = getattr(pdb_info, "chain_id", None) if pdb_info is not None else None
    pdb_res_num = getattr(pdb_info, "res_num", None) if pdb_info is not None else None
    pdb_res_name = getattr(pdb_info, "res_name", None) if pdb_info is not None else None
    if pdb_res_num is not None:
        pdb_res_num = np.asarray(pdb_res_num)

    for p in range(n_poses):
        row_names: List[str] = []
        row_chain: List[str] = []
        for r in range(max_n_res):
            bt_ind = int(bti[p, r])
            if bt_ind < 0:
                row_names.append("")
                row_chain.append("")
                continue
            bt = pbt.active_block_types[bt_ind]
            n_atoms = int(n_atoms_per_bt[bt_ind])

            name3 = str(getattr(bt, "name3", "") or getattr(bt, "name", ""))
            row_names.append(name3)

            start = int(block_coord_offset[p, r])
            end = start + n_atoms
            block_coords = coords_flat[p, start:end]

            explicit_map = getattr(bt, "canonical_ordering_ind_for_atom", None)
            if explicit_map is not None:
                can_map = np.asarray(explicit_map, dtype=np.int32)
            else:
                can_map = np.arange(n_atoms, dtype=np.int32)

            for a in range(n_atoms):
                can_a = int(can_map[a]) if a < len(can_map) else a
                if 0 <= can_a < max_n_can_atoms:
                    cf_coords[p, r, can_a, :] = block_coords[a, :]
                    atom_is_present[p, r, can_a] = True

            # chain / resnum from pdb_info when available, else defaults.
            if (
                pdb_chain_id is not None
                and p < len(pdb_chain_id)
                and r < len(pdb_chain_id[p])
            ):
                row_chain.append(str(pdb_chain_id[p][r]))
            else:
                row_chain.append("A")

            if (
                pdb_res_num is not None
                and pdb_res_num.shape[0] > p
                and pdb_res_num.shape[1] > r
                and pdb_res_num[p, r] >= 0
            ):
                res_num_arr[p, r] = int(pdb_res_num[p, r])

            if restype_classes and name3 in restype_classes:
                res_types_arr[p, r] = restype_classes.index(name3)

            # If pdb_info has richer names (e.g. patched variants) we
            # prefer those over the block type's name3.
            if (
                pdb_res_name is not None
                and p < len(pdb_res_name)
                and r < len(pdb_res_name[p])
                and pdb_res_name[p][r]
            ):
                row_names[-1] = str(pdb_res_name[p][r])

        res_name_per_pose.append(row_names)
        chain_id_per_pose.append(row_chain)

    # Fill in default res_num for poses where pdb_info didn't provide one.
    for p in range(n_poses):
        if np.any(res_num_arr[p] >= 0):
            continue
        counter = 1
        for r in range(max_n_res):
            if int(bti[p, r]) >= 0:
                res_num_arr[p, r] = counter
                counter += 1

    return {
        "coords": cf_coords,
        "atom_is_present": atom_is_present,
        "res_name": res_name_per_pose,
        "res_num": res_num_arr,
        "chain_id": chain_id_per_pose,
        "res_types": res_types_arr,
    }


__all__ = ["canonical_form_from_pose_stack"]
