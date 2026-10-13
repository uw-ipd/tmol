"""All-bonds construction (intra + inter-residue edges).

Two things live here:

1. :func:`bonds_and_bond_ranges` -- per-residue-type helper that merges
   intra-block bonds (each pair listed twice, symmetrically) with the
   "inter-block" edges implied by the residue's ``connections``. Ported
   from ``tmol.chemical.all_bonds`` (pure numpy, no numba).

2. :func:`all_bonds_for_pose_stack` -- new helper that walks a PoseStack's
   inter-residue connection table to produce the list of inter-residue
   bonds realised in that particular pose. The returned array has rows
   ``[pose_ind, res_a, atom_a, res_b, atom_b, path_distance]`` and is
   intentionally pure numpy -- this is a preprocessing step that runs
   once per PoseStack build, not inside the JAX jit-compiled scoring
   kernels.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def bonds_and_bond_ranges(
    n_atoms: int,
    intra_res_bonds: np.ndarray,  # int32[:, 2]
    ordered_connection_atoms: np.ndarray,  # int32[:]
) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate the set of intra- and inter-block bonds.

    The intra-block bonds should list each bond twice, once for each of the
    two atoms it connects. The ``ordered_connection_atoms`` array lists the
    atom on this block that connects to another block such that the ith
    position in this array represents the ith inter-block connection; the
    case when a single atom serves as a connection point to multiple other
    blocks (as might happen with metal ions) is handled correctly.

    Returns
    -------
    bonds: ``int32[n_all_bonds, 3]``
        Each row is ``(atom_ind, other_atom_ind_or_-1, conn_id_or_-1)``.
        If the partner is on this residue, ``other_atom_ind`` is the local
        index and ``conn_id`` is ``-1``. Otherwise ``other_atom_ind`` is
        ``-1`` and ``conn_id`` is the index into this residue's
        ``connections`` tuple.
    bond_ranges: ``int32[n_atoms, 2]``
        Start/end row indices in ``bonds`` for each atom. Bonds of a given
        atom are contiguous in ``bonds``.
    """
    int_dtype = intra_res_bonds.dtype
    n_intra_bonds = int(intra_res_bonds.shape[0])
    n_inter_bonds = int(ordered_connection_atoms.shape[0])
    n_all_bonds = n_intra_bonds + n_inter_bonds

    count_intra_bonds = np.zeros((n_atoms,), dtype=int_dtype)
    count_inter_bonds = np.zeros((n_atoms,), dtype=int_dtype)

    for i in range(n_intra_bonds):
        count_intra_bonds[intra_res_bonds[i, 0]] += 1
    for i in range(n_inter_bonds):
        count_inter_bonds[ordered_connection_atoms[i]] += 1

    bond_ranges = np.full((n_atoms, 2), -1, dtype=int_dtype)
    for i in range(n_atoms):
        i_n_bonds = count_intra_bonds[i] + count_inter_bonds[i]
        if i == 0:
            if i_n_bonds > 0:
                bond_ranges[i, 0] = 0
                bond_ranges[i, 1] = i_n_bonds
            else:
                # No bonds for atom 0 -- keep a valid (start == end) range
                bond_ranges[i, 0] = 0
                bond_ranges[i, 1] = 0
        else:
            bond_ranges[i, 0] = bond_ranges[i - 1, 1]
            bond_ranges[i, 1] = bond_ranges[i - 1, 1] + i_n_bonds

    bonds = np.zeros((n_all_bonds, 3), dtype=int_dtype)
    count_bond = np.zeros((n_atoms,), dtype=int_dtype)
    for i in range(n_intra_bonds):
        i_atm = int(intra_res_bonds[i, 0])
        i_count = int(count_bond[i_atm])
        count_bond[i_atm] += 1
        bond = int(bond_ranges[i_atm, 0]) + i_count
        bonds[bond, 0] = i_atm
        bonds[bond, 1] = intra_res_bonds[i, 1]
        bonds[bond, 2] = -1
    for i in range(n_inter_bonds):
        i_atm = int(ordered_connection_atoms[i])
        i_count = int(count_bond[i_atm])
        count_bond[i_atm] += 1
        bond = int(bond_ranges[i_atm, 0]) + i_count
        bonds[bond, 0] = i_atm
        bonds[bond, 1] = -1  # unresolved; on another residue
        bonds[bond, 2] = i  # i = index of inter-residue connection

    return bonds, bond_ranges


class AllBonds:
    """Adapter that packages the per-residue all-bonds data.

    Mostly a thin, typed view over the arrays returned by
    :func:`bonds_and_bond_ranges`; provided for parity with the upstream
    module that downstream code may expect to import.
    """

    __slots__ = ("n_atoms", "bonds", "bond_ranges", "ordered_connection_atoms")

    def __init__(
        self,
        n_atoms: int,
        intra_res_bonds: np.ndarray,
        ordered_connection_atoms: np.ndarray,
    ) -> None:
        bonds, bond_ranges = bonds_and_bond_ranges(
            n_atoms, intra_res_bonds, ordered_connection_atoms
        )
        self.n_atoms = n_atoms
        self.bonds = bonds
        self.bond_ranges = bond_ranges
        self.ordered_connection_atoms = ordered_connection_atoms


# --------------------------------------------------------------------- #
# PoseStack-level all-bonds                                              #
# --------------------------------------------------------------------- #


def _as_np_int(x) -> np.ndarray:
    """Coerce a JAX / numpy / torch-like array to numpy ``int64``."""
    arr = np.asarray(x)
    if arr.dtype.kind not in ("i", "u"):
        arr = arr.astype(np.int64)
    return arr


def all_bonds_for_pose_stack(ps) -> np.ndarray:
    """Enumerate inter-residue chemical bonds for every pose in ``ps``.

    Parameters
    ----------
    ps:
        A ``PoseStack``-like object. The function uses only
        duck-typed attributes:

        * ``ps.packed_block_types`` with an ``active_block_types`` list of
          ``RefinedResidueType`` instances (providing ``connection_to_cidx``
          or ``connections`` and ``ordered_connection_atoms``).
        * ``ps.block_type_ind`` -- ``int[n_poses, max_n_res]``; ``-1`` for
          non-real residues.
        * ``ps.inter_residue_connections`` --
          ``int[n_poses, max_n_res, max_n_conn, 2]`` storing
          ``(other_res, other_conn)`` or ``(-1, -1)``.
        * (optional) ``ps.inter_block_bondsep`` --
          ``int[n_poses, max_n_res, max_n_res, max_n_conn, max_n_conn]``
          giving the chemical bond separation between any two connections
          on any two residues. When present, its min over connection pairs
          is used as ``path_distance`` (the inter-block shortest chemical
          path). When absent, ``path_distance`` defaults to ``1`` (i.e. the
          one direct bond at the connection).

    Returns
    -------
    edges: ``int32[n_edges, 6]``
        Each row is ``[pose_ind, res_a, atom_a, res_b, atom_b,
        path_distance]``. Every chemical bond appears once (in canonical
        direction ``res_a < res_b`` or, for ties on residue index, the
        lower-numbered connection "owns" the edge).
    """

    block_type_ind = _as_np_int(ps.block_type_ind)
    inter_residue_connections = _as_np_int(ps.inter_residue_connections)

    active_block_types = list(ps.packed_block_types.active_block_types)

    bondsep = getattr(ps, "inter_block_bondsep", None)
    if bondsep is not None:
        bondsep = _as_np_int(bondsep)

    n_poses, max_n_res = block_type_ind.shape[:2]

    out_rows = []

    for p in range(n_poses):
        for ra in range(max_n_res):
            bt_a = int(block_type_ind[p, ra])
            if bt_a < 0:
                continue
            btype_a = active_block_types[bt_a]
            # number of connections on this residue
            n_conn_a = len(btype_a.connections)
            for ca in range(n_conn_a):
                rb = int(inter_residue_connections[p, ra, ca, 0])
                cb = int(inter_residue_connections[p, ra, ca, 1])
                if rb < 0 or cb < 0:
                    continue

                # canonical orientation: a < b, or (a == b and ca < cb).
                # This ensures we don't list the same chemical bond twice.
                if (rb, cb) < (ra, ca):
                    continue
                if (rb, cb) == (ra, ca):
                    continue  # self-loop, skip

                bt_b = int(block_type_ind[p, rb])
                if bt_b < 0:
                    continue
                btype_b = active_block_types[bt_b]
                atom_a = int(btype_a.ordered_connection_atoms[ca])
                atom_b = int(btype_b.ordered_connection_atoms[cb])

                if bondsep is not None:
                    path_distance = int(bondsep[p, ra, rb, ca, cb])
                else:
                    path_distance = 1

                out_rows.append(
                    (p, ra, atom_a, rb, atom_b, path_distance),
                )

    if not out_rows:
        return np.zeros((0, 6), dtype=np.int32)

    return np.asarray(out_rows, dtype=np.int32)


__all__ = [
    "bonds_and_bond_ranges",
    "AllBonds",
    "all_bonds_for_pose_stack",
]
