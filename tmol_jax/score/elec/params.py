"""Electrostatics parameter containers and per-atom charge assignment.

Port of :mod:`tmol.score.elec.params` to JAX.

This module exposes

* :class:`ElecGlobalParams` — a pytree holding the five sigmoidal-
  dielectric / cutoff scalars (``D``, ``D0``, ``S``, ``min_dis``,
  ``max_dis``), loaded from :class:`tmol_jax.database.scoring.elec.ElecDatabase`.
* :func:`elec_global_params_from_database` — builds the above from a
  database object.
* :func:`atom_charges_for_pose_stack` — produces a
  ``float32[n_poses, max_n_atoms_per_pose]`` array of per-atom partial
  charges, with 0 for atoms missing from the database.

The database stores partial charges keyed by ``(residue_name,
atom_name)``, with optional patch specialisation (e.g.
``ALA:cterm``). The resolution rule mirrors upstream:

1. Parse ``block_type.name`` into ``base:patch1[:patch2]``; the
   unpatched basename is always tried last.
2. If the base residue is missing from the DB, all atoms get charge 0.
3. Otherwise, for each atom walk the patch list in order and return
   the first hit; if no patch hit, fall back to the unpatched entry.

Everything happens in numpy on the host -- charges are static per
block type so we do not need (and would gain nothing from) a JAX
scan.
"""

from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.types import pytree_dataclass

# --------------------------------------------------------------------- #
# Global parameters                                                      #
# --------------------------------------------------------------------- #


@pytree_dataclass
class ElecGlobalParams:
    """Sigmoidal-dielectric constants and distance cutoffs.

    All fields are 0-D ``jax.Array`` scalars so the whole object is a
    pytree leaf bundle that flows naturally through ``jit`` / ``vmap``.
    """

    D: jax.Array  # dielectric asymptote at large distance
    D0: jax.Array  # dielectric limit at zero distance
    S: jax.Array  # sigmoid steepness
    min_dis: jax.Array  # lower cutoff for the flat / short-fade region
    max_dis: jax.Array  # upper cutoff (energy = 0 beyond here)


def elec_global_params_from_database(elec_db: Any) -> ElecGlobalParams:
    """Construct :class:`ElecGlobalParams` from an ``ElecDatabase``.

    Accepts either the :mod:`tmol_jax.database.scoring.elec`
    ``ElecGlobalParams`` attrs class or any duck-typed object with the
    same five attributes.
    """

    gp = elec_db.global_parameters
    return ElecGlobalParams(
        D=jnp.float32(gp.elec_sigmoidal_die_D),
        D0=jnp.float32(gp.elec_sigmoidal_die_D0),
        S=jnp.float32(gp.elec_sigmoidal_die_S),
        min_dis=jnp.float32(gp.elec_min_dis),
        max_dis=jnp.float32(gp.elec_max_dis),
    )


# --------------------------------------------------------------------- #
# Partial-charge lookup                                                  #
# --------------------------------------------------------------------- #


def _partial_charges_dict(elec_db: Any) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Turn ``elec_db.atom_charge_parameters`` into a nested dict
    ``{res: {atom_name: {patch: charge}}}`` where ``patch == ''`` means
    "unpatched".
    """

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for row in elec_db.atom_charge_parameters:
        tag = row.res.split(":")
        assert len(tag) <= 2, (
            "Each atom charge can only be specialized by one patch: " + row.res
        )
        if len(tag) == 1:
            base, patch = tag[0], ""
        else:
            base, patch = tag[0], tag[1]
        out.setdefault(base, {}).setdefault(row.atom, {})[patch] = float(row.charge)
    return out


def _charges_for_block_type(
    block_type: Any,
    partial_charges: Dict[str, Dict[str, Dict[str, float]]],
) -> np.ndarray:
    """Per-atom partial charges for one block type, as ``float32[n_atoms]``.

    Atoms absent from the DB return 0.0 (silently). That matches the
    upstream treatment of virtual atoms.
    """

    atoms = tuple(block_type.atoms)
    out = np.zeros(len(atoms), dtype=np.float32)

    # Restype-carried charges (e.g. a ligand RefinedResidueType built
    # from a ``.tmol`` params file) take precedence over the elec-DB
    # lookup. Canonical residues never set this attribute, so their
    # charges are byte-unchanged. Mirrors upstream's per-block
    # ``elec_partial_charge`` annotation.
    restype_charges = getattr(block_type, "elec_partial_charges", None)
    if restype_charges is not None:
        for i, atm in enumerate(atoms):
            atm_name = getattr(atm, "name", atm)
            if atm_name in restype_charges:
                out[i] = np.float32(restype_charges[atm_name])
        return out

    name = str(block_type.name)
    res, *patches = name.split(":")
    # Unpatched last — see upstream `get_partial_charges_for_block`.
    patches.append("")

    res_table = partial_charges.get(res)
    if res_table is None:
        return out

    for i, atm in enumerate(atoms):
        atm_name = getattr(atm, "name", atm)
        atom_table = res_table.get(atm_name)
        if atom_table is None:
            # Atom not in DB: upstream raises; we silently zero.
            continue
        for variant in patches:
            if variant in atom_table:
                out[i] = np.float32(atom_table[variant])
                break
    return out


def _per_block_type_charges(
    packed_block_types: Any,
    partial_charges: Dict[str, Dict[str, Dict[str, float]]],
) -> np.ndarray:
    """``float32[n_types, max_n_atoms]`` of charges, zero-padded."""

    n_types = int(packed_block_types.n_types)
    max_n_atoms = int(packed_block_types.max_n_atoms)
    out = np.zeros((n_types, max_n_atoms), dtype=np.float32)
    for i, bt in enumerate(packed_block_types.active_block_types):
        row = _charges_for_block_type(bt, partial_charges)
        out[i, : row.shape[0]] = row
    return out


def atom_charges_for_pose_stack(
    pose_stack: Any,
    packed_block_types: Any,
    elec_db: Any,
) -> jax.Array:
    """Assign a partial charge to every atom slot of a ``PoseStack``.

    Parameters
    ----------
    pose_stack:
        A :class:`~tmol_jax.pose.pose_stack.PoseStack` -- we use
        ``block_type_ind``, ``block_coord_offset``, and the shape of
        ``coords``.
    packed_block_types:
        The matching ``PackedBlockTypes`` palette. Must match the one
        stored on ``pose_stack``.
    elec_db:
        An ``ElecDatabase`` (or anything with ``atom_charge_parameters``
        of the same shape).

    Returns
    -------
    ``float32[n_poses, max_n_atoms_per_pose]``
        Row-major, with zero padding for non-real atom slots and for
        atoms missing from the charge DB.
    """
    # Per-BT charges depend only on the palette; reuse the table cached
    # on ``packed_block_types`` by :func:`_annotate_packed_block_types`
    # if present, otherwise build it once here.
    per_type_charges = getattr(packed_block_types, "elec_partial_charge", None)
    if per_type_charges is None:
        partial_charges = _partial_charges_dict(elec_db)
        per_type_charges = _per_block_type_charges(packed_block_types, partial_charges)
    per_type_charges = jnp.asarray(per_type_charges, dtype=jnp.float32)

    # Use the pose-level index cache shared across every term so a
    # single ``pose_atom_block_index`` trace is reused.
    from tmol_jax.score.common.count_pair import scatter_many_per_atom

    (charges,) = scatter_many_per_atom(
        (per_type_charges,), pose_stack, packed_block_types, defaults=(0.0,)
    )
    return charges


# --------------------------------------------------------------------- #
# Count-pair representative mapping                                      #
# --------------------------------------------------------------------- #


def _cp_reps_dict(elec_db: Any) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Parse ``atom_cp_reps_parameters`` into
    ``{res: {outer_atom: {patch: inner_atom}}}``.

    See the upstream comment in :mod:`tmol.score.elec.params` for the
    subtle "last-write-wins" semantics of overlapping entries; we keep
    that behaviour by iterating in declared order and simply
    overwriting.
    """

    out: Dict[str, Dict[str, Dict[str, str]]] = {}
    for row in elec_db.atom_cp_reps_parameters:
        tag = row.res.split(":")
        assert len(tag) <= 2, (
            "Each cp-rep mapping can only be specialized by one patch: " + row.res
        )
        if len(tag) == 1:
            base, patch = tag[0], ""
        else:
            base, patch = tag[0], tag[1]
        out.setdefault(base, {}).setdefault(row.atm_outer, {})[patch] = row.atm_inner
    return out


def representative_mapping_for_block_type(
    block_type: Any, cp_reps: Dict[str, Dict[str, Dict[str, str]]]
) -> np.ndarray:
    """``int32[n_atoms]`` remap ``atom -> representative_atom`` for one BT.

    An atom with no remap points at itself. This is the count-pair
    representative logic: pretend a closer atom is actually at the
    position of a specific further-out atom, so short-range
    interactions get correctly scaled.
    """

    n = len(block_type.atoms)
    mapping = np.arange(n, dtype=np.int32)

    name = str(block_type.name)
    res, *patches = name.split(":")
    patches.append("")

    res_table = cp_reps.get(res)
    if res_table is None:
        return mapping

    atom_to_idx = getattr(block_type, "atom_to_idx", None)
    if atom_to_idx is None:
        atom_to_idx = {a.name: i for i, a in enumerate(block_type.atoms)}

    for outer_name in atom_to_idx.keys():
        patch_table = res_table.get(outer_name)
        if patch_table is None:
            continue

        inner_name = None
        for v in patches:
            if v in patch_table:
                inner_name = patch_table[v]
                break
        if inner_name is None:
            continue
        if inner_name not in atom_to_idx:
            raise KeyError(
                "Invalid elec cp mapping: "
                + res
                + " "
                + outer_name
                + "->"
                + str(inner_name)
            )
        # "inner" is really the far atom used as outer's representative
        mapping[atom_to_idx[inner_name]] = atom_to_idx[outer_name]

    return mapping


__all__ = [
    "ElecGlobalParams",
    "elec_global_params_from_database",
    "atom_charges_for_pose_stack",
    "representative_mapping_for_block_type",
]
