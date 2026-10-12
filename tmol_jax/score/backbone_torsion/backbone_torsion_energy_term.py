"""BackboneTorsionEnergyTerm — Rama + bb-dependent omega in pure JAX.

This is the JAX port of
``tmol/score/backbone_torsion/bb_torsion_energy_term.py``. The
evaluation matches the three torsions defined by Rosetta:

* ``phi_i``   = dihedral(C_{i-1}, N_i,     CA_i,    C_i)
* ``psi_i``   = dihedral(N_i,     CA_i,    C_i,     N_{i+1})
* ``omega_i`` = dihedral(CA_i,    C_i,     N_{i+1}, CA_{i+1})

For each residue we bilinearly interpolate the 2-D ``(phi, psi)``
Ramachandran energy grid (``ScoreType.rama``) and the bb-dependent
harmonic omega penalty (``ScoreType.omega``). Residues lacking a
previous (``down`` connection) or next (``up`` connection) neighbour
are masked out — their contributions are simply 0.

The term lives inside the standard :class:`tmol_jax.score.EnergyTerm`
contract:

* :meth:`setup_for_pose_stack` runs once per PoseStack and returns a
  pytree of per-residue "backbone-atom pointers" plus the packed
  parameter tables.
* :meth:`__call__` runs inside jit/grad and is fully vectorised with
  :func:`jax.vmap` over residues and poses.

If the Rama or omega bb-dep zip payload cannot be loaded (e.g. because
``torch`` isn't installed), the packed tables are empty and both score
channels return a vector of zeros — so the term remains safe to include
in a score function even without the binary data.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.score.backbone_torsion.omega import (
    OmegaBBDepDatabaseAccess,
    OmegaParams,
    omega_score,
)
from tmol_jax.score.backbone_torsion.rama import (
    RamaDatabaseAccess,
    rama_score,
)
from tmol_jax.score.common.geom import dihedral_angle
from tmol_jax.score.energy_term import EnergyTerm
from tmol_jax.score.score_types import ScoreType
from tmol_jax.types import pytree_dataclass

# Sentinel for "atom not resolvable" — the whole-pose gather below uses a
# safe-index trick and then masks on this sentinel.
_NO_ATOM = np.int32(-1)


# --------------------------------------------------------------------------- #
# Per-pose setup container                                                    #
# --------------------------------------------------------------------------- #


@pytree_dataclass
class BackboneTorsionTermState:
    """Per-pose precomputed state for :class:`BackboneTorsionEnergyTerm`.

    Fields
    ------
    phi_atom_idx
        ``int32[n_poses, max_n_blocks, 4]`` — global (pose-local) atom
        indices of the four atoms defining phi for residue i. ``-1``
        entries mean phi is undefined (terminal).
    psi_atom_idx
        Same shape/semantics for psi.
    omega_atom_idx
        Same shape/semantics for omega.
    aa_rama_idx
        ``int32[n_poses, max_n_blocks]`` — per-residue index into the
        leading (AA) axis of the rama table. ``-1`` disables rama.
    aa_omega_idx
        ``int32[n_poses, max_n_blocks]`` — same for the omega tables.
    rama_table
        ``float[n_aa, n_phi, n_psi]`` packed energy grid.
    rama_bbstart, rama_bbstep
        ``float[2]`` axis start and step for the rama table.
    omega_params
        :class:`OmegaParams` bundle: ``k_table``, ``omega0_table``,
        ``bbstart``, ``bbstep``.
    """

    phi_atom_idx: jax.Array
    psi_atom_idx: jax.Array
    omega_atom_idx: jax.Array
    aa_rama_idx: jax.Array
    aa_omega_idx: jax.Array
    rama_table: jax.Array
    rama_bbstart: jax.Array
    rama_bbstep: jax.Array
    omega_params: OmegaParams


# --------------------------------------------------------------------------- #
# Helpers — CPU-side setup                                                    #
# --------------------------------------------------------------------------- #


def _find_atom_idx(block_type: Any, name: str) -> int:
    """Return the in-block atom index for ``name`` or ``-1`` if absent."""
    return int(block_type.atom_to_idx.get(name, -1))


def _connection_atom(block_type: Any, conn_name: str) -> int:
    """Return the in-block atom index attached to ``conn_name`` or ``-1``."""
    atom_name: Optional[str] = None
    for conn in block_type.connections:
        if conn.name == conn_name:
            atom_name = conn.atom
            break
    if atom_name is None:
        return -1
    return _find_atom_idx(block_type, atom_name)


def _connection_index(block_type: Any, conn_name: str) -> int:
    """Return the index of ``conn_name`` within the block type's
    connection list (or ``-1``)."""
    for i, conn in enumerate(block_type.connections):
        if conn.name == conn_name:
            return i
    return -1


def _resolve_torsion_atoms(
    pose_stack: Any,
    pbt: Any,
    per_block_atom_N: np.ndarray,
    per_block_atom_CA: np.ndarray,
    per_block_atom_C: np.ndarray,
    per_block_down_cind: np.ndarray,
    per_block_up_cind: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the global ``(phi, psi, omega)`` atom-index tensors.

    Fully vectorised numpy — no per-block Python loop. For every
    ``(pose, block)`` we look up N/CA/C locally, then resolve the
    previous / next residue via the matching ``down`` / ``up``
    connection to build the full four-atom index tuples. Sentinel
    ``-1`` marks "torsion undefined for this residue".
    """
    block_type_ind = np.asarray(pose_stack.block_type_ind, dtype=np.int32)
    block_coord_offset = np.asarray(pose_stack.block_coord_offset, dtype=np.int32)
    inter_conn = np.asarray(pose_stack.inter_residue_connections, dtype=np.int32)

    n_poses, max_n_blocks = block_type_ind.shape

    real = block_type_ind >= 0  # [P, B]
    safe_bti = np.where(real, block_type_ind, 0)

    # Per-(pose, block) N/CA/C locals.
    n_local = per_block_atom_N[safe_bti]  # [P, B]
    ca_local = per_block_atom_CA[safe_bti]
    c_local = per_block_atom_C[safe_bti]
    down_cind = per_block_down_cind[safe_bti]
    up_cind = per_block_up_cind[safe_bti]

    has_ncac = real & (n_local >= 0) & (ca_local >= 0) & (c_local >= 0)

    n_g = np.where(has_ncac, block_coord_offset + n_local, -1)
    ca_g = np.where(has_ncac, block_coord_offset + ca_local, -1)
    c_g = np.where(has_ncac, block_coord_offset + c_local, -1)

    # --- Previous residue via "down" connection --- #
    # Gather inter_conn at (p, b, down_cind) -> [P, B, 2].
    P_ar = np.arange(n_poses)[:, None]
    B_ar = np.arange(max_n_blocks)[None, :]
    safe_down = np.where(has_ncac & (down_cind >= 0), down_cind, 0)
    prev_entry = inter_conn[P_ar, B_ar, safe_down]  # [P, B, 2]
    prev_b = prev_entry[..., 0]  # [P, B]
    prev_valid = has_ncac & (down_cind >= 0) & (prev_b >= 0)
    safe_prev_b = np.where(prev_valid, prev_b, 0)
    prev_bt = block_type_ind[P_ar, safe_prev_b]  # [P, B]
    prev_valid = prev_valid & (prev_bt >= 0)
    safe_prev_bt = np.where(prev_valid, prev_bt, 0)
    prev_c_local = per_block_atom_C[safe_prev_bt]
    prev_valid = prev_valid & (prev_c_local >= 0)
    prev_c_g = np.where(
        prev_valid,
        block_coord_offset[P_ar, safe_prev_b] + prev_c_local,
        -1,
    )

    # --- Next residue via "up" connection --- #
    safe_up = np.where(has_ncac & (up_cind >= 0), up_cind, 0)
    next_entry = inter_conn[P_ar, B_ar, safe_up]
    next_b = next_entry[..., 0]
    next_valid_base = has_ncac & (up_cind >= 0) & (next_b >= 0)
    safe_next_b = np.where(next_valid_base, next_b, 0)
    next_bt = block_type_ind[P_ar, safe_next_b]
    next_valid_base = next_valid_base & (next_bt >= 0)
    safe_next_bt = np.where(next_valid_base, next_bt, 0)
    next_n_local = per_block_atom_N[safe_next_bt]
    next_ca_local = per_block_atom_CA[safe_next_bt]
    next_n_valid = next_valid_base & (next_n_local >= 0)
    next_ca_valid = next_valid_base & (next_ca_local >= 0)
    next_n_g = np.where(
        next_n_valid,
        block_coord_offset[P_ar, safe_next_b] + next_n_local,
        -1,
    )
    next_ca_g = np.where(
        next_ca_valid,
        block_coord_offset[P_ar, safe_next_b] + next_ca_local,
        -1,
    )

    phi = np.full((n_poses, max_n_blocks, 4), -1, dtype=np.int32)
    psi = np.full((n_poses, max_n_blocks, 4), -1, dtype=np.int32)
    omega = np.full((n_poses, max_n_blocks, 4), -1, dtype=np.int32)

    phi_ok = prev_valid & has_ncac & (prev_c_g >= 0)
    psi_ok = has_ncac & (next_n_g >= 0)
    omega_ok = has_ncac & (next_n_g >= 0) & (next_ca_g >= 0)

    phi_tup = np.stack([prev_c_g, n_g, ca_g, c_g], axis=-1)
    psi_tup = np.stack([n_g, ca_g, c_g, next_n_g], axis=-1)
    omega_tup = np.stack([ca_g, c_g, next_n_g, next_ca_g], axis=-1)

    phi = np.where(phi_ok[..., None], phi_tup, phi)
    psi = np.where(psi_ok[..., None], psi_tup, psi)
    omega = np.where(omega_ok[..., None], omega_tup, omega)

    return phi, psi, omega


# --------------------------------------------------------------------------- #
# Vectorised per-residue evaluation                                           #
# --------------------------------------------------------------------------- #


def _gather_coords(coords_pose: jax.Array, atom_idx: jax.Array) -> jax.Array:
    """Gather 4 atom coordinates while tolerating ``-1`` sentinels.

    Returns ``float32[4, 3]``. ``-1`` entries are clamped to index 0
    on gather; the caller zeros their contribution via a mask.
    """
    safe = jnp.where(atom_idx >= 0, atom_idx, 0)
    return coords_pose[safe]


def _residue_energies(
    coords_pose: jax.Array,
    phi_idx: jax.Array,  # int32[4]
    psi_idx: jax.Array,  # int32[4]
    omega_idx: jax.Array,  # int32[4]
    aa_rama: jax.Array,  # scalar
    aa_omega: jax.Array,  # scalar
    rama_table: jax.Array,
    rama_bbstart: jax.Array,
    rama_bbstep: jax.Array,
    omega_params: OmegaParams,
) -> Tuple[jax.Array, jax.Array]:
    """Compute ``(rama, omega)`` energies for a single residue."""

    phi_ok = jnp.all(phi_idx >= 0)
    psi_ok = jnp.all(psi_idx >= 0)
    omega_ok = jnp.all(omega_idx >= 0)
    bb_ok = jnp.logical_and(phi_ok, psi_ok)

    phi_atoms = _gather_coords(coords_pose, phi_idx)
    psi_atoms = _gather_coords(coords_pose, psi_idx)
    omega_atoms = _gather_coords(coords_pose, omega_idx)

    phi = dihedral_angle(phi_atoms[0], phi_atoms[1], phi_atoms[2], phi_atoms[3])
    psi = dihedral_angle(psi_atoms[0], psi_atoms[1], psi_atoms[2], psi_atoms[3])
    omega = dihedral_angle(
        omega_atoms[0], omega_atoms[1], omega_atoms[2], omega_atoms[3]
    )

    # Also require ``aa_rama`` / ``aa_omega`` >= 0 before considering
    # rama/omega valid — matches upstream's ``rama_table_ind >= 0`` and
    # ``omega_table_ind >= 0`` branch guards.
    rama_e = rama_score(
        phi,
        psi,
        aa_rama,
        rama_table,
        rama_bbstart,
        rama_bbstep,
    )
    rama_e = jnp.where(bb_ok, rama_e, jnp.zeros_like(rama_e))

    # Omega branching mirrors upstream:
    #   * valid phipsi + bb-dep table available -> bb-dep Gaussian
    #   * otherwise (e.g. N-terminal residue where phi is undefined, or
    #     a residue without a tabulated omega table) -> simple harmonic
    #     ``omega_V_dV(omega, 32.8)`` as a fall back. The fall back's
    #     offset wrap matches ``if omega > pi/2: omega -= pi; elif
    #     omega < -pi/2: omega += pi`` so trans omegas near +/- pi
    #     contribute near-zero energy rather than the huge ``K*pi^2``
    #     that a naive ``K * omega^2`` would yield.
    bb_dep_ok = jnp.logical_and(bb_ok, aa_omega >= 0)
    omega_e_bbdep = omega_score(phi, psi, omega, aa_omega, omega_params)

    half_pi = 0.5 * jnp.pi
    omega_offset = omega
    omega_offset = jnp.where(
        omega_offset > half_pi, omega_offset - jnp.pi, omega_offset
    )
    omega_offset = jnp.where(
        omega_offset < -half_pi, omega_offset + jnp.pi, omega_offset
    )
    omega_e_simple = 32.8 * omega_offset * omega_offset

    omega_e = jnp.where(bb_dep_ok, omega_e_bbdep, omega_e_simple)
    omega_e = jnp.where(omega_ok, omega_e, jnp.zeros_like(omega_e))

    return rama_e, omega_e


# --------------------------------------------------------------------------- #
# EnergyTerm                                                                  #
# --------------------------------------------------------------------------- #


class BackboneTorsionEnergyTerm(EnergyTerm):
    """JAX energy term computing :data:`ScoreType.rama` and
    :data:`ScoreType.omega`.

    Parameters
    ----------
    param_db
        :class:`tmol_jax.database.ParameterDatabase` (or any object
        exposing ``scoring.rama`` and ``scoring.omega_bbdep``).
    """

    def __init__(self, param_db: Any):
        super().__init__(param_db=param_db)

        self.rama_access = RamaDatabaseAccess(param_db.scoring.rama)
        self.omega_access = OmegaBBDepDatabaseAccess(param_db.scoring.omega_bbdep)

    # ------------------------------------------------------------------ #
    # Metadata                                                           #
    # ------------------------------------------------------------------ #

    def score_types_provided(self) -> list[ScoreType]:
        return [ScoreType.rama, ScoreType.omega]

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_idx_with_prepro(
        pose_stack: Any,
        block_type_ind_np: np.ndarray,
        per_block_aa_table2: np.ndarray,
        per_block_up_cind: np.ndarray,
        per_block_is_pro: np.ndarray,
    ) -> np.ndarray:
        """Vectorised resolve of per-residue table indices honouring the
        prepro vs non-prepro split (used for both rama and omega).

        For each residue we look up the upper-neighbour block (via the
        pose's ``inter_residue_connections``), inspect its
        ``is_pro`` flag, and pick column 0 or 1 of the 2-col per-block
        table mapping. Residues with no resolvable upper neighbour
        (terminal, gaps) fall back to the non-prepro column.
        """

        n_poses, max_n_blocks = block_type_ind_np.shape
        inter_conn = np.asarray(pose_stack.inter_residue_connections, dtype=np.int32)

        real = block_type_ind_np >= 0
        safe_bti = np.where(real, block_type_ind_np, 0)
        up_cind = per_block_up_cind[safe_bti]  # [P, B]

        P_ar = np.arange(n_poses)[:, None]
        B_ar = np.arange(max_n_blocks)[None, :]
        safe_up = np.where(up_cind >= 0, up_cind, 0)
        next_entry = inter_conn[P_ar, B_ar, safe_up]
        next_b = next_entry[..., 0]
        next_valid = real & (up_cind >= 0) & (next_b >= 0)
        safe_next_b = np.where(next_valid, next_b, 0)
        next_bt = block_type_ind_np[P_ar, safe_next_b]
        next_valid = next_valid & (next_bt >= 0)
        safe_next_bt = np.where(next_valid, next_bt, 0)

        is_next_pro = np.where(next_valid, per_block_is_pro[safe_next_bt], 0).astype(
            np.int32
        )

        out = np.where(
            real,
            per_block_aa_table2[safe_bti, is_next_pro],
            -1,
        ).astype(np.int32)
        return out

    # ------------------------------------------------------------------ #
    # Setup                                                              #
    # ------------------------------------------------------------------ #

    def setup_for_pose_stack(
        self, pose_stack: Any, pbt: Any = None
    ) -> BackboneTorsionTermState:
        if pbt is None:
            pbt = pose_stack.packed_block_types

        active = pbt.active_block_types
        n_types = len(active)

        per_block_atom_N = np.full((n_types,), -1, dtype=np.int32)
        per_block_atom_CA = np.full((n_types,), -1, dtype=np.int32)
        per_block_atom_C = np.full((n_types,), -1, dtype=np.int32)
        per_block_down_cind = np.full((n_types,), -1, dtype=np.int32)
        per_block_up_cind = np.full((n_types,), -1, dtype=np.int32)
        per_block_is_pro = np.zeros((n_types,), dtype=np.int32)
        # Upstream indexes both rama and omega tables as
        # ``block_type_<name>_table[bt][upper_nbr_is_pro]``; we build the
        # same 2-column mapping (column 0: non-prepro, column 1: prepro).
        per_block_aa_rama2 = np.full((n_types, 2), -1, dtype=np.int32)
        per_block_aa_omega2 = np.full((n_types, 2), -1, dtype=np.int32)

        for i, bt in enumerate(active):
            per_block_atom_N[i] = _find_atom_idx(bt, "N")
            per_block_atom_CA[i] = _find_atom_idx(bt, "CA")
            per_block_atom_C[i] = _find_atom_idx(bt, "C")
            per_block_down_cind[i] = _connection_index(bt, "down")
            per_block_up_cind[i] = _connection_index(bt, "up")
            per_block_is_pro[i] = 1 if bt.rama_reference == "PRO" else 0

            reference = bt.rama_reference
            per_block_aa_rama2[i] = self.rama_access.reference_to_table_idx_2.get(
                reference, (-1, -1)
            )
            per_block_aa_omega2[i] = self.omega_access.reference_to_table_idx_2.get(
                reference, (-1, -1)
            )
        phi_idx_np, psi_idx_np, omega_idx_np = _resolve_torsion_atoms(
            pose_stack,
            pbt,
            per_block_atom_N,
            per_block_atom_CA,
            per_block_atom_C,
            per_block_down_cind,
            per_block_up_cind,
        )

        # Per-residue AA indices are keyed off the pose's block_type_ind.
        block_type_ind_np = np.asarray(pose_stack.block_type_ind, dtype=np.int32)

        # Both rama and omega resolve the upper neighbour via the pose's
        # inter-residue connections and pick the prepro vs non-prepro
        # column of the 2-col mapping accordingly. This mirrors
        # upstream's ``block_type_<name>_table[bt][upper_nbr_is_pro]``
        # indexing pattern used in
        # ``backbone_torsion_pose_score.impl.hh``.
        aa_rama_idx_np = self._resolve_idx_with_prepro(
            pose_stack,
            block_type_ind_np,
            per_block_aa_rama2,
            per_block_up_cind,
            per_block_is_pro,
        )
        aa_omega_idx_np = self._resolve_idx_with_prepro(
            pose_stack,
            block_type_ind_np,
            per_block_aa_omega2,
            per_block_up_cind,
            per_block_is_pro,
        )

        return BackboneTorsionTermState(
            phi_atom_idx=jnp.asarray(phi_idx_np, dtype=jnp.int32),
            psi_atom_idx=jnp.asarray(psi_idx_np, dtype=jnp.int32),
            omega_atom_idx=jnp.asarray(omega_idx_np, dtype=jnp.int32),
            aa_rama_idx=jnp.asarray(aa_rama_idx_np, dtype=jnp.int32),
            aa_omega_idx=jnp.asarray(aa_omega_idx_np, dtype=jnp.int32),
            rama_table=self.rama_access.rama_table,
            rama_bbstart=self.rama_access.bbstart,
            rama_bbstep=self.rama_access.bbstep,
            omega_params=self.omega_access.params,
        )

    # ------------------------------------------------------------------ #
    # Evaluation                                                         #
    # ------------------------------------------------------------------ #

    def __call__(
        self,
        pose_stack: Any,
        term_state: BackboneTorsionTermState,
        coords_override: Optional[jax.Array] = None,
    ) -> dict[ScoreType, jax.Array]:
        coords = coords_override if coords_override is not None else pose_stack.coords
        # coords: float[n_poses, max_n_atoms_per_pose, 3]

        # Fast path: if either table payload is empty we can skip the
        # whole computation and return zeros. This also covers the
        # "no torch, no zip" degraded-database case.
        n_poses = term_state.phi_atom_idx.shape[0]
        empty_rama = term_state.rama_table.shape[0] == 0
        empty_omega = term_state.omega_params.k_table.shape[0] == 0
        if empty_rama and empty_omega:
            zeros = jnp.zeros((n_poses,), dtype=jnp.float32)
            return {ScoreType.rama: zeros, ScoreType.omega: zeros}

        # vmap over residues, then vmap over poses. `coords` is indexed
        # per pose so we pass `coords[pose_idx]` inside the per-pose
        # vmap.
        per_residue = jax.vmap(
            _residue_energies,
            in_axes=(None, 0, 0, 0, 0, 0, None, None, None, None),
        )

        def _per_pose(
            coords_pose,
            phi_idx,  # [max_n_blocks, 4]
            psi_idx,
            omega_idx,
            aa_rama,  # [max_n_blocks]
            aa_omega,
        ):
            rama_vec, omega_vec = per_residue(
                coords_pose,
                phi_idx,
                psi_idx,
                omega_idx,
                aa_rama,
                aa_omega,
                term_state.rama_table,
                term_state.rama_bbstart,
                term_state.rama_bbstep,
                term_state.omega_params,
            )
            return jnp.sum(rama_vec), jnp.sum(omega_vec)

        rama_per_pose, omega_per_pose = jax.vmap(
            _per_pose,
            in_axes=(0, 0, 0, 0, 0, 0),
        )(
            coords,
            term_state.phi_atom_idx,
            term_state.psi_atom_idx,
            term_state.omega_atom_idx,
            term_state.aa_rama_idx,
            term_state.aa_omega_idx,
        )

        return {
            ScoreType.rama: rama_per_pose.astype(jnp.float32),
            ScoreType.omega: omega_per_pose.astype(jnp.float32),
        }


__all__ = [
    "BackboneTorsionEnergyTerm",
    "BackboneTorsionTermState",
]
