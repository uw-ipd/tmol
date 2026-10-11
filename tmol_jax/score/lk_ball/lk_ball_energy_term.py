"""Differentiable LK-ball water placement and anisotropic solvation."""

from __future__ import annotations

from functools import partial as _partial

from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.score.common.count_pair import pairwise_bonded_path_length

from tmol_jax.score.energy_term import EnergyTerm
from tmol_jax.score.lk_ball.potentials import (
    HYBRIDIZATION_NONE,
    HYBRIDIZATION_RING,
    HYBRIDIZATION_SP2,
    HYBRIDIZATION_SP3,
    LK_BALL_DEFAULTS,
    lk_ball_pair,
    place_waters_for_polar_atom,
)
from tmol_jax.score.score_types import ScoreType
from tmol_jax.types import pytree_dataclass

_HYB_NAME_TO_INT = {
    None: HYBRIDIZATION_NONE,
    "sp2": HYBRIDIZATION_SP2,
    "sp3": HYBRIDIZATION_SP3,
    "ring": HYBRIDIZATION_RING,
}


# --------------------------------------------------------------------------- #
# Fused palette-level per-pose scatter                                        #
# --------------------------------------------------------------------------- #


@_partial(jax.jit, static_argnames=("max_n_atoms_per_pose",))
def _lkball_pose_palette_jit(
    block_type_ind: jax.Array,  # [P, B] int32
    block_coord_offset: jax.Array,  # [P, B] int32
    n_atoms_per_type: jax.Array,  # [T] int32
    lj_radius_table: jax.Array,
    lk_dgfree_table: jax.Array,
    lk_lambda_table: jax.Array,
    lk_volume_table: jax.Array,
    is_donor_table: jax.Array,
    is_acc_table: jax.Array,
    is_hxl_table: jax.Array,
    is_ph_table: jax.Array,
    is_hydrogen_table: jax.Array,
    B_local_table: jax.Array,
    B0_local_table: jax.Array,
    acc_hyb_table: jax.Array,
    H_local_table: jax.Array,
    max_n_atoms_per_pose: int,
) -> Tuple[jax.Array, ...]:
    """Fused LK-ball palette gather — one jit trace replaces ~10 separate
    scatter calls.

    Every per-BT palette table that LK-ball needs is gathered into the
    pose-atom layout inside a single jit-compiled graph. On 1UBQ-class
    poses this takes ~100us after the trace is cached, down from ~3 ms
    for the per-call approach.
    """
    from tmol_jax.score.common.count_pair import pose_atom_block_index

    block_for_atom, local_for_atom, atom_is_real = pose_atom_block_index(
        block_type_ind,
        block_coord_offset,
        n_atoms_per_type,
        max_n_atoms_per_pose,
    )
    P = block_type_ind.shape[0]
    safe_bti = jnp.where(block_type_ind >= 0, block_type_ind, 0)
    P_ar = jnp.arange(P)[:, None]
    bti_of_atom = safe_bti[P_ar, block_for_atom]  # [P, N]
    block_offset = block_coord_offset[P_ar, block_for_atom]  # [P, N]

    # Simple per-atom scatters. Masking padding to 0 / False preserves
    # the semantics of ``scatter_per_atom(..., default=0)``.
    def _s_f(table):
        return jnp.where(atom_is_real, table[bti_of_atom, local_for_atom], 0.0)

    def _s_b(table):
        return jnp.where(atom_is_real, table[bti_of_atom, local_for_atom], False)

    lj_radius = _s_f(lj_radius_table)
    lk_dgfree = _s_f(lk_dgfree_table)
    lk_lambda = _s_f(lk_lambda_table)
    lk_volume = _s_f(lk_volume_table)
    is_donor = _s_b(is_donor_table)
    is_acc = _s_b(is_acc_table)
    is_hxl = _s_b(is_hxl_table)
    is_ph = _s_b(is_ph_table)
    is_hydrogen = _s_b(is_hydrogen_table)

    heavy = jnp.logical_not(is_hydrogen)
    is_polar = jnp.logical_and(heavy, jnp.logical_or(is_donor, is_acc))

    # Acceptor bases + donor Hs.
    B_loc = B_local_table[bti_of_atom, local_for_atom]
    B0_loc = B0_local_table[bti_of_atom, local_for_atom]
    acc_hyb_pose = acc_hyb_table[bti_of_atom, local_for_atom]
    H_loc = H_local_table[bti_of_atom, local_for_atom]

    B_valid = (B_loc >= 0) & atom_is_real
    B0_valid = (B0_loc >= 0) & atom_is_real
    H_valid = (H_loc >= 0) & atom_is_real[:, :, None]
    B_idx = jnp.where(B_valid, block_offset + jnp.where(B_valid, B_loc, 0), -1)
    B0_idx = jnp.where(B0_valid, block_offset + jnp.where(B0_valid, B0_loc, 0), -1)
    att_H_idx = jnp.where(
        H_valid,
        block_offset[:, :, None] + jnp.where(H_valid, H_loc, 0),
        -1,
    )
    acc_hyb_out = jnp.where(atom_is_real, acc_hyb_pose, 0)
    n_H_out = jnp.sum(H_valid.astype(jnp.int32), axis=-1)

    return (
        lj_radius,
        lk_dgfree,
        lk_lambda,
        lk_volume,
        is_donor,
        is_acc,
        is_hxl,
        is_ph,
        is_polar,
        heavy,
        atom_is_real,
        B_idx.astype(jnp.int32),
        B0_idx.astype(jnp.int32),
        acc_hyb_out.astype(jnp.int32),
        att_H_idx.astype(jnp.int32),
        H_valid,
        n_H_out,
    )


# --------------------------------------------------------------------------- #
# Term state                                                                   #
# --------------------------------------------------------------------------- #


@pytree_dataclass
class LKBallTermState:
    """Precomputed per-pose-stack state needed by the hot-path.

    All integer arrays encode palette-level metadata (polar indices,
    neighbor indices, LK params by atom) broadcast onto the pose-stack
    coordinate layout. Float arrays are per-atom LK parameters.
    """

    lj_radius: jax.Array
    lk_dgfree: jax.Array
    lk_lambda: jax.Array
    lk_volume: jax.Array
    is_donor: jax.Array
    is_acceptor: jax.Array
    is_hydroxyl: jax.Array
    is_polarh: jax.Array

    # Polar-atom mask, [n_poses, max_n_atoms_per_pose].
    is_polar: jax.Array
    # Heavy-occluder mask (polars + other heavy atoms),
    # [n_poses, max_n_atoms_per_pose].
    is_occluder: jax.Array
    # Atom realness mask (for padding).
    is_real: jax.Array

    # ---- Upstream-faithful water-placement metadata --------------------- #
    # Per-polar-atom acceptor base (B) and base-of-base (B0) atom indices
    # (pose-level). ``-1`` when the atom is not an acceptor. Shape
    # ``[n_poses, max_n_atoms_per_pose]``.
    B_idx: jax.Array
    B0_idx: jax.Array
    # Per-polar-atom acceptor hybridization code (0=none, 1=sp2, 2=sp3,
    # 3=ring). Shape ``[n_poses, max_n_atoms_per_pose]``.
    acc_hybridization: jax.Array
    # Per-polar-atom attached-polar-H indices (pose-level), padded with
    # -1; shape ``[n_poses, max_n_atoms_per_pose, max_H_per_donor]``.
    attached_H_idx: jax.Array
    # Bool mask for the above. Shape ``[n_poses, max_n_atoms_per_pose,
    # max_H_per_donor]``.
    attached_H_mask: jax.Array
    # Per-polar-atom number of attached polar Hs (scalar per atom;
    # int32). Waters 0..n_attached_H-1 are donor waters; later slots are
    # acceptor waters.
    n_attached_H: jax.Array

    # ---- Count-pair (intra-residue) bonded path distance --------------- #
    # Pairwise shortest-path distance along bonds, clamped at ``>6``
    # (``cp_clamp``). Shape ``[n_poses, max_n_atoms_per_pose,
    # max_n_atoms_per_pose]`` int32. Only intra-block distances are
    # filled; inter-block pairs are set to the clamp value (so the
    # ``bpl > 4`` count-pair weight gate returns 1.0 for them).
    bonded_path_length: jax.Array
    polar_indices: jax.Array
    occluder_indices: jax.Array


class LKBallEnergyTerm(EnergyTerm):
    """Anisotropic solvation; incomplete chemical metadata is an error."""

    def __init__(
        self, param_db, defaults=LK_BALL_DEFAULTS, n_waters=4, max_H_per_donor=3
    ):
        super().__init__(param_db)
        self.defaults = defaults
        self.n_waters = int(n_waters)
        self.max_H_per_donor = int(max_H_per_donor)

    # ------------------------------------------------------------------ #
    # Metadata                                                            #
    # ------------------------------------------------------------------ #

    def score_types_provided(self) -> List[ScoreType]:
        return [
            ScoreType.lk_ball_iso,
            ScoreType.lk_ball,
            ScoreType.lk_bridge,
            ScoreType.lk_bridge_uncpl,
        ]

    # ------------------------------------------------------------------ #
    # Setup                                                               #
    # ------------------------------------------------------------------ #

    def setup_for_pose_stack(self, pose_stack, pbt=None):
        """Build coordinate-independent solvation metadata."""
        pbt = pose_stack.packed_block_types if pbt is None else pbt
        (
            lj_radius_2d,
            lk_dgfree_2d,
            lk_lambda_2d,
            lk_volume_2d,
            is_donor_2d,
            is_acc_2d,
            is_hxl_2d,
            is_ph_2d,
            is_polar_2d,
            is_occluder_2d,
            is_real_2d,
            B_idx_2d,
            B0_idx_2d,
            acc_hyb_2d,
            att_H_idx_3d,
            att_H_mask_3d,
            n_att_H_2d,
        ) = self._per_atom_full_palette(pose_stack, pbt, self.max_H_per_donor)
        bpl_3d = self._per_atom_bonded_path_length(pose_stack, pbt)

        def compact(mask):
            mask = np.asarray(mask)
            width = max(1, int(mask.sum(axis=1).max()))
            indices = np.full((mask.shape[0], width), -1, dtype=np.int32)
            for pose, row in enumerate(mask):
                active = np.flatnonzero(row)
                indices[pose, : len(active)] = active
            return jnp.asarray(indices)

        return LKBallTermState(
            lj_radius=lj_radius_2d,
            lk_dgfree=lk_dgfree_2d,
            lk_lambda=lk_lambda_2d,
            lk_volume=lk_volume_2d,
            is_donor=is_donor_2d.astype(jnp.float32),
            is_acceptor=is_acc_2d.astype(jnp.float32),
            is_hydroxyl=is_hxl_2d.astype(jnp.float32),
            is_polarh=is_ph_2d.astype(jnp.float32),
            is_polar=is_polar_2d,
            is_occluder=is_occluder_2d,
            is_real=is_real_2d,
            B_idx=B_idx_2d,
            B0_idx=B0_idx_2d,
            acc_hybridization=acc_hyb_2d,
            attached_H_idx=att_H_idx_3d,
            attached_H_mask=att_H_mask_3d,
            n_attached_H=n_att_H_2d,
            bonded_path_length=bpl_3d,
            polar_indices=compact(is_polar_2d & is_real_2d),
            occluder_indices=compact(is_occluder_2d & is_real_2d),
        )

    # ------------------------------------------------------------------ #
    # Hot path                                                            #
    # ------------------------------------------------------------------ #

    def __call__(
        self,
        pose_stack: Any,
        term_state: LKBallTermState,
        coords_override: Optional[jax.Array] = None,
    ) -> dict:
        """Evaluate the four LK-ball channels for every pose."""

        coords = pose_stack.coords if coords_override is None else coords_override
        return self._eval_full(coords, term_state)

    def _eval_full(self, coords: jax.Array, state: LKBallTermState) -> dict:
        """Full (upstream-faithful) lk-ball over all atom pairs."""

        def eval_one_pose(pose_coords, pose_state):
            waters = self._place_all_waters(
                pose_coords,
                pose_state["is_donor"],
                pose_state["is_acceptor"],
                pose_state["is_polar"],
                pose_state["B_idx"],
                pose_state["B0_idx"],
                pose_state["acc_hybridization"],
                pose_state["attached_H_idx"],
                pose_state["attached_H_mask"],
                pose_state["n_attached_H"],
            )
            return self._score_all_pairs(pose_coords, waters, pose_state)

        # Extract per-pose slices of term state.
        per_pose = {
            "lj_radius": state.lj_radius,
            "lk_dgfree": state.lk_dgfree,
            "lk_lambda": state.lk_lambda,
            "lk_volume": state.lk_volume,
            "is_donor": state.is_donor,
            "is_acceptor": state.is_acceptor,
            "is_hydroxyl": state.is_hydroxyl,
            "is_polarh": state.is_polarh,
            "is_polar": state.is_polar,
            "is_occluder": state.is_occluder,
            "is_real": state.is_real,
            "B_idx": state.B_idx,
            "B0_idx": state.B0_idx,
            "acc_hybridization": state.acc_hybridization,
            "attached_H_idx": state.attached_H_idx,
            "attached_H_mask": state.attached_H_mask,
            "n_attached_H": state.n_attached_H,
            "bonded_path_length": state.bonded_path_length,
            "polar_indices": state.polar_indices,
            "occluder_indices": state.occluder_indices,
        }

        totals = jax.vmap(eval_one_pose)(coords, per_pose)
        # totals: [n_poses, 4]

        return {
            ScoreType.lk_ball_iso: totals[:, 0],
            ScoreType.lk_ball: totals[:, 1],
            ScoreType.lk_bridge: totals[:, 2],
            ScoreType.lk_bridge_uncpl: totals[:, 3],
        }

    def _place_all_waters(
        self,
        coords: jax.Array,
        is_donor: jax.Array,
        is_acceptor: jax.Array,
        is_polar: jax.Array,
        B_idx: jax.Array,
        B0_idx: jax.Array,
        acc_hybridization: jax.Array,
        attached_H_idx: jax.Array,
        attached_H_mask: jax.Array,
        n_attached_H: jax.Array,
    ) -> jax.Array:
        """Upstream-faithful water placement for every atom in a pose.

        Dispatches on (is_donor, is_acceptor, hybridization) via
        :func:`place_waters_for_polar_atom`. Non-polar atoms get NaN
        water coords (the upstream sentinel used by :func:`lk_ball_pair`
        to early-exit).
        """

        n_atoms = coords.shape[0]

        # Gather acceptor base coordinates via ``take_along_axis``-like
        # safe gather. -1 is clipped to 0; we zero-out downstream via the
        # acceptor-flag gate in ``place_waters_for_polar_atom``.
        safe_B = jnp.where(B_idx >= 0, B_idx, 0)
        safe_B0 = jnp.where(B0_idx >= 0, B0_idx, 0)
        B_coords = coords[safe_B]  # [N, 3]
        B0_coords = coords[safe_B0]  # [N, 3]

        # Attached-H coordinates: [N, max_H, 3], gathered per-atom.
        safe_H = jnp.where(attached_H_idx >= 0, attached_H_idx, 0)
        H_coords = coords[safe_H]  # [N, max_H, 3]

        def place_one(i):
            return place_waters_for_polar_atom(
                coord=coords[i],
                is_donor=is_donor[i],
                is_acceptor=is_acceptor[i],
                hybridization=acc_hybridization[i],
                n_attached_H=n_attached_H[i],
                H_coords=H_coords[i],
                H_mask=attached_H_mask[i],
                B_xyz=B_coords[i],
                B0_xyz=B0_coords[i],
                n_waters=self.n_waters,
                water_dist=self.defaults.lkb_water_dist,
            )

        waters = jax.vmap(place_one)(jnp.arange(n_atoms))  # [N, n_waters, 3]

        # For non-polar atoms, mark waters NaN so the pair routine gates them out.
        polar_f = is_polar[:, None, None].astype(jnp.float32)
        nan_waters = jnp.full_like(waters, jnp.nan)
        return jnp.where(polar_f > 0.5, waters, nan_waters)

    def _score_all_pairs(
        self,
        coords: jax.Array,
        waters: jax.Array,
        state: dict,
    ) -> jax.Array:
        """Sum the four channels across all polar×occluder pairs.

        Returns ``[4]`` scalars = (lk_iso, lk_ball, lk_bridge,
        lk_bridge_uncpl) for this pose.
        """

        def pair(i, j):
            params = {
                "i_lj_radius": state["lj_radius"][i],
                "j_lj_radius": state["lj_radius"][j],
                "i_lk_dgfree": state["lk_dgfree"][i],
                "i_lk_lambda": state["lk_lambda"][i],
                "j_lk_volume": state["lk_volume"][j],
                "i_is_donor": state["is_donor"][i],
                "j_is_donor": state["is_donor"][j],
                "i_is_hydroxyl": state["is_hydroxyl"][i],
                "j_is_hydroxyl": state["is_hydroxyl"][j],
                "i_is_polarh": state["is_polarh"][i],
                "j_is_polarh": state["is_polarh"][j],
                "i_is_acceptor": state["is_acceptor"][i],
                "j_is_acceptor": state["is_acceptor"][j],
            }
            bpl = state["bonded_path_length"][i, j]
            res = lk_ball_pair(
                coord_i=coords[i],
                waters_i=waters[i],
                coord_j=coords[j],
                waters_j=waters[j],
                params=params,
                defaults=self.defaults,
                bonded_path_length=bpl,
            )
            # Mask: only score when i is polar, j is an occluder, i != j,
            # and both atoms are real.
            gate = (
                (i >= 0)
                & (j >= 0)
                & (i != j)
                & state["is_real"][i]
                & state["is_real"][j]
            )
            g = gate.astype(jnp.float32)
            return jnp.stack(
                [
                    g * res.lk_iso,
                    g * res.lk_ball,
                    g * res.lk_bridge,
                    g * res.lk_bridge_uncpl,
                ]
            )

        ii, jj = jnp.meshgrid(
            state["polar_indices"], state["occluder_indices"], indexing="ij"
        )
        per_pair = jax.vmap(jax.vmap(pair))(ii, jj)  # [polar atoms, heavy atoms, 4]
        return jnp.sum(per_pair, axis=(0, 1))  # [4]

    # ------------------------------------------------------------------ #
    # Palette metadata extraction                                         #
    # ------------------------------------------------------------------ #

    def _annotate_pbt_ljlk_tables(self, pbt: Any) -> None:
        """Cache per-BT LJLK palette tables on ``pbt`` once.

        After this call ``pbt`` exposes four per-(BT, atom) float32 tables
        (``lkball_bt_lj_radius``, ``lkball_bt_lk_dgfree``,
        ``lkball_bt_lk_lambda``, ``lkball_bt_lk_volume``) ready for the
        pure-JAX scatter in :func:`_lkball_pose_palette_jit`.
        """
        if hasattr(pbt, "lkball_bt_lj_radius"):
            return
        try:
            ljlk = self.param_db.scoring.ljlk
        except AttributeError as e:
            raise _LKBallSetupUnavailable("param_db.scoring.ljlk not available") from e

        type_params = list(ljlk.atom_type_parameters)
        type_names = [p.name for p in type_params]
        name_to_idx = {n: i for i, n in enumerate(type_names)}

        lj_radius_table = np.asarray(
            [p.lj_radius for p in type_params], dtype=np.float32
        )
        lk_dgfree_table = np.asarray(
            [p.lk_dgfree for p in type_params], dtype=np.float32
        )
        lk_lambda_table = np.asarray(
            [p.lk_lambda for p in type_params], dtype=np.float32
        )
        lk_volume_table = np.asarray(
            [p.lk_volume for p in type_params], dtype=np.float32
        )

        chem_db = pbt.chem_db
        chem_atom_type_names = [at.name for at in chem_db.atom_types]
        chem_to_ljlk = np.asarray(
            [name_to_idx.get(n, -1) for n in chem_atom_type_names],
            dtype=np.int32,
        )
        atom_types_np = np.asarray(pbt.atom_types)  # [n_types, max_n_atoms_per_block]
        ljlk_idx_np = np.where(
            atom_types_np >= 0, chem_to_ljlk[np.clip(atom_types_np, 0, None)], -1
        )
        safe_idx = np.clip(ljlk_idx_np, 0, None)

        def _gather(tab):
            out = tab[safe_idx]
            return np.where(ljlk_idx_np >= 0, out, 0.0).astype(np.float32)

        setattr(pbt, "lkball_bt_lj_radius", jnp.asarray(_gather(lj_radius_table)))
        setattr(pbt, "lkball_bt_lk_dgfree", jnp.asarray(_gather(lk_dgfree_table)))
        setattr(pbt, "lkball_bt_lk_lambda", jnp.asarray(_gather(lk_lambda_table)))
        setattr(pbt, "lkball_bt_lk_volume", jnp.asarray(_gather(lk_volume_table)))

    def _annotate_pbt_polar_flags(self, pbt: Any) -> None:
        """Cache per-BT polar-flag tables on ``pbt`` once.

        Produces four bool ``[n_types, max_n_atoms]`` arrays
        (donor / acceptor / hydroxyl / polarh) plus ``is_hydrogen`` for
        the heavy-atom gate. Everything depends only on the palette +
        chemical DB, not on pose layout.
        """
        if hasattr(pbt, "lkball_bt_is_donor"):
            return
        chem_db = pbt.chem_db
        chem_is_donor = np.asarray(
            [at.is_donor for at in chem_db.atom_types], dtype=bool
        )
        chem_is_acc = np.asarray(
            [at.is_acceptor for at in chem_db.atom_types], dtype=bool
        )
        chem_is_hxl = np.asarray(
            [at.is_hydroxyl for at in chem_db.atom_types], dtype=bool
        )
        chem_is_ph = np.asarray([at.is_polarh for at in chem_db.atom_types], dtype=bool)

        atom_types_np = np.asarray(pbt.atom_types)
        safe_types = np.clip(atom_types_np, 0, None)
        real = atom_types_np >= 0

        def _g(tab):
            out = tab[safe_types]
            return np.where(real, out, False)

        setattr(pbt, "lkball_bt_is_donor", jnp.asarray(_g(chem_is_donor)))
        setattr(pbt, "lkball_bt_is_acc", jnp.asarray(_g(chem_is_acc)))
        setattr(pbt, "lkball_bt_is_hxl", jnp.asarray(_g(chem_is_hxl)))
        setattr(pbt, "lkball_bt_is_ph", jnp.asarray(_g(chem_is_ph)))
        is_hydrogen_per_block = np.asarray(pbt.atom_is_hydrogen, dtype=bool)
        setattr(
            pbt,
            "lkball_bt_is_hydrogen",
            jnp.asarray(is_hydrogen_per_block),
        )

    def _annotate_pbt_acceptor_bases(self, pbt: Any, max_H_per_donor: int) -> None:
        """Cache per-BT acceptor-base / donor-H tables on ``pbt``.

        These are a pure function of the palette: each block type's
        intra-residue bond graph and per-atom-type flags determine (for
        every atom) its acceptor B/B0 (local indices) and its attached
        polar hydrogen list. The scatter into per-pose coords is done
        below with pure-JAX gathers — no Python loops on the hot path.

        Attached attributes on ``pbt`` (all ``int32``):

        * ``lkball_bt_B_local``   ``[n_types, max_atoms_block]``
        * ``lkball_bt_B0_local``  ``[n_types, max_atoms_block]``
        * ``lkball_bt_acc_hyb``   ``[n_types, max_atoms_block]``
        * ``lkball_bt_H_local``   ``[n_types, max_atoms_block, max_H_per_donor]``
        * ``lkball_bt_n_H``       ``[n_types, max_atoms_block]``
        """
        attr = f"lkball_bt_B_local_k{max_H_per_donor}"
        if hasattr(pbt, attr):
            return

        chem_db = pbt.chem_db
        atom_types_list = list(chem_db.atom_types)
        atype_is_donor = np.asarray([a.is_donor for a in atom_types_list], dtype=bool)
        atype_is_acceptor = np.asarray(
            [a.is_acceptor for a in atom_types_list], dtype=bool
        )
        atype_is_polarh = np.asarray([a.is_polarh for a in atom_types_list], dtype=bool)
        atype_is_hydrogen = np.asarray(
            [a.element == "H" for a in atom_types_list], dtype=bool
        )
        atype_hybridization = np.asarray(
            [
                _HYB_NAME_TO_INT.get(a.acceptor_hybridization, 0)
                for a in atom_types_list
            ],
            dtype=np.int32,
        )

        n_types = int(pbt.n_types)
        max_atoms_block = int(pbt.max_n_atoms)
        pbt_n_atoms = np.asarray(pbt.n_atoms)
        atom_types_pbt = np.asarray(pbt.atom_types)
        bond_indices_pbt = np.asarray(pbt.bond_indices)
        n_bonds_pbt = np.asarray(pbt.n_bonds)

        B_local = np.full((n_types, max_atoms_block), -1, dtype=np.int32)
        B0_local = np.full((n_types, max_atoms_block), -1, dtype=np.int32)
        acc_hyb = np.zeros((n_types, max_atoms_block), dtype=np.int32)
        H_local = np.full(
            (n_types, max_atoms_block, max_H_per_donor), -1, dtype=np.int32
        )
        n_H = np.zeros((n_types, max_atoms_block), dtype=np.int32)

        for bt in range(n_types):
            n_ats = int(pbt_n_atoms[bt])
            if n_ats <= 0:
                continue
            ats_here = atom_types_pbt[bt, :n_ats]

            # Per-BT adjacency list (intra-block only; inter-block bonds
            # are ignored for LK ball, matching the simplified port).
            n_b = int(n_bonds_pbt[bt])
            adj: list[list[int]] = [[] for _ in range(n_ats)]
            for e in range(n_b):
                a = int(bond_indices_pbt[bt, e, 0])
                b = int(bond_indices_pbt[bt, e, 1])
                if 0 <= a < n_ats and 0 <= b < n_ats:
                    adj[a].append(b)

            for local_i in range(n_ats):
                at_i = int(ats_here[local_i])
                if at_i < 0:
                    continue

                # Donor heavy -> attached polar Hs list.
                if atype_is_donor[at_i] and not atype_is_hydrogen[at_i]:
                    k = 0
                    for nbr in adj[local_i]:
                        at_nbr = int(ats_here[nbr])
                        if at_nbr < 0:
                            continue
                        if atype_is_polarh[at_nbr]:
                            if k < max_H_per_donor:
                                H_local[bt, local_i, k] = nbr
                                k += 1
                    n_H[bt, local_i] = k

                # Acceptor -> (B_local, B0_local) by hybridization rules.
                if atype_is_acceptor[at_i]:
                    hyb = int(atype_hybridization[at_i])
                    acc_hyb[bt, local_i] = hyb

                    bL = -1
                    b0L = -1
                    if hyb == 1:  # sp2
                        for nbr in adj[local_i]:
                            at_nbr = int(ats_here[nbr])
                            if at_nbr < 0:
                                continue
                            if not atype_is_hydrogen[at_nbr]:
                                bL = nbr
                                break
                        if bL >= 0:
                            for nbr2 in adj[bL]:
                                if nbr2 == local_i:
                                    continue
                                at_nbr2 = int(ats_here[nbr2])
                                if at_nbr2 < 0:
                                    continue
                                b0L = nbr2
                                break
                    elif hyb == 2:  # sp3
                        for nbr in adj[local_i]:
                            at_nbr = int(ats_here[nbr])
                            if at_nbr < 0:
                                continue
                            if atype_is_hydrogen[at_nbr]:
                                b0L = nbr
                                break
                        for nbr in adj[local_i]:
                            if nbr == b0L:
                                continue
                            at_nbr = int(ats_here[nbr])
                            if at_nbr < 0:
                                continue
                            bL = nbr
                            break
                    elif hyb == 3:  # ring
                        for nbr in adj[local_i]:
                            at_nbr = int(ats_here[nbr])
                            if at_nbr < 0:
                                continue
                            if atype_is_hydrogen[at_nbr]:
                                continue
                            if bL < 0:
                                bL = nbr
                            elif b0L < 0:
                                b0L = nbr
                                break

                    if bL >= 0 and b0L < 0:
                        b0L = bL
                    B_local[bt, local_i] = bL
                    B0_local[bt, local_i] = b0L

        setattr(pbt, attr, jnp.asarray(B_local, dtype=jnp.int32))
        setattr(
            pbt,
            f"lkball_bt_B0_local_k{max_H_per_donor}",
            jnp.asarray(B0_local, dtype=jnp.int32),
        )
        setattr(
            pbt,
            f"lkball_bt_acc_hyb_k{max_H_per_donor}",
            jnp.asarray(acc_hyb, dtype=jnp.int32),
        )
        setattr(
            pbt,
            f"lkball_bt_H_local_k{max_H_per_donor}",
            jnp.asarray(H_local, dtype=jnp.int32),
        )
        setattr(
            pbt,
            f"lkball_bt_n_H_k{max_H_per_donor}",
            jnp.asarray(n_H, dtype=jnp.int32),
        )

    def _per_atom_bonded_path_length(self, pose_stack: Any, pbt: Any) -> jax.Array:
        """Pose-level pairwise bonded path distance.

        Delegates to :func:`tmol_jax.score.common.count_pair.pairwise_bonded_path_length`,
        which threads intra-residue path distances with
        ``pose_stack.inter_block_bondsep`` through every
        (conn_a, conn_b) pair so cross-residue BPL is correct for
        peptide-bonded atoms (e.g., ``O(res N) ↔ H(res N+1)`` resolves
        to 3 bonds → count-pair weight 0, matching upstream).
        """
        return pairwise_bonded_path_length(pose_stack, pbt)

    def _per_atom_full_palette(
        self,
        pose_stack: Any,
        pbt: Any,
        max_H_per_donor: int,
    ) -> Tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,  # lj/dgfree/lambda/volume
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,  # flags
        jax.Array,  # is_occluder
        jax.Array,  # is_real
        jax.Array,
        jax.Array,
        jax.Array,  # B_idx, B0_idx, acc_hyb
        jax.Array,
        jax.Array,
        jax.Array,  # attached_H_idx, _mask, n_H
    ]:
        """Batched per-pose-atom scatter of all LK-ball palette tables.

        Consolidates every LK-ball palette scatter (simple LJ/LK params,
        polar flags, acceptor bases, donor Hs) into a single jit-compiled
        function :func:`_lkball_pose_palette_jit`. ``max_n_neighbors`` is
        accepted for backwards compatibility but unused.
        """
        # Ensure every palette-level cache is populated. Idempotent.
        self._annotate_pbt_ljlk_tables(pbt)
        self._annotate_pbt_polar_flags(pbt)
        self._annotate_pbt_acceptor_bases(pbt, max_H_per_donor)

        bti = jnp.asarray(pose_stack.block_type_ind, dtype=jnp.int32)
        offsets = jnp.asarray(pose_stack.block_coord_offset, dtype=jnp.int32)
        n_atoms_per_type = jnp.asarray(pbt.n_atoms, dtype=jnp.int32)
        A = int(pose_stack.max_n_atoms_per_pose)

        return _lkball_pose_palette_jit(
            bti,
            offsets,
            n_atoms_per_type,
            pbt.lkball_bt_lj_radius,
            pbt.lkball_bt_lk_dgfree,
            pbt.lkball_bt_lk_lambda,
            pbt.lkball_bt_lk_volume,
            pbt.lkball_bt_is_donor,
            pbt.lkball_bt_is_acc,
            pbt.lkball_bt_is_hxl,
            pbt.lkball_bt_is_ph,
            pbt.lkball_bt_is_hydrogen,
            getattr(pbt, f"lkball_bt_B_local_k{max_H_per_donor}"),
            getattr(pbt, f"lkball_bt_B0_local_k{max_H_per_donor}"),
            getattr(pbt, f"lkball_bt_acc_hyb_k{max_H_per_donor}"),
            getattr(pbt, f"lkball_bt_H_local_k{max_H_per_donor}"),
            A,
        )


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #


class _LKBallSetupUnavailable(Exception):
    """Required LK-ball parameter metadata is unavailable."""


__all__ = [
    "LKBallEnergyTerm",
    "LKBallTermState",
]
