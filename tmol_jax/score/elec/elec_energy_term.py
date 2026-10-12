"""Whole-pose electrostatics scoring term.

Port of :mod:`tmol.score.elec.elec_energy_term` to JAX. The portable path
uses a dense ``[max_n_atoms, max_n_atoms]`` reduction and remains available
on CPU or without the optional CUDA package. Float32 GPU installations may
dispatch the pair sum to a bounded-memory adapter around TMol's canonical,
Torch-independent scalar potential.

The term produces one score channel --
:attr:`tmol_jax.score.score_types.ScoreType.fa_elec` -- and stores all
per-type setup on the packed-block-types instance so repeated
evaluation of the same pose does no extra work.

Setup pipeline (done once on the host, using numpy):

1. Build :class:`ElecGlobalParams` from the database.
2. Per block type: compute partial charges (length ``n_atoms``) and
   the count-pair representative remapping ``repr[j]``.
3. Per block type: ``inter_repr_path_distance[a, b] =
   path_distance[a, repr[b]]`` and
   ``intra_repr_path_distance[a, b] = path_distance[repr[a],
   repr[b]]``. These mirror upstream's
   ``elec_{inter,intra}_repr_path_distance``.
4. Stack those per-type arrays over the packed-block-types palette.

Portable scoring (pure JAX, ``jit``-safe):

1. Gather per-atom charges ``[P, N]`` from ``pbt.elec_partial_charge``.
2. Compute a ``[P, N, N]`` pair distance matrix from coords.
3. Compute a ``[P, N, N]`` chemical-bond path-length matrix by
   combining the intra-residue path distances and the inter-residue
   ``inter_block_bondsep`` -- see :func:`pair_bonded_path_length`.
4. Call :func:`elec_pair` element-wise, mask the diagonal and the
   lower triangle (so each pair contributes once), and sum per pose.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.score.common.count_pair import (
    pair_bonded_path_length_direct,
    pose_atom_real_mask,
)

from tmol_jax.score.elec.params import (
    ElecGlobalParams,
    _cp_reps_dict,
    _partial_charges_dict,
    atom_charges_for_pose_stack,
    elec_global_params_from_database,
    representative_mapping_for_block_type,
)
from tmol_jax.score.elec.potentials import elec_pair
from tmol_jax.score.energy_term import EnergyTerm
from tmol_jax.score.score_types import ScoreType
from tmol_jax.types import pytree_dataclass

# Max chemical-bond separation that matters for count-pair (anything
# larger gets clamped to this; the weight function already returns 1.0
# for > 4 so the clamp just keeps the integer arithmetic small).
_MAX_COUNTPAIR_SEPARATION: int = 6


# --------------------------------------------------------------------- #
# Host-side packed-block-type setup                                      #
# --------------------------------------------------------------------- #


def _annotate_packed_block_types(packed_block_types: Any, elec_db: Any) -> None:
    """Attach the four elec-specific arrays to ``packed_block_types``.

    Idempotent: if the attributes are already present we short-circuit.
    The arrays are ``jnp`` so that later scoring can use them directly
    without host->device transfers.

    Attached attributes:

    ``elec_partial_charge``
        ``float32[n_types, max_n_atoms]``. Zero for non-real atom
        slots and for atoms missing from the DB.

    ``elec_inter_repr_path_distance``
        ``int32[n_types, max_n_atoms, max_n_atoms]``. For a given
        block type ``t``, position ``[a, b]`` holds the number of
        chemical bonds between atom ``a`` on ``t`` and the count-pair
        representative of atom ``b`` on ``t``. Used to compute the
        inter-block count-pair separation by summing through
        connection atoms on each side.

    ``elec_intra_repr_path_distance``
        ``int32[n_types, max_n_atoms, max_n_atoms]``. Holds
        ``path_distance[repr[a], repr[b]]``; used directly as the
        path length for atom pairs on the same block.
    """

    if (
        hasattr(packed_block_types, "elec_partial_charge")
        and hasattr(packed_block_types, "elec_inter_repr_path_distance")
        and hasattr(packed_block_types, "elec_intra_repr_path_distance")
    ):
        return

    partial_charges = _partial_charges_dict(elec_db)
    cp_reps = _cp_reps_dict(elec_db)

    n_types = int(packed_block_types.n_types)
    max_n_atoms = int(packed_block_types.max_n_atoms)

    charges = np.zeros((n_types, max_n_atoms), dtype=np.float32)
    inter_rpd = np.zeros((n_types, max_n_atoms, max_n_atoms), dtype=np.int32)
    intra_rpd = np.zeros((n_types, max_n_atoms, max_n_atoms), dtype=np.int32)

    # Fall back to the PBT-packed path_distance if a block type has no
    # ``path_distance`` attribute of its own.
    packed_pd = np.asarray(packed_block_types.intrares_path_distance)

    for i, bt in enumerate(packed_block_types.active_block_types):
        # --- charges
        from tmol_jax.score.elec.params import _charges_for_block_type

        rc = _charges_for_block_type(bt, partial_charges)
        charges[i, : rc.shape[0]] = rc

        # --- representative mapping and path-distance tables
        rep = representative_mapping_for_block_type(bt, cp_reps)
        n = rep.shape[0]

        pd = getattr(bt, "path_distance", None)
        if pd is None:
            pd = packed_pd[i, :n, :n]
        pd = np.asarray(pd, dtype=np.int32)[:n, :n]

        # inter_rpd[a, b] = pd[a, rep[b]]
        inter_rpd[i, :n, :n] = pd[:, rep]
        # intra_rpd[a, b] = pd[rep[a], rep[b]]
        intra_rpd[i, :n, :n] = pd[rep][:, rep]

    setattr(packed_block_types, "elec_partial_charge", jnp.asarray(charges))
    setattr(
        packed_block_types,
        "elec_inter_repr_path_distance",
        jnp.asarray(inter_rpd),
    )
    setattr(
        packed_block_types,
        "elec_intra_repr_path_distance",
        jnp.asarray(intra_rpd),
    )


# --------------------------------------------------------------------- #
# Per-atom-pair scoring                                                  #
# --------------------------------------------------------------------- #


def pair_bonded_path_length(pose_stack: Any, packed_block_types: Any) -> jax.Array:
    """Compute the ``[n_poses, max_n_atoms_per_pose, max_n_atoms_per_pose]``
    matrix of bonded path lengths.

    Thin wrapper around the jitted
    :func:`~tmol_jax.score.common.count_pair.pair_bonded_path_length_direct`
    — which implements the same no-APSP elec path-length logic (use
    ``inter_block_bondsep`` for the 1-hop connection link, and the
    rep-remapped per-BT tables for each atom's distance to the block's
    connection atom). Pure JAX on-device, so repeated setup calls for
    equivalently-shaped PoseStacks amortise the initial XLA compile.
    """
    return pair_bonded_path_length_direct(
        pose_stack,
        packed_block_types,
        packed_block_types.elec_intra_repr_path_distance,
        packed_block_types.elec_inter_repr_path_distance,
    )


# --------------------------------------------------------------------- #
# Pairwise energy (JAX, ``jit``-safe)                                    #
# --------------------------------------------------------------------- #


def _pairwise_distance(coords: jax.Array) -> jax.Array:
    """Return the ``[P, N, N]`` Euclidean distance matrix for ``coords``.

    Uses the "double-where" idiom so ``jnp.sqrt`` never sees a zero input
    under autograd. Its gradient is ``1 / (2 sqrt(x))`` which is ``inf``
    at ``x == 0``; the outer ``where`` would then multiply ``inf * 0``
    producing ``NaN`` in the cotangent. Masking the square first with a
    safe constant avoids that.
    """
    diff = coords[:, :, None, :] - coords[:, None, :, :]
    dist_sq = jnp.sum(diff * diff, axis=-1)
    safe = dist_sq > 1e-6
    dist = jnp.where(
        safe,
        jnp.sqrt(jnp.where(safe, dist_sq, 1.0)),
        1.0,
    )
    return dist


def _atom_is_real_mask(pose_stack: Any, packed_block_types: Any) -> jax.Array:
    """Boolean ``[P, N]`` mask of real (non-padding) atoms.

    Thin JAX-on-device wrapper around
    :func:`~tmol_jax.score.common.count_pair.pose_atom_real_mask`. Runs
    through ``jit`` so repeated calls with the same pose-stack shape
    amortise the trace.
    """
    bti = jnp.asarray(pose_stack.block_type_ind, dtype=jnp.int32)
    offsets = jnp.asarray(pose_stack.block_coord_offset, dtype=jnp.int32)
    n_atoms_per_type = jnp.asarray(packed_block_types.n_atoms, dtype=jnp.int32)
    return pose_atom_real_mask(
        bti, offsets, n_atoms_per_type, int(pose_stack.coords.shape[1])
    )


def elec_pose_score(
    coords: jax.Array,  # [P, N, 3] float32
    charges: jax.Array,  # [P, N] float32
    bonded_path_length: jax.Array,  # [P, N, N] int32
    real_atom_mask: jax.Array,  # [P, N] bool
    global_params: ElecGlobalParams,
) -> jax.Array:
    """Whole-pose elec score, ``[P]`` float32.

    All inputs are JAX arrays; this function is ``jit``-safe and
    differentiable in ``coords``.
    """

    # Pairwise distances and charges.
    dist = _pairwise_distance(coords)  # [P, N, N]
    e_i = charges[:, :, None]  # [P, N, 1]
    e_j = charges[:, None, :]  # [P, 1, N]

    E_pair = elec_pair(dist, e_i, e_j, bonded_path_length, global_params)

    # Mask: keep only (i < j) with both atoms real. This way each pair
    # contributes exactly once and padding atoms contribute zero.
    n = dist.shape[-1]
    iu = jnp.arange(n)[:, None] < jnp.arange(n)[None, :]
    realm = real_atom_mask[:, :, None] & real_atom_mask[:, None, :]
    mask = iu[None, :, :] & realm
    # Preserve ``E_pair``'s dtype instead of pinning to float32 —
    # see note in ``__call__`` below about float64 gradchecks.
    E_pair = jnp.where(mask, E_pair, jnp.asarray(0.0, dtype=E_pair.dtype))

    return jnp.sum(E_pair, axis=(-1, -2))


# --------------------------------------------------------------------- #
# The EnergyTerm-facing class                                            #
# --------------------------------------------------------------------- #


@pytree_dataclass
class _ElecState:
    """Per-pose-stack precomputed elec state.

    All arrays are shaped to pose-atom layout (``[P, N]`` or
    ``[P, N, N]``). The :class:`ElecGlobalParams` pytree is carried
    alongside so the hot path is fully pure.
    """

    # [P, N] float32 partial charges, zero at padding slots.
    charges: jax.Array
    # [P, N, N] int32 bonded path length; defaults to _MAX_COUNTPAIR_SEPARATION
    # for cross-block pairs that can't be reached within the cutoff.
    bonded_path_length: jax.Array
    # [P, N] bool real-atom mask.
    atom_is_real: jax.Array

    # ElecGlobalParams scalars (kept as jax arrays so the state is a
    # flat pytree).
    D: jax.Array
    D0: jax.Array
    S: jax.Array
    min_dis: jax.Array
    max_dis: jax.Array
    cuda_global_parameters: jax.Array


class ElecEnergyTerm(EnergyTerm):
    """Whole-pose electrostatics energy term.

    Provides :attr:`ScoreType.fa_elec`. The hot path is pure JAX and
    differentiable in the coordinates -- safe under ``jit`` / ``grad``
    / ``vmap``.

    :meth:`setup_for_pose_stack` walks each block type on the host to
    gather per-atom partial charges and the count-pair bonded
    path-length matrix, then returns a :class:`_ElecState` pytree that
    is threaded into :meth:`__call__`.
    """

    def __init__(
        self,
        param_db: Any,
        # 6.0 by default to match LJLK's cutoff; elec's actual max_dis is
        # 5.5 in the default params, but the extra 0.5 A of neighbours just
        # contribute E=0 inside the switch function. Same flag value across
        # terms also lets sfxn-level toggling be one number.
    ):
        # ``param_db`` may be either a ParameterDatabase (preferred, matches
        # the EnergyTerm base) or an ElecDatabase directly (legacy). Retain
        # both shapes so this term slots into the shared ScoreFunction
        # registry while still being usable standalone in tests.
        if hasattr(param_db, "scoring"):
            super().__init__(param_db)
            self.elec_db = param_db.scoring.elec
        else:
            super().__init__(param_db=None)
            self.elec_db = param_db
        self.global_params: ElecGlobalParams = elec_global_params_from_database(
            self.elec_db
        )

    # ---- class-level metadata ---------------------------------------

    @classmethod
    def class_name(cls) -> str:
        return "Elec"

    @classmethod
    def score_types(cls) -> Tuple[ScoreType, ...]:
        return (ScoreType.fa_elec,)

    def score_types_provided(self) -> list[ScoreType]:
        return [ScoreType.fa_elec]

    def n_bodies(self) -> int:
        return 2

    # ---- setup ------------------------------------------------------

    def setup_block_type(self, block_type: Any) -> None:  # noqa: D401 - keep
        """No-op; elec annotations are batched at the PBT level."""
        return

    def setup_packed_block_types(self, packed_block_types: Any) -> None:
        """Annotate the PBT with charges and count-pair path tables.

        Kept public so callers that pre-build a shared PBT palette
        (e.g. the factor-graph minimizer) can hoist this work out of
        the per-pose setup. Idempotent.
        """
        _annotate_packed_block_types(packed_block_types, self.elec_db)

    def setup_for_pose_stack(
        self, pose_stack: Any, pbt: Optional[Any] = None
    ) -> _ElecState:
        """Precompute per-pose charges + path-length matrix + atom mask.

        Returns a :class:`_ElecState` pytree. The setup is conformation-
        independent -- it depends only on the block types laid out on
        the pose, not on the coordinates -- so it can be reused across
        a trajectory of ``coords_override`` values.
        """
        if pbt is None:
            pbt = pose_stack.packed_block_types
        if not hasattr(pbt, "elec_partial_charge"):
            self.setup_packed_block_types(pbt)

        charges = atom_charges_for_pose_stack(pose_stack, pbt, self.elec_db)
        bpl = pair_bonded_path_length(pose_stack, pbt)
        mask = _atom_is_real_mask(pose_stack, pbt)

        gp = self.global_params
        from tmol_jax.score.elec import cuda

        bpl_jax = jnp.asarray(bpl, dtype=jnp.int32)
        mask_jax = jnp.asarray(mask, dtype=bool)

        # Optional cell-list construction (mirror the LJLK pattern). State
        # stays None on the dense path so the pytree has no extra leaves.
        return _ElecState(
            charges=jnp.asarray(charges, dtype=jnp.float32),
            bonded_path_length=bpl_jax,
            atom_is_real=mask_jax,
            D=jnp.asarray(gp.D, dtype=jnp.float32),
            D0=jnp.asarray(gp.D0, dtype=jnp.float32),
            S=jnp.asarray(gp.S, dtype=jnp.float32),
            min_dis=jnp.asarray(gp.min_dis, dtype=jnp.float32),
            max_dis=jnp.asarray(gp.max_dis, dtype=jnp.float32),
            cuda_global_parameters=cuda._pack_global_parameters(gp),
        )

    # ---- scoring ----------------------------------------------------

    def __call__(
        self,
        pose_stack: Any,
        term_state: _ElecState,
        coords_override: Optional[jax.Array] = None,
    ) -> dict[ScoreType, jax.Array]:
        """Return ``{ScoreType.fa_elec: [P]}`` for the given pose stack.

        If ``coords_override`` is supplied, it replaces
        ``pose_stack.coords`` for this evaluation -- used by the
        minimizer so ``jax.grad(fn)(coords)`` can trace cleanly without
        rebuilding a :class:`PoseStack` around the candidate array.
        """

        coords = coords_override if coords_override is not None else pose_stack.coords

        if coords.dtype == jnp.float32 and jax.default_backend() == "gpu":
            from tmol_jax.score.elec import cuda

            if cuda.available():
                return {ScoreType.fa_elec: cuda.score(term_state, coords)}

        # NaN-safe pairwise distances (double-where on the inside of sqrt).
        dist = _pairwise_distance(coords)  # [P, N, N]

        # elec_pair is agnostic to how global_params is presented; the
        # state just needs D / D0 / S / min_dis / max_dis fields, which
        # :class:`_ElecState` exposes directly.
        e_i = term_state.charges[:, :, None]
        e_j = term_state.charges[:, None, :]
        E_pair = elec_pair(dist, e_i, e_j, term_state.bonded_path_length, term_state)

        # Mask: upper triangle (each pair once) with both atoms real.
        n = dist.shape[-1]
        iu = jnp.arange(n)[:, None] < jnp.arange(n)[None, :]
        realm = (
            term_state.atom_is_real[:, :, None] & term_state.atom_is_real[:, None, :]
        )
        mask = iu[None, :, :] & realm
        # Masked zero must match ``E_pair``'s dtype, not be pinned to
        # float32; otherwise a float64 caller (gradchecks) silently
        # loses precision when the where-broadcast promotes ``E_pair``
        # down to float32 on the masked path.
        E_pair = jnp.where(mask, E_pair, jnp.asarray(0.0, dtype=E_pair.dtype))

        # Preserve the working precision rather than casting back to
        # float32. Production callers feed float32 coords and will see
        # float32 scores as before; gradcheck callers feed float64 and
        # get float64 out, which is what the FD loop needs to avoid
        # ~1e-3 truncation noise on ``dE/dx ~ O(1)``.
        score = jnp.sum(E_pair, axis=(-1, -2))
        return {ScoreType.fa_elec: score}


__all__ = [
    "ElecEnergyTerm",
    "elec_pose_score",
    "pair_bonded_path_length",
]
