"""LK-ball potential primitives in pure JAX.

Upstream references:

* ``tmol/score/lk_ball/potentials/lk_ball.hh`` — ``lk_fraction``,
  ``lk_bridge_fraction``, and the main ``lk_ball_score::V`` driver.
* ``tmol/score/lk_ball/potentials/water.hh`` — water placement from
  donor/acceptor geometry.
* ``tmol/score/ljlk/potentials/lk_isotropic.hh`` — ``f_desolv`` (the
  isotropic LK gaussian in distance), reused verbatim as the LK-ball
  magnitude.

The upstream code distinguishes donor-water placement (``build_don_water``:
a water along the D-H axis at a fixed distance) from acceptor-water
placement (``build_water_for_acc``: a coordinate built with a fixed
bond-angle and one of several canonical torsions relative to the
acceptor's hybridization — sp2 / sp3 / ring). Both are implemented
here via :func:`place_waters_for_polar_atom`, dispatching on
``(is_donor, is_acceptor, hybridization)``. The main
:func:`lk_ball_pair` routine is independent of how the waters got
placed and is a faithful port of the upstream algorithm.

All functions are pure JAX, shape-polymorphic, and safe to use under
``jit``, ``vmap``, and ``grad``. Downstream in
:mod:`.lk_ball_energy_term` they are composed via ``vmap`` across the
atom / pose axes.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from tmol_jax.types import Array

# --------------------------------------------------------------------------- #
# NaN-safe primitives                                                         #
# --------------------------------------------------------------------------- #
#
# When this module is composed under ``jax.vmap`` across all atom pairs in a
# pose stack, many inputs land in "padded" or "virtual" slots where the two
# coordinates happen to coincide (or a water is NaN-sentinel-placed for a
# non-polar atom). The raw ``jnp.linalg.norm`` and ``jnp.sqrt`` gradients at
# zero evaluate to ``0/0 = NaN``, which then pollutes gradients even for
# unrelated (real) atom pairs once the backward pass chains them through the
# same outputs.
#
# The canonical fix is the "double-where" trick: inside the pathological
# branch, substitute a safe dummy (``1.0``) for the raw input *before* taking
# the square root, then mask the result back to ``0.0`` outside. This yields
# a finite forward value and a finite (zero) gradient on the masked-off path.


def safe_norm(
    v: Array, axis: int = -1, keepdims: bool = False, eps: float = 1e-8
) -> Array:
    """NaN-safe Euclidean norm along ``axis``.

    Returns ``0`` where ``|v|^2 <= eps``, and the true norm otherwise.
    Gradients are finite everywhere (``0`` on the masked-off branch).
    """
    sq = jnp.sum(v * v, axis=axis, keepdims=keepdims)
    safe = sq > eps
    return jnp.where(safe, jnp.sqrt(jnp.where(safe, sq, 1.0)), 0.0)


def safe_normalize(v: Array, axis: int = -1, eps: float = 1e-8) -> Array:
    """NaN-safe unit-vector along ``axis``.

    Returns the zero vector where ``|v|`` is (approximately) zero; otherwise
    ``v / |v|``. Gradients are finite everywhere.
    """
    n = safe_norm(v, axis=axis, keepdims=True, eps=eps)
    nz = n > 0
    return jnp.where(nz, v / jnp.where(nz, n, 1.0), 0.0)


# --------------------------------------------------------------------------- #
# Constants (upstream ``lkball_globals``, ``LJLKGlobalParams``)               #
# --------------------------------------------------------------------------- #
#
# These are the upstream-hard-coded lk-ball tolerances; upstream marks them
# "could be exposed as global parameters" but they are fixed in every
# production database so we fold them into a frozen dict.  Callers can
# override on a per-call basis via the ``params`` argument to
# :func:`lk_ball_pair`.


class LKBallDefaults(NamedTuple):
    """Default lk-ball tunables shared with upstream tmol."""

    # Water-placement distance (center-of-atom to center-of-water), Å.
    # Upstream: ``LJLKGlobalParams.lkb_water_dist = 2.828``.
    lkb_water_dist: float = 2.828

    # ``lk_fraction`` widths. Upstream: ``lkball_globals``.
    ramp_width_A2: float = 3.709

    # ``lk_bridge_fraction`` widths.
    overlap_gap_A2: float = 0.5
    overlap_width_A2: float = 2.6
    # ``angle_overlap_A2 = 2.8 * overlap_width_A2`` in upstream.
    angle_overlap_A2: float = 2.8 * 2.6

    # Upper heavy-atom distance cutoff, Å (upstream:
    # ``LJLKGlobalParams.max_dis`` / ``LKBallGlobalParams.distance_threshold``).
    # ponytail: upstream commit 1d7e18c now threads this from
    # ``max_dis`` end-to-end. tmol_jax keeps the literal; thread from
    # LKBallGlobalParams if tmol_jax ever needs a non-default cutoff.
    distance_threshold: float = 6.0

    # LJ sigma fallbacks (used when we need a default sigma for f_desolv's
    # anchor; see :func:`lk_isotropic_pair_simplified`).
    lj_hbond_dis: float = 3.0
    lj_hbond_OH_donor_dis: float = 2.6
    lj_hbond_hdis: float = 1.75


LK_BALL_DEFAULTS = LKBallDefaults()


# --------------------------------------------------------------------------- #
# f_desolv (shared with LJLK)                                                 #
# --------------------------------------------------------------------------- #
#
# Upstream ``tmol/score/ljlk/potentials/lk_isotropic.hh::f_desolv`` is the
# "gaussian desolvation" primitive that sits at the heart of *every* LK
# score. We reimplement it here so this module has no hard dependency on
# the sibling :mod:`tmol_jax.score.ljlk.potentials` being ported (parallel
# porting; see the repository README / Rule #1).
#
# If :mod:`tmol_jax.score.ljlk.potentials` ever provides a canonical
# ``f_desolv``, importing from there is safe -- the formulae are the same.


_PI_POW_1P5 = 5.56832799683  # upstream hard-coded constant == pi ** 1.5


def f_desolv(
    dist: Array,
    lj_radius_i: Array,
    lk_dgfree_i: Array,
    lk_lambda_i: Array,
    lk_volume_j: Array,
) -> Array:
    """Lazaridis-Karplus gaussian desolvation energy.

    The i/j asymmetry is intentional — the upstream ``lk_isotropic_score``
    is the sum ``f_desolv(i, j) + f_desolv(j, i)``; the lk-ball term only
    uses the i-against-j direction because the water set ``WI`` belongs to
    ``I``. See the header of ``lk_isotropic.hh`` in the upstream tree.

    All arguments broadcast via ordinary ``jnp`` rules. Returns an array
    with the common broadcasted shape.

    NaN-safety: ``dist`` can be 0 for colinear pairs (padding / virtual
    atoms); ``lk_lambda_i`` can be 0 when the i-atom is padded (the
    LK-ball term fills padded rows with zero parameters). Both are
    guarded with the double-where trick so both primal and cotangent
    are finite on the masked-off pair.
    """

    dist = jnp.asarray(dist)
    lj_radius_i = jnp.asarray(lj_radius_i)
    lk_dgfree_i = jnp.asarray(lk_dgfree_i)
    lk_lambda_i = jnp.asarray(lk_lambda_i)
    lk_volume_j = jnp.asarray(lk_volume_j)

    safe_lambda = lk_lambda_i != 0.0
    lambda_safe = jnp.where(safe_lambda, lk_lambda_i, 1.0)
    safe_d = dist > 1e-6
    safe_dist = jnp.where(safe_d, dist, 1.0)
    r = (dist - lj_radius_i) / lambda_safe
    prefactor = -lk_volume_j * lk_dgfree_i / (2.0 * _PI_POW_1P5 * lambda_safe)
    V = prefactor / (safe_dist * safe_dist) * jnp.exp(-r * r)
    ok = safe_lambda & safe_d
    return jnp.where(ok, V, jnp.zeros_like(V))


def lk_isotropic_pair_simplified(
    dist: Array,
    lj_sigma_ij: Array,
    lj_radius_i: Array,
    lk_dgfree_i: Array,
    lk_lambda_i: Array,
    lk_volume_j: Array,
    bonded_path_length: Array | None = None,
) -> Array:
    """Isotropic LK (i-against-j direction) with upstream's piecewise ramp.

    This is a faithful port of ``lk_isotropic_pair::V`` from
    ``tmol/score/ljlk/potentials/lk_isotropic.hh``:

    * For ``dist < cpoly_close_dmin`` → flat plateau at ``f_desolv(d_min)``.
    * For ``[cpoly_close_dmin, cpoly_close_dmax]`` → cubic-Hermite ramp
      from ``(d_min, f_desolv(d_min), slope=0)`` to
      ``(cpoly_close_dmax, f_desolv(cpoly_close_dmax), slope=f'(cpoly_close_dmax))``.
    * For ``[cpoly_close_dmax, cpoly_far_dmin]`` → raw ``f_desolv(dist)``.
    * For ``[cpoly_far_dmin, cpoly_far_dmax]`` → cubic-Hermite ramp from
      ``(cpoly_far_dmin, f_desolv(cpoly_far_dmin), slope=f'(cpoly_far_dmin))``
      to ``(cpoly_far_dmax, 0, 0)``.
    * For ``dist > cpoly_far_dmax`` → 0.

    Parameters
    ----------
    dist, lj_sigma_ij, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j
        Per-pair scalars (broadcast). ``lj_sigma_ij`` is the effective LJ
        sigma for the pair (see ``tmol/score/ljlk/potentials/common.hh``
        :func:`lj_sigma`).
    bonded_path_length
        Optional; if provided, a count-pair weight is applied as in
        upstream (>4: 1.0, ==4: 0.2, <4: 0.0). If ``None`` we treat the
        pair as "far enough along the graph" and return unweighted.

    Returns
    -------
    Array
        LK desolvation with the upstream piecewise ramp applied.
    """
    from tmol_jax.score.ljlk.potentials import _lk_one_sided_with_fade
    from tmol_jax.score.common.count_pair import connectivity_weight

    value = _lk_one_sided_with_fade(
        dist,
        lj_sigma_ij,
        lj_radius_i,
        lk_dgfree_i,
        lk_lambda_i,
        lk_volume_j,
        jnp.asarray(6.0),
        jnp.asarray(4.5),
    )
    return (
        value
        if bonded_path_length is None
        else value * connectivity_weight(bonded_path_length)
    )


# --------------------------------------------------------------------------- #
# Water placement                                                             #
# --------------------------------------------------------------------------- #
#
# Upstream tmol places waters per-polar-atom with two distinct geometries:
#
# * **Donor H** (``build_don_water`` in ``water.hh``):
#     one water per polar-H at position ``D + dist * normalize(H - D)``,
#     i.e. the water sits on the extended D→H ray at ``lkb_water_dist``
#     from the heavy donor.
#
# * **Acceptor** (``build_water_for_acc`` in ``water.hh``):
#     waters placed with ``build_coordinate<Real>::V`` in
#     ``common/gen_coord.hh`` — an "internal-coordinate" frame construction
#     given ``(A, B, B0, dist, angle, torsion)``:
#         M[:,0] = normalize(A - B)
#         M[:,2] = normalize((A-B) × (C-B))            ; C = B0
#         M[:,1] = M[:,2] × M[:,0]
#         water  = A + M @ [ dist*cos(pi-angle),
#                            dist*sin(pi-angle)*cos(torsion),
#                            dist*sin(pi-angle)*sin(torsion) ]
#     The bond-angle and the list of torsions vary by hybridization:
#         sp2  : angle = 134.5°; tors = [0°, 180°]              → 2 waters
#         sp3  : angle = 109.3°; tors = [120°, 240°]            → 2 waters
#         ring : angle = 180.0°; tors = [0°]                    → 1 water
#                B is taken as the midpoint of (B, B0) before placing.
#
# The per-acceptor waters are laid into the ``[n_waters, 3]`` output array
# *after* any donor-H waters attached to the same heavy atom (e.g. the
# hydroxyl OH on SER/THR, which is both a donor-heavy and an acceptor).
# This matches upstream's ``water_offset = res_dat.acc_n_attached_H[acc_ind]``.
#
# All norm / normalize calls below use the double-where trick inherited
# from :func:`safe_normalize` so padded or near-degenerate inputs produce
# finite forward values *and* finite (zero) gradients.


# Upstream constants (``ljlk.yaml`` + ``water.hh``).
_LKB_WATER_ANGLE_SP2 = jnp.pi * 134.5 / 180.0
_LKB_WATER_ANGLE_SP3 = jnp.pi * 109.3 / 180.0
_LKB_WATER_ANGLE_RING = jnp.pi  # 180°
_LKB_WATER_TORS_SP2 = jnp.asarray([0.0, jnp.pi], dtype=jnp.float32)
_LKB_WATER_TORS_SP3 = jnp.asarray(
    [2.0 * jnp.pi / 3.0, 4.0 * jnp.pi / 3.0], dtype=jnp.float32
)
_LKB_WATER_TORS_RING = jnp.asarray([0.0], dtype=jnp.float32)

# Hybridization codes match :mod:`tmol_jax.score.hbond.params`.
HYBRIDIZATION_NONE = 0
HYBRIDIZATION_SP2 = 1
HYBRIDIZATION_SP3 = 2
HYBRIDIZATION_RING = 3


def build_don_water(
    donor_xyz: Array,
    h_xyz: Array,
    water_dist: float = LK_BALL_DEFAULTS.lkb_water_dist,
    eps: float = 1e-8,
) -> Array:
    """Donor-H water: ``D + water_dist * normalize(H - D)``.

    Upstream: ``build_don_water<Real>::V``. NaN-safe via
    :func:`safe_normalize`.
    """
    d = jnp.asarray(donor_xyz)
    h = jnp.asarray(h_xyz)
    unit = safe_normalize(h - d, axis=-1, eps=eps)
    return d + water_dist * unit


def build_coordinate(
    A: Array,
    B: Array,
    C: Array,
    dist: Array,
    angle: Array,
    torsion: Array,
    eps: float = 1e-8,
) -> Array:
    """Internal-coordinate placement.

    Upstream: ``tmol::score::common::build_coordinate<Real>::V`` in
    ``common/gen_coord.hh``.

    Given parent ``A``, grandparent ``B``, great-grandparent ``C``, a
    bond length ``dist``, a bond angle (radians) and a torsion
    (radians), return the 3-D position of the new atom.

    NaN-safe: when ``A == B`` or the frame is degenerate (``A``, ``B``,
    ``C`` collinear), the double-where trick ensures both the forward
    value and the backward gradient are finite. Degenerate frames fall
    back to an arbitrary orthonormal basis so the output is still
    well-defined — callers that feed padded inputs should gate the
    result downstream.
    """
    A = jnp.asarray(A)
    B = jnp.asarray(B)
    C = jnp.asarray(C)

    # Column 0: A - B (normalized).
    ab = A - B
    m0 = safe_normalize(ab, axis=-1, eps=eps)

    # Column 2: m0 × (C - B); if degenerate, fall back to a non-collinear
    # reference so the cross product is non-zero.
    cb = C - B
    cross = jnp.cross(m0, cb)
    cross_sq = jnp.sum(cross * cross, axis=-1)
    degenerate = cross_sq < eps

    # Pick an alternate ``C - B`` that is guaranteed non-parallel to m0.
    # Upstream uses +x if m0[0] != 1 else +y; we use the more-orthogonal
    # option between +x and +y.
    alt1 = jnp.asarray([1.0, 0.0, 0.0])
    alt2 = jnp.asarray([0.0, 1.0, 0.0])
    alt = jnp.where(jnp.abs(m0[..., 0:1]) < 0.9, alt1, alt2)
    cross_alt = jnp.cross(m0, alt)
    cross_safe = jnp.where(degenerate[..., None], cross_alt, cross)
    m2 = safe_normalize(cross_safe, axis=-1, eps=eps)

    # Column 1: m2 × m0 (already unit because m0 ⊥ m2 and both unit).
    m1 = jnp.cross(m2, m0)

    # Build the water in this frame.
    sin_a = jnp.sin(jnp.pi - angle)
    cos_a = jnp.cos(jnp.pi - angle)
    sin_t = jnp.sin(torsion)
    cos_t = jnp.cos(torsion)

    local = jnp.stack(
        [dist * cos_a, dist * sin_a * cos_t, dist * sin_a * sin_t],
        axis=-1,
    )
    # M @ local : [..., 3, 3] @ [..., 3] = [..., 3]
    # Instead of assembling M explicitly we contract column-wise.
    return A + (m0 * local[..., 0:1] + m1 * local[..., 1:2] + m2 * local[..., 2:3])


def place_donor_waters(
    D_xyz: Array,
    H_xyz: Array,
    donH_mask: Array,
    water_dist: float = LK_BALL_DEFAULTS.lkb_water_dist,
) -> Array:
    """Place one water per attached polar H on a donor heavy atom.

    Parameters
    ----------
    D_xyz
        ``[3]`` heavy donor coordinate.
    H_xyz
        ``[max_H, 3]`` polar-H coordinates attached to ``D``.
    donH_mask
        ``[max_H]`` boolean mask — ``True`` for a real attached H.
    water_dist
        Water distance (Å).

    Returns
    -------
    waters
        ``[max_H, 3]`` water coordinates. Padded slots hold ``NaN`` so
        downstream gates (e.g. :func:`lk_ball_pair`) can skip them.
    """
    D = jnp.asarray(D_xyz)
    H = jnp.asarray(H_xyz)
    mask = jnp.asarray(donH_mask)

    # Vectorise ``build_don_water`` over the ``max_H`` axis. We need a
    # finite dummy for masked-off slots so the gradient is well-behaved.
    H_safe = jnp.where(mask[:, None], H, D[None, :] + jnp.array([1.0, 0.0, 0.0]))
    w = D[None, :] + water_dist * safe_normalize(H_safe - D[None, :], axis=-1)
    nan = jnp.full_like(w, jnp.nan)
    return jnp.where(mask[:, None], w, nan)


def place_acceptor_waters(
    A_xyz: Array,
    B_xyz: Array,
    B0_xyz: Array,
    hybridization: Array,
    n_waters: int,
    water_dist: float = LK_BALL_DEFAULTS.lkb_water_dist,
) -> Array:
    """Place ``n_waters`` water sites on an acceptor atom.

    Dispatches on ``hybridization`` (sp2 / sp3 / ring, as integer codes
    matching :data:`HYBRIDIZATION_SP2` etc.) and uses the hard-coded
    upstream torsion tables. Returns ``[n_waters, 3]``; trailing slots
    beyond the hybridization-specific water count (2 for sp2/sp3,
    1 for ring) are set to ``NaN``.

    Parameters
    ----------
    A_xyz, B_xyz, B0_xyz
        ``[3]`` coordinates. For ring acceptors we use the midpoint of
        (B, B0) as the grandparent (upstream does
        ``Bxyz = (Bxyz + B0xyz) / 2`` before calling ``build_coordinate``).
    hybridization
        Scalar integer ``{none=0, sp2=1, sp3=2, ring=3}``.
    n_waters
        Static output size (usually the term's ``MAX_N_WATER``).
    water_dist
        Water distance (Å).
    """
    A = jnp.asarray(A_xyz)
    B = jnp.asarray(B_xyz)
    B0 = jnp.asarray(B0_xyz)
    hyb = jnp.asarray(hybridization)

    # Per-hybridization (angle, torsions, n_valid_waters).
    # All three hybridization types use the same B/B0 except ring, which
    # substitutes the midpoint for B.
    B_ring = 0.5 * (B + B0)
    B_eff = jnp.where(hyb == HYBRIDIZATION_RING, B_ring, B)

    # Compute water sites for all three hybridizations up-front; pick by
    # ``hyb`` via ``where``. This is cheap (3x constant work) and lets
    # XLA fuse the dispatches across all polar atoms in the pose.
    def _place_fixed(angle: Array, tors: Array) -> Array:
        """Place len(tors) waters, then pad with NaN to ``n_waters``."""

        def one(t):
            return build_coordinate(A, B_eff, B0, water_dist, angle, t)

        # vmap over torsions: [k, 3]
        ws = jax.vmap(one)(tors)
        k = tors.shape[0]
        pad = jnp.full((n_waters - k, 3), jnp.nan) if n_waters > k else None
        return ws if pad is None else jnp.concatenate([ws, pad], axis=0)

    sp2_waters = _place_fixed(_LKB_WATER_ANGLE_SP2, _LKB_WATER_TORS_SP2)
    sp3_waters = _place_fixed(_LKB_WATER_ANGLE_SP3, _LKB_WATER_TORS_SP3)
    ring_waters = _place_fixed(_LKB_WATER_ANGLE_RING, _LKB_WATER_TORS_RING)

    # Dispatch. We use a 3-way nested ``where``; the result is `[n_waters, 3]`.
    is_sp2 = hyb == HYBRIDIZATION_SP2
    is_sp3 = hyb == HYBRIDIZATION_SP3
    is_ring = hyb == HYBRIDIZATION_RING

    out = jnp.where(
        is_sp2,
        sp2_waters,
        jnp.where(
            is_sp3,
            sp3_waters,
            jnp.where(
                is_ring,
                ring_waters,
                jnp.full_like(sp2_waters, jnp.nan),
            ),
        ),
    )
    return out


def place_waters_for_polar_atom(
    coord: Array,
    is_donor: Array,
    is_acceptor: Array,
    hybridization: Array,
    n_attached_H: Array,
    H_coords: Array,
    H_mask: Array,
    B_xyz: Array,
    B0_xyz: Array,
    n_waters: int = 4,
    water_dist: float = LK_BALL_DEFAULTS.lkb_water_dist,
) -> Array:
    """Faithful port of upstream's polar-atom water placement.

    Builds up to ``n_waters`` water positions for a single polar heavy
    atom, laying donor-H waters in the first ``n_attached_H`` slots and
    acceptor waters starting at slot ``n_attached_H``. Padded slots are
    ``NaN``-filled (the upstream sentinel, gated by :func:`lk_ball_pair`).

    Parameters
    ----------
    coord
        ``[3]`` polar-atom coordinate (``D`` if donor, ``A`` if acceptor).
    is_donor, is_acceptor
        Scalar ``bool``/``int`` flags. An atom may be both (hydroxyl OH).
    hybridization
        Acceptor hybridization code (``0=none, 1=sp2, 2=sp3, 3=ring``).
        Ignored when ``is_acceptor`` is false.
    n_attached_H
        Scalar int: number of polar hydrogens bonded to ``coord``. These
        occupy waters ``[0 .. n_attached_H)``.
    H_coords
        ``[max_H, 3]`` attached polar-H coordinates. ``max_H`` is static
        and >= the maximum over the whole pose.
    H_mask
        ``[max_H]`` bool. The first ``n_attached_H`` entries are True.
    B_xyz
        ``[3]`` acceptor base. Used only when ``is_acceptor``; may be
        ``coord`` for donor-only atoms (value unused in that branch).
    B0_xyz
        ``[3]`` acceptor base-of-base. Same caveat as ``B_xyz``.
    n_waters
        Static output size (``MAX_N_WATER`` in upstream; usually 4).
    water_dist
        Water distance (Å).

    Returns
    -------
    ``[n_waters, 3]`` water coords. Non-polar or padded atoms should be
    masked by the caller (this function assumes a polar atom).
    """
    coord = jnp.asarray(coord)
    max_H = H_coords.shape[0]

    # ---- donor waters (one per polar H, in the first slots) ----
    # Pad/truncate to ``n_waters``.
    don_waters_raw = place_donor_waters(
        coord, H_coords, H_mask, water_dist=water_dist
    )  # [max_H, 3]
    if max_H >= n_waters:
        don_waters = don_waters_raw[:n_waters]
    else:
        pad = jnp.full((n_waters - max_H, 3), jnp.nan)
        don_waters = jnp.concatenate([don_waters_raw, pad], axis=0)

    # ---- acceptor waters (at offset ``n_attached_H``) ----
    acc_waters = place_acceptor_waters(
        coord, B_xyz, B0_xyz, hybridization, n_waters, water_dist=water_dist
    )  # [n_waters, 3]

    # Slot-wise pick: for slot i,
    #   if is_donor and i < n_attached_H  -> donor water i
    #   elif is_acceptor and i >= n_attached_H and (i - n_attached_H) < n_acc_waters
    #       -> acceptor water (i - n_attached_H)
    #   else -> NaN
    idx = jnp.arange(n_waters)
    acc_slot = idx - n_attached_H  # may be negative; NaN pad handles that
    # Gather acceptor_waters by acc_slot (clip to valid range; NaN
    # propagates through invalid slots via ``where``).
    acc_slot_safe = jnp.clip(acc_slot, 0, n_waters - 1)
    acc_gathered = acc_waters[acc_slot_safe]

    use_donor = jnp.logical_and(is_donor > 0, idx < n_attached_H)
    use_acceptor = jnp.logical_and(
        is_acceptor > 0, jnp.logical_and(acc_slot >= 0, acc_slot < n_waters)
    )

    out = jnp.where(
        use_donor[:, None],
        don_waters,
        jnp.where(use_acceptor[:, None], acc_gathered, jnp.nan),
    )
    return out


# --------------------------------------------------------------------------- #
# lk_fraction / lk_bridge_fraction (verbatim ports)                           #
# --------------------------------------------------------------------------- #


def lk_fraction(
    waters_i: Array,
    coord_j: Array,
    lj_radius_j: Array,
    ramp_width_A2: float = LK_BALL_DEFAULTS.ramp_width_A2,
    waters_i_mask: Array | None = None,
) -> Array:
    """Water-desolvation fraction. Upstream: ``lk_fraction<Real, MW>::V``.

    Given the waters of atom ``I`` (``waters_i``) and the coordinate of
    an occluder ``J``, compute the scale factor in ``[0, 1]`` that
    indicates how much ``J`` is "near" any of the waters. Near-overlap
    → 1 (strong occlusion, full LK desolvation); far → 0.

    Upstream uses the log-sum-exp-style weighted-squared-distance
    ``wted_d2_delta = -log(sum_w exp(-(|J - W_i|^2 - d2_low)))`` and
    then a squared-squared ramp to ``[0, 1]``. Water slots marked NaN
    are skipped; we emulate that via ``waters_i_mask``.

    Parameters
    ----------
    waters_i
        ``[MW, 3]`` water positions for atom I.
    coord_j
        ``[3]`` occluder position.
    lj_radius_j
        Scalar. Upstream's ``d2_low`` is
        ``max(0, (1.4 + lj_radius_j)^2 - ramp_width_A2)``.
    ramp_width_A2
        The "ramp is fully active over this squared-Å range" constant
        from ``lkball_globals``.
    waters_i_mask
        Optional ``[MW]`` bool mask — ``False`` entries (padded / NaN
        water slots) are excluded from the log-sum-exp. Upstream does
        this via ``!std::isnan(d2_delta)``.
    """
    waters_i = jnp.asarray(waters_i)
    coord_j = jnp.asarray(coord_j)

    d2_low = jnp.maximum((1.4 + jnp.asarray(lj_radius_j)) ** 2 - ramp_width_A2, 0.0)

    # |J - W|^2 - d2_low for each water, then log-sum-exp.
    delta = coord_j[None, :] - waters_i  # [MW, 3]
    d2 = jnp.sum(delta * delta, axis=-1) - d2_low  # [MW]

    if waters_i_mask is None:
        wted = -jax.nn.logsumexp(-d2, axis=-1)
    else:
        # ``logsumexp(..., b=mask)`` excludes masked-off entries (b==0
        # contributes ``exp(x) * 0 = 0`` to the sum). This is the NaN-safe
        # analogue of upstream's ``if (!std::isnan(d2_delta))`` guard.
        mask = jnp.asarray(waters_i_mask, dtype=d2.dtype)
        wted = -jax.nn.logsumexp(-d2, axis=-1, b=mask)

    # Squared-squared ramp.
    frac = jnp.where(
        wted < 0.0,
        jnp.ones_like(wted),
        jnp.where(
            wted < ramp_width_A2,
            jnp.square(1.0 - jnp.square(wted / ramp_width_A2)),
            jnp.zeros_like(wted),
        ),
    )
    return frac


def lk_bridge_fraction(
    coord_i: Array,
    coord_j: Array,
    waters_i: Array,
    waters_j: Array,
    lkb_water_dist: float = LK_BALL_DEFAULTS.lkb_water_dist,
    overlap_gap_A2: float = LK_BALL_DEFAULTS.overlap_gap_A2,
    overlap_width_A2: float = LK_BALL_DEFAULTS.overlap_width_A2,
    angle_overlap_A2: float = LK_BALL_DEFAULTS.angle_overlap_A2,
    waters_i_mask: Array | None = None,
    waters_j_mask: Array | None = None,
) -> Array:
    """Water-bridging fraction. Upstream: ``lk_bridge_fraction<Real, MW>::V``.

    Two factors:

    * ``overlapfrac`` — quadratic-ramp on the minimum water-water
      squared distance (log-sum-exp over all water pairs, then ramp).
    * ``anglefrac``   — quadratic-ramp on how close ``|I - J|^2`` is to
      the ideal bridging length ``8/3 * water_dist^2``.

    ``waters_i_mask`` / ``waters_j_mask`` follow the same NaN-skipping
    semantics as :func:`lk_fraction`. Returns the product, in ``[0, 1]``.
    """
    waters_i = jnp.asarray(waters_i)
    waters_j = jnp.asarray(waters_j)
    coord_i = jnp.asarray(coord_i)
    coord_j = jnp.asarray(coord_j)

    # Native overlap saturates at one; masked/absent waters contribute zero.
    valid_i = jnp.isfinite(waters_i).all(axis=-1)
    valid_j = jnp.isfinite(waters_j).all(axis=-1)
    if waters_i_mask is not None:
        valid_i &= jnp.asarray(waters_i_mask, dtype=bool)
    if waters_j_mask is not None:
        valid_j &= jnp.asarray(waters_j_mask, dtype=bool)
    wi = jnp.where(valid_i[:, None], waters_i, 0)
    wj = jnp.where(valid_j[:, None], waters_j, 0)
    d2 = jnp.sum((wi[:, None, :] - wj[None, :, :]) ** 2, axis=-1) - overlap_gap_A2
    total = jnp.where(valid_i[:, None] & valid_j[None, :], jnp.exp(-d2), 0).sum()
    bounded = jnp.clip(total, jnp.exp(-overlap_width_A2), 1.0)
    overlapfrac = jnp.square(1 - jnp.square(-jnp.log(bounded) / overlap_width_A2))
    overlapfrac = jnp.where(total <= jnp.exp(-overlap_width_A2), 0, overlapfrac)

    # ---- angle (base-length) fraction ------------------------------------
    target_len2 = (8.0 / 3.0) * (lkb_water_dist**2)
    base_delta = jnp.sum((coord_i - coord_j) ** 2) - target_len2

    anglefrac = jnp.where(
        jnp.abs(base_delta) > angle_overlap_A2,
        jnp.zeros_like(base_delta),
        jnp.square(1.0 - jnp.square(base_delta / angle_overlap_A2)),
    )

    return overlapfrac * anglefrac


# --------------------------------------------------------------------------- #
# lj_sigma (upstream: tmol/score/ljlk/potentials/common.hh)                   #
# --------------------------------------------------------------------------- #


def lj_sigma(
    lj_radius_i: Array,
    lj_radius_j: Array,
    is_donor_i: Array,
    is_donor_j: Array,
    is_hydroxyl_i: Array,
    is_hydroxyl_j: Array,
    is_polarh_i: Array,
    is_polarh_j: Array,
    is_acceptor_i: Array,
    is_acceptor_j: Array,
    lj_hbond_dis: float = LK_BALL_DEFAULTS.lj_hbond_dis,
    lj_hbond_OH_donor_dis: float = LK_BALL_DEFAULTS.lj_hbond_OH_donor_dis,
    lj_hbond_hdis: float = LK_BALL_DEFAULTS.lj_hbond_hdis,
) -> Array:
    """Effective LJ sigma for a pair (upstream: ``lj_sigma`` in common.hh)."""
    ij_donor_acc = jnp.logical_and(
        jnp.logical_and(is_donor_i, jnp.logical_not(is_hydroxyl_i)), is_acceptor_j
    )
    ji_donor_acc = jnp.logical_and(
        jnp.logical_and(is_donor_j, jnp.logical_not(is_hydroxyl_j)), is_acceptor_i
    )
    donor_acc = jnp.logical_or(ij_donor_acc, ji_donor_acc)

    ij_oh = jnp.logical_and(jnp.logical_and(is_donor_i, is_hydroxyl_i), is_acceptor_j)
    ji_oh = jnp.logical_and(jnp.logical_and(is_donor_j, is_hydroxyl_j), is_acceptor_i)
    oh_acc = jnp.logical_or(ij_oh, ji_oh)

    polh_acc = jnp.logical_or(
        jnp.logical_and(is_polarh_i, is_acceptor_j),
        jnp.logical_and(is_polarh_j, is_acceptor_i),
    )

    plain = lj_radius_i + lj_radius_j

    return jnp.where(
        donor_acc,
        lj_hbond_dis,
        jnp.where(
            oh_acc,
            lj_hbond_OH_donor_dis,
            jnp.where(polh_acc, lj_hbond_hdis, plain),
        ),
    )


# --------------------------------------------------------------------------- #
# Main pair routine                                                            #
# --------------------------------------------------------------------------- #


class LKBallPairResult(NamedTuple):
    """All four lk-ball scores for a single polar-occluder pair.

    Fields correspond to :class:`tmol_jax.score.score_types.ScoreType`
    names.
    """

    lk_iso: Array
    lk_ball: Array
    lk_bridge: Array
    lk_bridge_uncpl: Array


def lk_ball_pair(
    coord_i: Array,
    waters_i: Array,
    coord_j: Array,
    waters_j: Array,
    params: dict,
    defaults: LKBallDefaults = LK_BALL_DEFAULTS,
    bonded_path_length: Array | None = None,
) -> LKBallPairResult:
    """Full lk-ball pair energy, polar-against-occluder direction.

    Upstream reference: ``lk_ball_score<Real, MAX_WATER>::V`` in
    ``tmol/score/lk_ball/potentials/lk_ball.hh``.

    Parameters
    ----------
    coord_i
        ``[3]`` polar atom ``I`` position.
    waters_i
        ``[MW, 3]`` water positions for ``I``. If the first water is
        NaN, upstream returns zero — we replicate that gate.
    coord_j
        ``[3]`` occluder ``J`` position.
    waters_j
        ``[MW, 3]`` water positions for ``J``. Only used when ``J`` is
        itself a donor or acceptor (``params['j_is_donor' | 'j_is_acceptor']``).
    params
        Dict with the LK / LJ type parameters, keyed by the upstream
        struct names:

        * ``i_lj_radius, j_lj_radius``
        * ``i_lk_dgfree``, ``i_lk_lambda``, ``j_lk_volume``
        * ``i_is_donor / j_is_donor`` etc. (bool / 0-1)
        * Any key from :class:`LKBallDefaults` overrides the default.
    defaults
        Fallback constants for keys missing from ``params``.
    bonded_path_length
        If given, count-pair scaling is applied.

    Returns
    -------
    LKBallPairResult
        Four scalars (all broadcasting-compatible with the input shapes).
    """

    coord_i = jnp.asarray(coord_i)
    coord_j = jnp.asarray(coord_j)
    waters_i = jnp.asarray(waters_i)
    waters_j = jnp.asarray(waters_j)

    # Upstream gate: if WI[0] is NaN the polar atom has no waters → no score.
    has_waters = jnp.logical_not(jnp.isnan(waters_i[0, 0]))

    # Per-slot water-validity mask (upstream: ``!std::isnan(d2_delta)``).
    # A slot is "valid" iff any component is finite (in practice we pad
    # with NaN in the placement helpers, so the x-component is enough).
    waters_i_mask = jnp.logical_not(jnp.isnan(waters_i[:, 0]))
    waters_j_mask = jnp.logical_not(jnp.isnan(waters_j[:, 0]))

    # Sanitize NaN-sentinel waters before they flow into the
    # fraction/bridge computations. ``NaN * 0 == NaN`` in IEEE, so the
    # final ``* gate`` multiplication is NOT sufficient on its own to
    # scrub NaNs out of the backward pass — we must replace NaNs with a
    # finite dummy *before* the distance computations. The result is
    # gated back to zero below via ``has_waters``.
    safe_waters_i = jnp.where(jnp.isnan(waters_i), 0.0, waters_i)
    safe_waters_j = jnp.where(jnp.isnan(waters_j), 0.0, waters_j)

    # --- LJ sigma ---------------------------------------------------------
    sigma = lj_sigma(
        lj_radius_i=params["i_lj_radius"],
        lj_radius_j=params["j_lj_radius"],
        is_donor_i=params.get("i_is_donor", 0.0),
        is_donor_j=params.get("j_is_donor", 0.0),
        is_hydroxyl_i=params.get("i_is_hydroxyl", 0.0),
        is_hydroxyl_j=params.get("j_is_hydroxyl", 0.0),
        is_polarh_i=params.get("i_is_polarh", 0.0),
        is_polarh_j=params.get("j_is_polarh", 0.0),
        is_acceptor_i=params.get("i_is_acceptor", 0.0),
        is_acceptor_j=params.get("j_is_acceptor", 0.0),
        lj_hbond_dis=defaults.lj_hbond_dis,
        lj_hbond_OH_donor_dis=defaults.lj_hbond_OH_donor_dis,
        lj_hbond_hdis=defaults.lj_hbond_hdis,
    )

    # NaN-safe pair distance. ``coord_i == coord_j`` happens when both
    # atoms are i==j (diagonal pair, masked out later) or in padded
    # slots. The raw ``norm(0) -> 0`` gradient is NaN, so use the
    # double-where trick.
    dist = safe_norm(coord_i - coord_j, axis=-1)

    # --- Distance cutoff --------------------------------------------------
    within_cutoff = dist < defaults.distance_threshold

    # --- LK isotropic (upstream uses the pair-direction only) -------------
    lk_iso = lk_isotropic_pair_simplified(
        dist=dist,
        lj_sigma_ij=sigma,
        lj_radius_i=params["i_lj_radius"],
        lk_dgfree_i=params["i_lk_dgfree"],
        lk_lambda_i=params["i_lk_lambda"],
        lk_volume_j=params["j_lk_volume"],
        bonded_path_length=bonded_path_length,
    )

    # --- Water desolvation fraction ---------------------------------------
    frac_desolv = lk_fraction(
        waters_i=safe_waters_i,
        coord_j=coord_j,
        lj_radius_j=params["j_lj_radius"],
        ramp_width_A2=defaults.ramp_width_A2,
        waters_i_mask=waters_i_mask,
    )

    # --- Bridging fraction (only if J is donor or acceptor) ---------------
    j_is_polar = jnp.logical_or(
        jnp.asarray(params.get("j_is_donor", 0.0)) > 0.5,
        jnp.asarray(params.get("j_is_acceptor", 0.0)) > 0.5,
    )
    frac_bridge = jnp.where(
        j_is_polar,
        lk_bridge_fraction(
            coord_i=coord_i,
            coord_j=coord_j,
            waters_i=safe_waters_i,
            waters_j=safe_waters_j,
            lkb_water_dist=defaults.lkb_water_dist,
            overlap_gap_A2=defaults.overlap_gap_A2,
            overlap_width_A2=defaults.overlap_width_A2,
            angle_overlap_A2=defaults.angle_overlap_A2,
            waters_i_mask=waters_i_mask,
            waters_j_mask=waters_j_mask,
        ),
        jnp.zeros((), dtype=lk_iso.dtype),
    )

    # --- Assemble the four channels ---------------------------------------
    # Upstream ``lk_ball_atom_energy_full`` gates *all four* channels
    # when ``cp_separation <= 3`` (1-2 / 1-3 bonded pairs never contribute
    # to any lk-ball channel). The ``lk_iso`` channel already has the
    # BPL count-pair weight baked in via ``lk_isotropic_pair_simplified``
    # (and therefore ``lk_ball`` and ``lk_bridge`` inherit it through
    # their ``lk_iso`` factor). ``lk_bridge_uncpl = frac_bridge / 2`` has
    # no ``lk_iso`` factor — upstream only gates it at the hard <=3 cut.
    if bonded_path_length is None:
        cp_hard_gate = jnp.asarray(1.0, dtype=lk_iso.dtype)
    else:
        bpl = jnp.asarray(bonded_path_length)
        cp_hard_gate = jnp.where(
            bpl > 3,
            jnp.asarray(1.0, dtype=lk_iso.dtype),
            jnp.asarray(0.0, dtype=lk_iso.dtype),
        )

    gate = jnp.logical_and(has_waters, within_cutoff).astype(lk_iso.dtype)

    lk_iso_v = lk_iso * gate
    lk_ball_v = lk_iso * frac_desolv * gate
    lk_bridge_v = lk_iso * frac_bridge * gate
    lk_bridge_uncpl_v = (frac_bridge / 2.0) * gate * cp_hard_gate

    return LKBallPairResult(
        lk_iso=lk_iso_v,
        lk_ball=lk_ball_v,
        lk_bridge=lk_bridge_v,
        lk_bridge_uncpl=lk_bridge_uncpl_v,
    )


__all__ = [
    "HYBRIDIZATION_NONE",
    "HYBRIDIZATION_SP2",
    "HYBRIDIZATION_SP3",
    "HYBRIDIZATION_RING",
    "LKBallDefaults",
    "LKBallPairResult",
    "LK_BALL_DEFAULTS",
    "build_don_water",
    "build_coordinate",
    "f_desolv",
    "lj_sigma",
    "lk_bridge_fraction",
    "lk_ball_pair",
    "lk_fraction",
    "lk_isotropic_pair_simplified",
    "place_acceptor_waters",
    "place_donor_waters",
    "place_waters_for_polar_atom",
]
