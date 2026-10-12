"""Dunbrack scoring primitives in pure JAX.

The upstream Dunbrack scoring (``tmol/score/dunbrack/potentials/*.hh``) is a
big C++ template library that does three things per rotameric residue:

* ``rot``   -- look up ``-ln P(rotamer | phi, psi)`` (a 2-D b-spline over
  the (phi, psi) grid with a b-spline coefficient table per rotamer).
* ``rotdev`` -- for every chi of a residue, add a Gaussian penalty
  ``(chi - mean(phi,psi))^2 / (2 * sigma(phi,psi)^2)`` where the mean
  and sigma are themselves 2-D (phi, psi) b-splines.
* ``semirot`` -- for aromatic residues that have a continuous last chi
  (PHE, TYR, HIS, ASN, ASP, GLN, GLU, TRP), add a 3-D (phi, psi, chi_last)
  b-spline ``-ln P(chi_last | rotamer, phi, psi)``.

This port takes the *simplified but structurally correct* interpretation
requested by the spec: bilinear / trilinear interpolation (instead of
cubic b-splines) on the *already-evaluated* probability grids. The grids
themselves come straight out of the library tables (``rotamer_probabilities``
etc.) and the wrap / scale rules are identical to upstream (angles are
stored in radians and wrapped into ``[bb_start, bb_start + period)``
before being divided by ``bb_step``).

All functions are ``jit`` / ``vmap`` / ``grad``-safe: they rely only on
``jax.numpy`` primitives, take static axes via broadcasting, and use
no Python control flow keyed on traced values.

The three public entry points are:

* :func:`dunbrack_rot_score` -- ``-log P(rotamer | phi, psi)`` for a
  single rotamer assignment on a single residue.
* :func:`dunbrack_rotdev_score` -- summed Gaussian chi-deviation penalty
  over a residue's chis, given per-chi means and sigmas.
* :func:`dunbrack_semirot_score` -- 3-D interpolation of the semirotameric
  ``-log P`` table at ``(phi, psi, chi_last)``.

Each is written to operate on a *single table* already sliced to the
correct rotamer/amino acid; the energy term above does the dispatch.
"""

from __future__ import annotations

import jax.numpy as jnp

from tmol_jax.types import Array

# --------------------------------------------------------------------------- #
# Angle wrapping                                                              #
# --------------------------------------------------------------------------- #


def _wrap_angle(angle: Array, start: Array, period: Array) -> Array:
    """Wrap ``angle`` (radians) into ``[start, start + period)``.

    A pure-functional equivalent of the ``while`` loops in
    ``potentials.hh``::

        while (d < 0)        d += period;
        while (d > period)   d -= period;

    We use ``jnp.mod`` which works for any real input and is
    ``jit``-friendly. Because ``jnp.mod`` returns a non-negative
    result when ``period > 0``, the output is guaranteed to live in
    ``[0, period)`` after subtracting the start offset.

    Parameters
    ----------
    angle
        Input angle in radians. Any shape; broadcast-compatible with
        ``start`` and ``period``.
    start
        Lower bound of the target interval (e.g. ``-pi`` for the
        rotameric phi/psi grids).
    period
        Width of the target interval (e.g. ``2 * pi`` for full rotation).
    """
    return start + jnp.mod(angle - start, period)


def _frac_index(angle: Array, start: Array, step: Array, period: Array) -> Array:
    """Convert an angle (radians) into a *fractional grid index*.

    The grids tabulate a function on equally spaced samples starting at
    ``start`` and separated by ``step``. Given ``angle``, this function
    returns where that angle lies within the grid, in units of grid
    cells. The caller is responsible for wrapping the returned value
    back into ``[0, n-1]`` if it uses it as a lookup index.

    Returns
    -------
    fractional index
        ``(wrap(angle) - start) / step``. Broadcast over any leading
        dims of ``angle``.
    """
    wrapped = _wrap_angle(angle, start, period)
    return (wrapped - start) / step


# --------------------------------------------------------------------------- #
# Cubic B-spline (periodic) — coefficient precomputation + 2-D interpolation  #
# --------------------------------------------------------------------------- #
#
# Port of ``tmol/numeric/bspline_compiled/bspline.hh`` restricted to 2-D,
# DEGREE=3 (bicubic) with periodic boundary conditions. Upstream uses the
# same code path for Dunbrack's rot / mean / sdev tables (see ``params.py``
# in upstream dunbrack, ``BSplineInterpolation.from_coordinates``).
#
# Two pieces:
#
# * :func:`bspline3_coefficients_for_table` — convert the final periodic grid
#   axes into B-spline coefficients. Runs on CPU (SciPy), called once at setup.
# * :func:`_bicubic_bspline_interpolate` — grad-safe 2-D interpolation of
#   the already-converted coefficient table at a fractional (i, j) index.
#
# The separable coefficient filter lets us convert an N-D grid by running
# the 1-D filter along each axis in turn.


def bspline3_coefficients_for_table(values, n_grid_axes):
    """Compute periodic cubic B-spline coefficients on final grid axes.

    SciPy ``grid-wrap`` implements the same periodic cubic pole recursion as
    TMol. Filtering the axes sequentially is the separable N-D prefilter,
    without tracing and compiling hundreds of setup-only JAX scans.
    """
    import numpy as np
    from scipy.ndimage import spline_filter1d

    arr = np.asarray(values, dtype=np.float64)
    if n_grid_axes < 1 or n_grid_axes > arr.ndim:
        raise ValueError("invalid number of B-spline grid axes")
    for axis in range(arr.ndim - n_grid_axes, arr.ndim):
        arr = spline_filter1d(arr, order=3, axis=axis, mode="grid-wrap")
    return arr.astype(np.float32)


def bspline3_coeffs_for_2d_table(values):
    """Compute bicubic B-spline coefficients for one 2-D table.

    Convenience wrapper used by the Dunbrack energy-term setup — the
    table's last two axes are the (phi, psi) grid; leading axes can be
    rotamer / AA / channel stacks and are batched over.
    """
    return bspline3_coefficients_for_table(values, 2)


def _bspline3_weights(f):
    """Cubic-B-spline evaluation weights at fractional offset ``f``.

    Matches ``_get_weights`` in ``bspline.hh`` for DEGREE=3. Returns a
    length-4 weight vector. Broadcast-friendly: ``f`` can be any shape.
    """
    w3 = (1.0 / 6.0) * (f**3)
    w0 = (1.0 / 6.0) + 0.5 * f * (f - 1.0) - w3
    w2 = f + w0 - 2.0 * w3
    w1 = 1.0 - w0 - w2 - w3
    return w0, w1, w2, w3


def _bicubic_bspline_interpolate(
    coeffs: Array,
    frac_i: Array,
    frac_j: Array,
) -> Array:
    """Bicubic B-spline interpolation on a 2-D periodic coefficient grid.

    ``coeffs`` must already have been run through
    :func:`bspline3_coeffs_for_2d_table` (i.e. be B-spline coefficients,
    not raw sample values). Both axes are assumed periodic.

    Port of the ``interpolate`` routine in ``bspline.hh`` for NDIM=2 and
    DEGREE=3. Reads a 4x4 patch of coefficients centred on the fractional
    index, gathers with periodic wrap, and applies the tensor-product
    weights from :func:`_bspline3_weights`.
    """
    ni = coeffs.shape[0]
    nj = coeffs.shape[1]

    i_f = jnp.floor(frac_i).astype(jnp.int32)
    j_f = jnp.floor(frac_j).astype(jnp.int32)
    di = frac_i - i_f
    dj = frac_j - j_f

    # Spline support spans [idx - DEGREE/2, idx + DEGREE/2 + 1) for
    # DEGREE odd = 3: that's [idx - 1, idx + 3), i.e. 4 samples.
    base_i = i_f - 1
    base_j = j_f - 1

    # Weights along each axis.
    wi0, wi1, wi2, wi3 = _bspline3_weights(di)
    wj0, wj1, wj2, wj3 = _bspline3_weights(dj)

    # Gather the 4x4 neighbourhood with periodic wrap on both axes.
    def idx(axis_size, base, k):
        return jnp.mod(base + k, axis_size)

    result = jnp.zeros_like(frac_i)
    wi_tuple = (wi0, wi1, wi2, wi3)
    wj_tuple = (wj0, wj1, wj2, wj3)
    for a in range(4):
        ia = idx(ni, base_i, a)
        for b in range(4):
            jb = idx(nj, base_j, b)
            result = result + wi_tuple[a] * wj_tuple[b] * coeffs[ia, jb]
    return result


# --------------------------------------------------------------------------- #
# Bilinear / trilinear interpolation on a periodic grid                       #
# --------------------------------------------------------------------------- #


def _bilinear_interpolate(
    table: Array,
    frac_i: Array,
    frac_j: Array,
) -> Array:
    """Bilinear interpolation on a 2-D periodic grid.

    ``table`` has shape ``[N_i, N_j]`` and is assumed periodic along
    both axes (so index ``N_i`` wraps to ``0`` etc.). ``frac_i`` and
    ``frac_j`` are fractional grid indices as returned by
    :func:`_frac_index`.

    All fractional-index / modular arithmetic is done with ``jnp``
    primitives so the function is ``grad``-safe; a single ``floor``
    produces the lower-left corner, and the four corners are combined
    with standard bilinear weights.

    Returns
    -------
    value
        Interpolated scalar value at ``(frac_i, frac_j)``. Shape is
        the broadcast of ``frac_i`` and ``frac_j``.
    """
    ni = table.shape[0]
    nj = table.shape[1]

    i0 = jnp.floor(frac_i).astype(jnp.int32)
    j0 = jnp.floor(frac_j).astype(jnp.int32)
    di = frac_i - i0
    dj = frac_j - j0

    # Wrap to [0, n).
    i0 = jnp.mod(i0, ni)
    j0 = jnp.mod(j0, nj)
    i1 = jnp.mod(i0 + 1, ni)
    j1 = jnp.mod(j0 + 1, nj)

    v00 = table[i0, j0]
    v01 = table[i0, j1]
    v10 = table[i1, j0]
    v11 = table[i1, j1]

    return (
        v00 * (1.0 - di) * (1.0 - dj)
        + v01 * (1.0 - di) * dj
        + v10 * di * (1.0 - dj)
        + v11 * di * dj
    )


def _trilinear_interpolate(
    table: Array,
    frac_i: Array,
    frac_j: Array,
    frac_k: Array,
) -> Array:
    """Trilinear interpolation on a 3-D periodic grid.

    Exactly the same idea as :func:`_bilinear_interpolate`, extended to
    three axes. The ``table`` has shape ``[N_i, N_j, N_k]``; all three
    axes are treated as periodic.
    """
    ni = table.shape[0]
    nj = table.shape[1]
    nk = table.shape[2]

    i0 = jnp.floor(frac_i).astype(jnp.int32)
    j0 = jnp.floor(frac_j).astype(jnp.int32)
    k0 = jnp.floor(frac_k).astype(jnp.int32)
    di = frac_i - i0
    dj = frac_j - j0
    dk = frac_k - k0

    i0 = jnp.mod(i0, ni)
    j0 = jnp.mod(j0, nj)
    k0 = jnp.mod(k0, nk)
    i1 = jnp.mod(i0 + 1, ni)
    j1 = jnp.mod(j0 + 1, nj)
    k1 = jnp.mod(k0 + 1, nk)

    c000 = table[i0, j0, k0]
    c001 = table[i0, j0, k1]
    c010 = table[i0, j1, k0]
    c011 = table[i0, j1, k1]
    c100 = table[i1, j0, k0]
    c101 = table[i1, j0, k1]
    c110 = table[i1, j1, k0]
    c111 = table[i1, j1, k1]

    return (
        c000 * (1 - di) * (1 - dj) * (1 - dk)
        + c001 * (1 - di) * (1 - dj) * dk
        + c010 * (1 - di) * dj * (1 - dk)
        + c011 * (1 - di) * dj * dk
        + c100 * di * (1 - dj) * (1 - dk)
        + c101 * di * (1 - dj) * dk
        + c110 * di * dj * (1 - dk)
        + c111 * di * dj * dk
    )


# --------------------------------------------------------------------------- #
# Public scoring primitives                                                   #
# --------------------------------------------------------------------------- #


def dunbrack_rot_score(
    phi: Array,
    psi: Array,
    neglnprob_table: Array,
    bb_start: Array,
    bb_step: Array,
    bb_period: Array,
) -> Array:
    """Rotamer log-probability penalty at ``(phi, psi)``.

    Looks up a single rotamer's ``-log P`` value on the tabulated
    (phi, psi) grid via bilinear interpolation. Upstream uses a cubic
    b-spline; we use bilinear, which is both smooth enough for the
    simplified scoring and ``grad``-safe.

    Parameters
    ----------
    phi, psi
        Backbone dihedrals in radians. Scalar or arbitrary shape
        (broadcast together inside the function).
    neglnprob_table
        ``[N_phi, N_psi]`` array of ``-log P(rotamer | phi, psi)``
        values for one specific rotamer. Already sliced by the caller
        to the rotamer assigned to this residue.
    bb_start
        2-vector ``(phi_start, psi_start)`` in radians marking the lower
        bound of the grid (upstream stores this in the rotlib as
        ``backbone_dihedral_start`` * pi/180).
    bb_step
        2-vector ``(phi_step, psi_step)`` in radians — the tabulation
        spacing.
    bb_period
        2-vector ``(phi_period, psi_period)`` in radians — usually
        ``(2pi, 2pi)``.
    """
    fi = _frac_index(phi, bb_start[0], bb_step[0], bb_period[0])
    fj = _frac_index(psi, bb_start[1], bb_step[1], bb_period[1])
    return _bilinear_interpolate(neglnprob_table, fi, fj)


def dunbrack_rotdev_score(
    chis: Array,
    chi_means: Array,
    chi_sigmas: Array,
    chi_mask: Array | None = None,
) -> Array:
    """Summed Gaussian deviation penalty over the chi angles of a residue.

    Implements the standard rotamer-deviation term used by the Dunbrack
    library::

        sum_i  (chi_i - mean_i)^2 / (2 * sigma_i^2)

    Upstream (``block_chi_deviation_penalty`` in
    ``tmol/score/dunbrack/potentials/potentials.hh``) uses an
    *asymmetric* wrap to keep ``chi_dev`` on the trans branch::

        chi_dev = chi < -120deg ? chi + 2*pi - mean : chi - mean

    This is the exact form we replicate here. A symmetric shortest-arc
    wrap (``mod(diff + pi, 2pi) - pi``) gives the same numerical answer
    in the dominant case (chi in the centre of its rotamer well) but
    differs at the chi=-120deg seam between the trans and g-/g+ bins
    — and that difference back-propagates non-zero gradient discrepancies
    on backbone atoms that influence the (phi, psi)-dependent mean and
    sigma. Using the upstream asymmetric formula brings the
    ``dunbrack_rotdev`` per-atom gradient diff to float precision.

    Parameters
    ----------
    chis
        ``[..., n_chi]`` chi values in radians.
    chi_means, chi_sigmas
        ``[..., n_chi]`` means and sigmas (radians) from the library,
        typically produced by bilinearly interpolating the mean /
        sigma tables at ``(phi, psi)``.
    chi_mask
        Optional ``[..., n_chi]`` bool mask. ``False`` entries contribute
        zero. Used to handle residues with fewer than ``max_nchi``
        chi angles in a batched setting.
    """
    # Upstream's asymmetric wrap: shift chi by 2*pi when chi < -120 deg
    # before subtracting the mean. The means table has already been
    # pre-shifted at setup time so that "trans" rotamer means live in
    # roughly [180, 240] deg (see DunbrackEnergyTerm._stack_rotameric_tables
    # — ``means_deg = where(means_deg < -120, means_deg + 360, means_deg)``).
    deg120 = jnp.deg2rad(120.0)
    two_pi = 2.0 * jnp.pi
    chi_shifted = jnp.where(chis < -deg120, chis + two_pi, chis)
    diff = chi_shifted - chi_means

    # Clamp sigma to avoid division by zero in padding slots.
    safe_sigma = jnp.where(chi_sigmas > 1e-6, chi_sigmas, 1.0)
    penalty = (diff * diff) / (2.0 * safe_sigma * safe_sigma)
    # Zero out contributions from padding / bad sigma.
    penalty = jnp.where(chi_sigmas > 1e-6, penalty, 0.0)

    if chi_mask is not None:
        penalty = jnp.where(chi_mask, penalty, 0.0)

    return jnp.sum(penalty, axis=-1)


def dunbrack_semirot_score(
    phi: Array,
    psi: Array,
    chi_last: Array,
    semirot_table: Array,
    semirot_start: Array,
    semirot_step: Array,
    semirot_period: Array,
) -> Array:
    """Semirotameric ``-ln P(chi_last | rotamer, phi, psi)`` penalty.

    For aromatic / carboxyl residues (PHE, TYR, HIS, ASN, ASP, GLN,
    GLU, TRP) the terminal chi is not discretised into three wells;
    instead the library ships a 3-D table of ``-log P`` sampled on a
    ``(phi, psi, chi_last)`` grid (per rotamer on the *preceding*
    chis). Tricubic B-spline interpolation on the prefiltered
    coefficient grid matches upstream ``ndspline<3, 3>::interpolate``
    to float32 precision.

    Parameters
    ----------
    phi, psi, chi_last
        Angles in radians. Scalars or broadcast-compatible shapes.
    semirot_table
        ``[N_phi, N_psi, N_chi_last]`` array of tricubic B-spline
        *coefficients* (already prefiltered; NOT raw samples).
    semirot_start
        3-vector start in radians.
    semirot_step
        3-vector step in radians.
    semirot_period
        3-vector period in radians.
    """
    from tmol_jax.numeric.bspline import bspline_interpolate

    fi = _frac_index(phi, semirot_start[0], semirot_step[0], semirot_period[0])
    fj = _frac_index(psi, semirot_start[1], semirot_step[1], semirot_period[1])
    fk = _frac_index(chi_last, semirot_start[2], semirot_step[2], semirot_period[2])
    X = jnp.stack([fi, fj, fk])
    return bspline_interpolate(semirot_table, X, degree=3)


# --------------------------------------------------------------------------- #
# Rotamer bin classification (utility used by the energy term)                #
# --------------------------------------------------------------------------- #


def classify_rotamer_bin(chis: Array, n_rotameric_chi: int) -> Array:
    """Classify each chi into a three-letter rotamer well index.

    Upstream ``classify_rotamer`` in ``potentials.hh`` uses the same
    cutoffs:

    * chi in ``[-180, -120) deg``  -> ``1`` (trans)
    * chi in ``[-120,    0) deg``  -> ``2`` (g-)
    * chi in ``[   0, +120) deg``  -> ``0`` (g+)
    * chi in ``[+120, +180) deg``  -> ``1`` (trans)

    The composite rotamer index is
    ``sum_i 3^(n-1-i) * bin(chi_i)``, i.e. the standard base-3 encoding
    where the first (outermost) chi is the most significant digit.
    This matches the ``rotameric_rotind2tableind`` layout used to look
    up rotamer-specific probability / mean / sigma tables.

    Parameters
    ----------
    chis
        ``[..., n_chi_max]`` chi values in radians. Only the first
        ``n_rotameric_chi`` entries are used; the rest are ignored.
    n_rotameric_chi
        Static int: how many of the leading chis are discretised into
        rotameric bins.

    Returns
    -------
    rot_ind
        ``[...]`` integer rotamer index in ``[0, 3^n_rotameric_chi)``.
    """
    deg120 = jnp.deg2rad(120.0)

    def bin_of(chi):
        # chi < -120    -> 1 (trans)
        # -120 <= chi <    0 -> 2 (g-)
        #    0 <= chi <  120 -> 0 (g+)
        #  120 <= chi       -> 1 (trans)
        return jnp.where(
            chi < -deg120,
            1,
            jnp.where(chi < 0.0, 2, jnp.where(chi < deg120, 0, 1)),
        )

    rot_ind = jnp.zeros(chis.shape[:-1], dtype=jnp.int32)
    for i in range(n_rotameric_chi):
        rot_ind = rot_ind * 3 + bin_of(chis[..., i])
    return rot_ind


__all__ = [
    "classify_rotamer_bin",
    "dunbrack_rot_score",
    "dunbrack_rotdev_score",
    "dunbrack_semirot_score",
]
