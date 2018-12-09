r"""
# Baseline `fa_sol` implementation

(From Alford et al with revisions):

Rosetta utilizes an implicit solvation model combining both isotropic and
anisotropic components.

The isotropic (LK) desolvation component is an gaussian exclusion model
$f_{desolv}$, derived from [Lazaridis, Karplus'99]
(https://www.ncbi.nlm.nih.gov/pubmed/10223287), describing the energy
required to desolvate an atom $i$ when it is approached by a neighboring atom
$j$. In contrast to the original model, the $ΔG^{ref}$ term is removed in favor
of explict per-residue reference energies.

The energy of the atom-pair interaction varies with:

* Separation distance: $d_{i,j}$.
* Atomic radii: $σ_{i}$
* Experimentally determined vapor-to-water transfer free energy: $ΔG^{free}_i$
* Correlation length: $λ_i$
* Atomic volume of the desolvating atom, $V_j$

With a functional form:

$$
f_{desolv} =
    -V_i \frac{ΔG^{free}_i}{3\pi^{3/2} \lambda_i d_{i,j}^2}
    \exp \left[- \left(\frac{d_{i,j} - {\sigma}_{i}}{\lambda_i}\right)^2 \right]
$$

To accomodate softened repulsive potentials, which otherwise prevent
overlapping atomic radii, the potential is defined as a constant for $d_{i,j} <
{\sigma}_{i,j}$. To accomodate efficient truncated evaluation, the potential is
defined as 0 for $d_{i,j} >= 6Å$. Cubic polynomial interpolation $f_cpoly$ is
used to smoothly transition between these regions for a final piecewise
definition:

$$
fa\_sol_{i,j}\left(d_{i,j} \right) = \left\{
\begin{array}{ll}
  f_{desolv}({\sigma}_{i,j}) &
      d_{i,j} \in [ 0Å, {\sigma}_{i,j} - c_0 ) \\
  f_{cpoly\_low}(d_{i,j}) &
      d_{i,j} \in [ {\sigma}_{i,j} - c_0, {\sigma}_{i,j} + c_1 ) \\
  f_{desolv}(d_{i,j}) &
      d_{i,j} \in [ {\sigma}_{i,j} + c_1, 4.5Å ) \\
  f_{cpoly\_hi}(d_{i,j}) &
      d_{i,j} \in [ 4.5Å, 6.0Å ) \\
  0 &
      d_{i,j} \in [ 6.0Å, +\infty ) \\
\end{array}
\right|
\begin{array}{ll}
  c_0 = 0.3 Å \\
  c_1 = 0.2 Å \\
  f_{cpoly\_low}(\sigma_{i,j} - c_0) = f_{desolv}({\sigma}_{i,j}), \,
  f^\prime_{cpoly\_low}(\sigma_{i,j} - c_0) = 0 \\
  f_{cpoly\_low}(\sigma_{i,j} + c_1) = f_{desolv}(\sigma_{i,j} + c_1), \,
  f^\prime_{cpoly\_low}({\sigma}_{i,j} + c_1) = f_{desolv}^\prime(\sigma_{i,j} + c_1) \\
  f_{cpoly\_hi}(4.5Å) = f_{desolv}(4.5Å), \,
  f^\prime_{cpoly\_hi}(4.5Å) = f^\prime_{desolv}(4.5Å) \\
  f_{cpoly\_hi}(6.0Å) = 0, \,
  f^\prime_{cpoly\_hi}(6.0Å) = 0 \\
\end{array}
$$
"""

import toolz

import numba
from numpy import exp, pi


import tmol.numeric.interpolation.cubic_hermite_polynomial as cubic_hermite_polynomial

jit = toolz.curry(numba.jit)(nopython=True)

interpolate = jit(cubic_hermite_polynomial.interpolate)
interpolate_dx = jit(cubic_hermite_polynomial.interpolate_dx)
interpolate_to_zero = jit(cubic_hermite_polynomial.interpolate_to_zero)
interpolate_to_zero_dx = jit(cubic_hermite_polynomial.interpolate_to_zero_dx)


@jit
def f_desolv(dist, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j):
    return (
        lk_volume_j
        * lk_dgfree_i
        / (2 * pi ** (3 / 2) * lk_lambda_i)
        * dist ** -2
        * exp(-((dist - lj_radius_i) / lk_lambda_i) ** 2)
    )


@jit
def f_desolv_d_dist(dist, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j):
    return (
        lk_volume_j
        * lk_dgfree_i
        / (2 * pi ** (3 / 2) * lk_lambda_i)
        * (  # (f * exp(g))' = f' * exp(g) + f g' exp(g)
            -2 * dist ** -3 * exp(-(dist - lj_radius_i) ** 2 / lk_lambda_i ** 2)
            + dist ** -2
            * -(2 * dist - 2 * lj_radius_i)
            / lk_lambda_i ** 2
            * exp(-(dist - lj_radius_i) ** 2 / lk_lambda_i ** 2)
        )
    )


@jit
def lk_isotropic(dist, lj_radius_i, lk_dgfree_i, lk_lambda_i, lj_radius_j, lk_volume_j):
    sigma_ij = lj_radius_i + lj_radius_j

    lk_cpoly_close_dmin = sigma_ij - 0.2
    lk_cpoly_close_dmax = sigma_ij + 0.3

    lk_cpoly_far_dmin = 4.5
    lk_cpoly_far_dmax = 6.0

    if dist < lk_cpoly_close_dmin:
        return f_desolv(sigma_ij, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j)
    elif dist < lk_cpoly_close_dmax:
        return interpolate(
            dist,
            lk_cpoly_close_dmin,
            f_desolv(sigma_ij, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j),
            0,
            lk_cpoly_close_dmax,
            f_desolv(
                lk_cpoly_close_dmax, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j
            ),
            f_desolv_d_dist(
                lk_cpoly_close_dmax, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j
            ),
        )
    elif dist < lk_cpoly_far_dmin:
        return f_desolv(dist, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j)
    elif dist < lk_cpoly_far_dmax:
        return interpolate_to_zero(
            dist,
            lk_cpoly_far_dmin,
            f_desolv(
                lk_cpoly_far_dmin, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j
            ),
            f_desolv_d_dist(
                lk_cpoly_far_dmin, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j
            ),
            lk_cpoly_far_dmax,
        )
    else:
        return 0.0


@jit
def d_lk_isotropic_d_dist(
    dist, lj_radius_i, lk_dgfree_i, lk_lambda_i, lj_radius_j, lk_volume_j
):
    sigma_ij = lj_radius_i + lj_radius_j

    lk_cpoly_close_dmin = sigma_ij - 0.2
    lk_cpoly_close_dmax = sigma_ij + 0.3

    lk_cpoly_far_dmin = 4.5
    lk_cpoly_far_dmax = 6.0

    if dist < lk_cpoly_close_dmin:
        return 0.0
    elif dist < lk_cpoly_close_dmax:
        return interpolate_dx(
            dist,
            lk_cpoly_close_dmin,
            f_desolv(sigma_ij, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j),
            0,
            lk_cpoly_close_dmax,
            f_desolv(
                lk_cpoly_close_dmax, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j
            ),
            f_desolv_d_dist(
                lk_cpoly_close_dmax, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j
            ),
        )
    elif dist < lk_cpoly_far_dmin:
        return f_desolv_d_dist(dist, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j)
    elif dist < lk_cpoly_far_dmax:
        return interpolate_to_zero_dx(
            dist,
            lk_cpoly_far_dmin,
            f_desolv(
                lk_cpoly_far_dmin, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j
            ),
            f_desolv_d_dist(
                lk_cpoly_far_dmin, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j
            ),
            lk_cpoly_far_dmax,
        )
    else:
        return 0.0


@numba.vectorize
def lk_isotropic_mutual(
    dist,
    lj_radius_i,
    lk_dgfree_i,
    lk_lambda_i,
    lk_volume_i,
    lj_radius_j,
    lk_dgfree_j,
    lk_lambda_j,
    lk_volume_j,
):
    return lk_isotropic(
        dist, lj_radius_i, lk_dgfree_i, lk_lambda_i, lj_radius_j, lk_volume_j
    ) + lk_isotropic(
        dist, lj_radius_j, lk_dgfree_j, lk_lambda_j, lj_radius_i, lk_volume_i
    )


@numba.vectorize
def d_lk_isotropic_mutual_d_dist(
    dist,
    lj_radius_i,
    lk_dgfree_i,
    lk_lambda_i,
    lk_volume_i,
    lj_radius_j,
    lk_dgfree_j,
    lk_lambda_j,
    lk_volume_j,
):
    return d_lk_isotropic_d_dist(
        dist, lj_radius_i, lk_dgfree_i, lk_lambda_i, lj_radius_j, lk_volume_j
    ) + d_lk_isotropic_d_dist(
        dist, lj_radius_j, lk_dgfree_j, lk_lambda_j, lj_radius_i, lk_volume_i
    )
