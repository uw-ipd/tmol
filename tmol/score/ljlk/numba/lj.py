r"""
# Baseline LJ implementation

(From Alford et al with revisions.)

Van der Waals (VdW) interactions are short-range attractive and repulsive
forces that vary with atom-pair distance.  Whereas attractive forces result
from the cross-correlated motions of electrons in neighboring nonbonded atoms,
repulsive forces occur because electrons cannot occupy the same orbitals by the
Pauli exclusion principle.

To model VdW interactions, Rosetta uses the Lennard-Jones (LJ) 6−12 potential
calculating the interaction energy of atoms i and j given:

- The sum of atomic radii: ${\sigma}_{i,j} = \sigma_i + \sigma_j $.
- The geometric mean of well depths: ${\epsilon}_{i,j} = \sqrt{\epsilon_i \epsilon_j}$.
- The atom-pair distance: $d_{i,j}$.

$$
f_{vdw}(i, j) = {\epsilon}_{i,j} \left[
    \left(\frac{\sigma_{i,j}}{d_{i,j}} \right)^{13}
    - 2 \left(\frac{\sigma_{i,j}}{d_{i,j}} \right) ^ {6}
\right]
$$

The atomic radii and well depths are derived from small molecule liquid-phase
data optimized in the context of the energy model.
Note that in this form $\sigma$ is equivalent to an $r_{min}$  such that
$\sigma  = d \left| f_{vdw}(d) = \operatorname{min}{f_{vdw}(d)} \right.$ rather
than the traditional $\sigma  = d \left| f_{vdw}(d) = 0\right. $.

To accomodate the partially-covalent behavior of hydrogen bonds, $\sigma_{i,j}$
is adjusted for donor/acceptor and hydrogen/acceptor pairs:

$$
\sigma_{i,j} = \sigma_{j,i} = \left\{
\begin{array}{ll}
  \sigma_{hbond\_dis} &
    \left| i \in \{donor\}, i \notin \{hydroxyl\}, j \in \{acceptor\} \right. \\
  \sigma_{hbond\_OH\_donor\_dis} &
    \left| i \in \{donor\}, i \in \{hydroxyl\}, j \in \{acceptor\} \right. \\
  \sigma_{hbond\_H\_dis} &
    \left| i \in \{polar\_H\}, j \in \{acceptor\} \right. \\
  \sigma_{i} + \sigma_{j}
\end{array}
\right.
$$


At short distances, the $d_{i,j}^{-12}$ term can cause poor performance in
minimization and scoring due to high magnitude derivative and score values.  To
alleviate this problem the potential is linearly extrapolated for $ d_{i,j} <
.6{\sigma}_{i,j}$.  To accomodate efficient truncated evaluation, the potential
is defined as 0 for $d_{i,j} >= 6Å$.  Cubic polynomial interpolation
$f_{cpoly}$ is used to smoothly transition for a final piecewise definition:

$$
vdw_{i,j}\left(d_{i,j} \right) = \left\{
\begin{array}{ll}
  f^{\prime}_{vdw}(d_{lin}) \cdot \left(d_{i,j} - d_{lin}\right) + f_{vdw}(d_{lin})  &
      d_{i,j} \in [ 0Å, d_{lin} ) \\
  f_{vdw}(d_{i,j}) &
      d_{i,j} \in [ d_{lin}, 4.5Å ) \\
  f_{cpoly}(d_{i,j}) &
      d_{i,j} \in [ 4.5Å, 6.0Å) \\
  0 &
      d_{i,j} \in [ 6.0Å, +\infty ) \\
\end{array}
\right|
\begin{array}{ll}
  d_{lin} = .6 {\sigma}_{i,j} \\
  f_{cpoly}(4.5Å) = f_{vdw}(4.5Å), \,
  f^\prime_{cpoly}(4.5Å) = f^\prime_{vdw}(4.5Å) \\
  f_{cpoly}(6.0Å) = 0, \,
  f^\prime_{cpoly}(6.0Å) = 0 \\
\end{array}
$$
"""

import math
import numba
import toolz
import tmol.numeric.interpolation.cubic_hermite_polynomial as cubic_hermite_polynomial

jit = toolz.curry(numba.jit)(nopython=True)

interpolate_to_zero = jit(cubic_hermite_polynomial.interpolate_to_zero)
interpolate_to_zero_dx = jit(cubic_hermite_polynomial.interpolate_to_zero_dx)


@jit
def f_vdw(dist, sigma, epsilon):
    return epsilon * ((sigma / dist) ** 12 - 2 * (sigma / dist) ** 6)


@jit
def f_vdw_d_dist(dist, sigma, epsilon):
    return epsilon * (12 * sigma ** 6 / dist ** 7 - 12 * sigma ** 12 / dist ** 13)


@jit
def lj_sigma(
    lj_radius_i,
    is_donor_i,
    is_hydroxyl_i,
    is_polarh_i,
    is_acceptor_i,
    lj_radius_j,
    is_donor_j,
    is_hydroxyl_j,
    is_polarh_j,
    is_acceptor_j,
    lj_hbond_dis,
    lj_hbond_OH_donor_dis,
    lj_hbond_hdis,
):
    if (is_donor_i and not is_hydroxyl_i and is_acceptor_j) or (
        is_donor_j and not is_hydroxyl_j and is_acceptor_i
    ):
        return lj_hbond_dis
    elif (is_donor_i and is_hydroxyl_i and is_acceptor_j) or (
        is_donor_j and is_hydroxyl_j and is_acceptor_i
    ):
        return lj_hbond_OH_donor_dis
    elif (is_polarh_i and is_acceptor_j) or (is_polarh_j and is_acceptor_i):
        return lj_hbond_hdis
    else:
        return lj_radius_i + lj_radius_j


@numba.vectorize
def lj(
    dist,
    lj_radius_i,
    lj_wdepth_i,
    is_donor_i,
    is_hydroxyl_i,
    is_polarh_i,
    is_acceptor_i,
    lj_radius_j,
    lj_wdepth_j,
    is_donor_j,
    is_hydroxyl_j,
    is_polarh_j,
    is_acceptor_j,
    lj_hbond_dis,
    lj_hbond_OH_donor_dis,
    lj_hbond_hdis,
):
    sigma = lj_sigma(
        lj_radius_i,
        is_donor_i,
        is_hydroxyl_i,
        is_polarh_i,
        is_acceptor_i,
        lj_radius_j,
        is_donor_j,
        is_hydroxyl_j,
        is_polarh_j,
        is_acceptor_j,
        lj_hbond_dis,
        lj_hbond_OH_donor_dis,
        lj_hbond_hdis,
    )

    epsilon = math.sqrt(lj_wdepth_i * lj_wdepth_j)

    d_lin = sigma * 0.6
    lj_cpoly_dmin = 4.5
    lj_cpoly_dmax = 6.0

    if dist < d_lin:
        return f_vdw_d_dist(d_lin, sigma, epsilon) * (dist - d_lin) + f_vdw(
            d_lin, sigma, epsilon
        )
    elif dist < lj_cpoly_dmin:
        return f_vdw(dist, sigma, epsilon)
    elif dist < lj_cpoly_dmax:
        return interpolate_to_zero(
            dist,
            lj_cpoly_dmin,
            f_vdw(lj_cpoly_dmin, sigma, epsilon),
            f_vdw_d_dist(lj_cpoly_dmin, sigma, epsilon),
            lj_cpoly_dmax,
        )
    else:
        return 0.0


@numba.vectorize
def d_lj_d_dist(
    dist,
    lj_radius_i,
    lj_wdepth_i,
    is_donor_i,
    is_hydroxyl_i,
    is_polarh_i,
    is_acceptor_i,
    lj_radius_j,
    lj_wdepth_j,
    is_donor_j,
    is_hydroxyl_j,
    is_polarh_j,
    is_acceptor_j,
    lj_hbond_dis,
    lj_hbond_OH_donor_dis,
    lj_hbond_hdis,
):
    sigma = lj_sigma(
        lj_radius_i,
        is_donor_i,
        is_hydroxyl_i,
        is_polarh_i,
        is_acceptor_i,
        lj_radius_j,
        is_donor_j,
        is_hydroxyl_j,
        is_polarh_j,
        is_acceptor_j,
        lj_hbond_dis,
        lj_hbond_OH_donor_dis,
        lj_hbond_hdis,
    )

    epsilon = math.sqrt(lj_wdepth_i * lj_wdepth_j)

    d_lin = sigma * 0.6
    lj_cpoly_dmin = 4.5
    lj_cpoly_dmax = 6.0

    if dist < d_lin:
        return f_vdw_d_dist(d_lin, sigma, epsilon)
    elif dist < lj_cpoly_dmin:
        return f_vdw_d_dist(dist, sigma, epsilon)
    elif dist < lj_cpoly_dmax:
        return interpolate_to_zero_dx(
            dist,
            lj_cpoly_dmin,
            f_vdw(lj_cpoly_dmin, sigma, epsilon),
            f_vdw_d_dist(lj_cpoly_dmin, sigma, epsilon),
            lj_cpoly_dmax,
        )
    else:
        return 0.0
