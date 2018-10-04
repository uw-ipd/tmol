#include <Eigen/Core>

namespace tmol
{
namespace score
{
namespace ljlk
{
template <typename Real, typename Int>
EIGEN_DEVICE_FUNC Real lj_potential(
    Real& dist,
    const Int& bonded_path_length,
    const Real& lj_sigma,
    const Real& lj_switch_slope,
    const Real& lj_switch_intercept,
    const Real& lj_coeff_sigma12,
    const Real& lj_coeff_sigma6,
    const Real& lj_spline_y0,
    const Real& lj_spline_dy0,
    const Real& lj_switch_dis2sigma,
    const Real& spline_start,
    const Real& max_dis)
{
    Real lj = 0.0;

    if (dist > max_dis) {
        // Outside of interaction distance
        return 0.0;
    } else if (bonded_path_length < 4) {
        // Within bonded distance
        return 0.0;
    } else if (dist > spline_start) {
        // lr spline fade

        Real x0 = spline_start;
        Real x1 = max_dis;

        auto x = dist;
        auto y0 = lj_spline_y0;
        auto dy0 = lj_spline_dy0;
        Real u0 = (3.0 / (x1 - x0)) * ((-y0) / (x1 - x0) - dy0);
        Real u1 = (3.0 / (x1 - x0)) * (y0 / (x1 - x0));

        lj = ((x - x1) * ((x - x0) * (u1 * (x0 - x) + u0 * (x - x1)) + 3.0 * y0))
             / (3.0 * (x0 - x1));
    } else if (dist > lj_switch_dis2sigma * lj_sigma) {
        // analytic 12-6

        Real invdist2 = 1.0 / (dist * dist);
        Real invdist6 = invdist2 * invdist2 * invdist2;
        Real invdist12 = invdist6 * invdist6;

        lj = (lj_coeff_sigma12 * invdist12) + (lj_coeff_sigma6 * invdist6);
    } else {
        // linear
        lj = dist * lj_switch_slope + lj_switch_intercept;
    }

    if (bonded_path_length == 4) {
        lj *= 0.2;
    }

    return lj;
}

}  // namespace ljlk
}  // namespace score
}  // namespace tmol
