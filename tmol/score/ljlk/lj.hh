#include <Eigen/Core>

/////Utility macros to support parameter passthrough in operators

// pybind11 parameter arguments
#define LJ_PARAM_PYARGS                                                   \
  "bonded_path_length"_a, "lj_sigma"_a, "lj_switch_slope"_a,              \
      "lj_switch_intercept"_a, "lj_coeff_sigma12"_a, "lj_coeff_sigma6"_a, \
      "lj_spline_y0"_a, "lj_spline_dy0"_a, "lj_switch_dis2sigma"_a,       \
      "spline_start"_a, "max_dis"_a

// c++ level parameter tensors
#define LJ_PARAM_ARGS                                                 \
  at::Tensor bonded_path_length_t, at::Tensor lj_sigma_t,             \
      at::Tensor lj_switch_slope_t, at::Tensor lj_switch_intercept_t, \
      at::Tensor lj_coeff_sigma12_t, at::Tensor lj_coeff_sigma6_t,    \
      at::Tensor lj_spline_y0_t, at::Tensor lj_spline_dy0_t,          \
      at::Tensor lj_switch_dis2sigma_t, at::Tensor spline_start_t,    \
      at::Tensor max_dis_t

// unpack parameter tensors into TensorAccessors
#define LJ_PARAM_UNPACK                                                      \
  auto bonded_path_length = tmol::view_tensor<PathLength, 2>(bonded_path_length_t); \
                                                                             \
  auto lj_sigma = tmol::view_tensor<Real, 2>(lj_sigma_t);                    \
  auto lj_switch_slope = tmol::view_tensor<Real, 2>(lj_switch_slope_t);      \
  auto lj_switch_intercept =                                                 \
      tmol::view_tensor<Real, 2>(lj_switch_intercept_t);                     \
  auto lj_coeff_sigma12 = tmol::view_tensor<Real, 2>(lj_coeff_sigma12_t);    \
  auto lj_coeff_sigma6 = tmol::view_tensor<Real, 2>(lj_coeff_sigma6_t);      \
  auto lj_spline_y0 = tmol::view_tensor<Real, 2>(lj_spline_y0_t);            \
  auto lj_spline_dy0 = tmol::view_tensor<Real, 2>(lj_spline_dy0_t);          \
  auto lj_switch_dis2sigma =                                                 \
      tmol::view_tensor<Real, 1>(lj_switch_dis2sigma_t.reshape(1));          \
  auto spline_start = tmol::view_tensor<Real, 1>(spline_start_t.reshape(1)); \
  auto max_dis = tmol::view_tensor<Real, 1>(max_dis_t.reshape(1));

namespace tmol {
namespace score {
namespace lj {
template <typename Real, typename PathLength>
EIGEN_DEVICE_FUNC Real
lj(Real& dist,
   const PathLength& bonded_path_length,
   const Real& lj_sigma,
   const Real& lj_switch_slope,
   const Real& lj_switch_intercept,
   const Real& lj_coeff_sigma12,
   const Real& lj_coeff_sigma6,
   const Real& lj_spline_y0,
   const Real& lj_spline_dy0,
   const Real& lj_switch_dis2sigma,
   const Real& spline_start,
   const Real& max_dis) {
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

template <typename Real, typename Int>
EIGEN_DEVICE_FUNC Real d_lj_d_dist(
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
    const Real& max_dis) {
  Real d_lj_d_dist = 0.0;

  if (dist > max_dis) {
    // Outside of interaction distance
    return 0.0;
  } else if (bonded_path_length < 4) {
    // Within bonded distance
    return 0.0;
  } else if (dist > spline_start) {
    // lr spline fade

    // Subexpressions
    Real x0 = -spline_start;
    Real x1 = max_dis + x0;
    Real x1_2 = x1 * x1;
    Real x2 = dist + x0;
    Real x3 = lj_spline_y0 * x2;
    Real x4 = dist - max_dis;
    Real x5 = lj_spline_dy0 * x1;
    Real x6 = x4 * (lj_spline_y0 + x5);

    // Expression
    d_lj_d_dist =
        0.33333333333333331
        * (-3.0 * lj_spline_y0 * x1_2 + 3.0 * x2 * (x3 + x6)
           + x4 * (x2 * (6.0 * lj_spline_y0 + 3.0 * x5) + 3.0 * x3 + 3.0 * x6))
        / (x1 * x1_2);

  } else if (dist > lj_switch_dis2sigma * lj_sigma) {
    // analytic 12-6
    Real invdist = 1.0 / dist;
    Real invdist2 = invdist * invdist;
    Real invdist6 = invdist2 * invdist2 * invdist2;
    Real invdist12 = invdist6 * invdist6;

    d_lj_d_dist = (-6.0 * lj_coeff_sigma6 * invdist6 * invdist)
                  + (-12.0 * lj_coeff_sigma12 * invdist12 * invdist);
  } else {
    // linear
    d_lj_d_dist = lj_switch_slope;
  }

  if (bonded_path_length == 4) {
    d_lj_d_dist *= 0.2;
  }

  return d_lj_d_dist;
}

}  // namespace lj
}  // namespace score
}  // namespace tmol
