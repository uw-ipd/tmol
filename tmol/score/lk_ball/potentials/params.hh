#pragma once

#include <tmol/utility/tensor/TensorPack.h>

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

// same (for now) as LKTypeParams
template <typename Real>
struct LKBallTypeParams {
  Real lj_radius;
  Real lk_dgfree;
  Real lk_lambda;
  Real lk_volume;
  Real is_donor;
  Real is_hydroxyl;
  Real is_polarh;
  Real is_acceptor;
};

// same (for now) as LJGlobalParams
template <typename Real>
struct LKBallGlobalParams {
  Real lj_hbond_dis;
  Real lj_hbond_OH_donor_dis;
  Real lj_hbond_hdis;
  Real lkb_water_dist;  // needed by both watergen and scoring
};

template <typename Int>
struct LKBallWaterGenTypeParams {
  Int is_acceptor;
  Int acceptor_hybridization;
  Int is_donor;
  Int is_hydrogen;
};

template <typename Real>
struct LKBallWaterGenGlobalParams {
  Real lkb_water_dist;
  Real lkb_water_angle_sp2;
  Real lkb_water_angle_sp3;
  Real lkb_water_angle_ring;
};

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol

namespace tmol {

// fd  this could probably be a macro?
template <typename Real>
struct enable_tensor_view<score::lk_ball::potentials::LKBallTypeParams<Real>> {
  static const bool enabled = enable_tensor_view<Real>::enabled;
  static const at::ScalarType scalar_type() {
    return enable_tensor_view<Real>::scalar_type();
  }

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0) ? sizeof(score::lk_ball::potentials::LKBallTypeParams<Real>)
                          / sizeof(Real)
                    : 0;
  }

  typedef typename enable_tensor_view<Real>::PrimitiveType PrimitiveType;
};

template <typename Real>
struct enable_tensor_view<
    score::lk_ball::potentials::LKBallGlobalParams<Real>> {
  static const bool enabled = enable_tensor_view<Real>::enabled;
  static const at::ScalarType scalar_type() {
    return enable_tensor_view<Real>::scalar_type();
  }

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0)
               ? sizeof(score::lk_ball::potentials::LKBallGlobalParams<Real>)
                     / sizeof(Real)
               : 0;
  }

  typedef typename enable_tensor_view<Real>::PrimitiveType PrimitiveType;
};

template <typename Int>
struct enable_tensor_view<
    score::lk_ball::potentials::LKBallWaterGenTypeParams<Int>> {
  static const bool enabled = enable_tensor_view<Int>::enabled;
  static const at::ScalarType scalar_type() {
    return enable_tensor_view<Int>::scalar_type();
  }

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0)
               ? sizeof(
                     score::lk_ball::potentials::LKBallWaterGenTypeParams<Int>)
                     / sizeof(Int)
               : 0;
  }

  typedef typename enable_tensor_view<Int>::PrimitiveType PrimitiveType;
};

template <typename Real>
struct enable_tensor_view<
    score::lk_ball::potentials::LKBallWaterGenGlobalParams<Real>> {
  static const bool enabled = enable_tensor_view<Real>::enabled;
  static const at::ScalarType scalar_type() {
    return enable_tensor_view<Real>::scalar_type();
  }

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0)
               ? sizeof(score::lk_ball::potentials::LKBallWaterGenGlobalParams<
                        Real>)
                     / sizeof(Real)
               : 0;
  }

  typedef typename enable_tensor_view<Real>::PrimitiveType PrimitiveType;
};

}  // namespace tmol
