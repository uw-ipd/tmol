#pragma once

#include <pybind11/pybind11.h>
#include <tmol/utility/tensor/pybind.h>

#include "lk_ball.hh"

namespace pybind11 {
namespace detail {

template <typename Real>
struct type_caster<tmol::score::lk_ball::potentials::lk_ball_Vt<Real>> {
  typedef tmol::score::lk_ball::potentials::lk_ball_Vt<Real> T;
  PYBIND11_TYPE_CASTER(T, _<T>());

  static handle cast(
      T src, return_value_policy /* policy */, handle /* parent */) {
    return pybind11::make_tuple(
               src.lkball_iso, src.lkball, src.lkbridge, src.lkbridge_uncpl)
        .release();
  }
};

template <typename Real>
struct type_caster<tmol::score::lk_ball::potentials::lk_ball_dV_dReal3<Real>> {
  typedef tmol::score::lk_ball::potentials::lk_ball_dV_dReal3<Real> T;
  PYBIND11_TYPE_CASTER(T, _<T>());

  static handle cast(
      T src, return_value_policy /* policy */, handle /* parent */) {
    return pybind11::make_tuple(
               src.d_lkball_iso,
               src.d_lkball,
               src.d_lkbridge,
               src.d_lkbridge_uncpl)
        .release();
  }
};

template <typename Real, int MAX_WATER>
struct type_caster<
    tmol::score::lk_ball::potentials::lk_ball_dV_dWater<Real, MAX_WATER>> {
  typedef tmol::score::lk_ball::potentials::lk_ball_dV_dWater<Real, MAX_WATER>
      T;
  PYBIND11_TYPE_CASTER(T, _<T>());

  static handle cast(
      T src, return_value_policy /* policy */, handle /* parent */) {
    return pybind11::make_tuple(
               src.d_lkball_iso,
               src.d_lkball,
               src.d_lkbridge,
               src.d_lkbridge_uncpl)
        .release();
  }
};

template <typename Real, int MAX_WATER>
struct type_caster<
    tmol::score::lk_ball::potentials::lk_ball_dVt<Real, MAX_WATER>> {
  typedef tmol::score::lk_ball::potentials::lk_ball_dVt<Real, MAX_WATER> T;
  PYBIND11_TYPE_CASTER(T, _<T>());

  static handle cast(
      T src, return_value_policy /* policy */, handle /* parent */) {
    return pybind11::make_tuple(src.dI, src.dJ, src.dWI, src.dWJ).release();
  }
};
}  // namespace detail
}  // namespace pybind11
