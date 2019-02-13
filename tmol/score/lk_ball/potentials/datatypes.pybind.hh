#pragma once

#include <pybind11/pybind11.h>
#include <tmol/utility/tensor/pybind.h>

#include "datatypes.hh"
#include "lk_ball.hh"

namespace pybind11 {
namespace detail {

#define CAST_ATTR(SRC, TARGET, NAME)                             \
  try {                                                          \
    TARGET.NAME = SRC.attr(#NAME).cast<decltype(TARGET.NAME)>(); \
  } catch (pybind11::cast_error) {                               \
    pybind11::print("Error casting: ", #NAME);                   \
    return false;                                                \
  }

template <tmol::Device D>
struct type_caster<tmol::score::lk_ball::potentials::AtomTypes<D>> {
 public:
  typedef tmol::score::lk_ball::potentials::AtomTypes<D> T;

  PYBIND11_TYPE_CASTER(T, _<T>());

  bool load(handle src, bool convert) {
    CAST_ATTR(src, value, is_acceptor);
    CAST_ATTR(src, value, acceptor_hybridization);
    CAST_ATTR(src, value, is_donor);
    CAST_ATTR(src, value, is_hydrogen);

    return true;
  }
};

template <typename Real, tmol::Device D>
struct type_caster<
    tmol::score::lk_ball::potentials::LKBallGlobalParameters<Real, D>> {
 public:
  typedef tmol::score::lk_ball::potentials::LKBallGlobalParameters<Real, D> T;

  PYBIND11_TYPE_CASTER(T, _<T>());

  bool load(handle src, bool convert) {
    CAST_ATTR(src, value, lkb_water_dist);
    CAST_ATTR(src, value, lkb_water_angle_sp2);
    CAST_ATTR(src, value, lkb_water_angle_sp3);
    CAST_ATTR(src, value, lkb_water_angle_ring);
    CAST_ATTR(src, value, lkb_water_tors_sp2);
    CAST_ATTR(src, value, lkb_water_tors_sp3);
    CAST_ATTR(src, value, lkb_water_tors_ring);

    return true;
  }
};

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
