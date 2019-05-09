#pragma once

#include <pybind11/pybind11.h>
#include <tmol/utility/tensor/pybind.h>

#include "params.hh"

namespace pybind11 {
namespace detail {

#define CAST_ATTR(SRC, TARGET, NAME)                             \
  try {                                                          \
    TARGET.NAME = SRC.attr(#NAME).cast<decltype(TARGET.NAME)>(); \
  } catch (pybind11::cast_error) {                               \
    pybind11::print("Error casting: ", #NAME);                   \
    return false;                                                \
  }

template <typename Real>
struct type_caster<tmol::score::ljlk::potentials::LKBallTypeParams<Real>> {
 public:
  typedef tmol::score::ljlk::potentials::LKBallTypeParams<Real> T;

  PYBIND11_TYPE_CASTER(T, _<T>());

  bool load(handle src, bool convert) {
    nvtx_range_function();

    CAST_ATTR(src, value, lj_radius);
    CAST_ATTR(src, value, lk_dgfree);
    CAST_ATTR(src, value, lk_lambda);
    CAST_ATTR(src, value, lk_volume);
    CAST_ATTR(src, value, is_donor);
    CAST_ATTR(src, value, is_hydroxyl);
    CAST_ATTR(src, value, is_polarh);
    CAST_ATTR(src, value, is_acceptor);

    return true;
  }
};

template <typename Real>
struct type_caster<tmol::score::ljlk::potentials::LKBallGlobalParams<Real>> {
 public:
  typedef tmol::score::ljlk::potentials::LKBallGlobalParams<Real> T;

  PYBIND11_TYPE_CASTER(T, _<T>());

  bool load(handle src, bool convert) {
    nvtx_range_function();

    CAST_ATTR(src, value, lj_hbond_dis);
    CAST_ATTR(src, value, lj_hbond_OH_donor_dis);
    CAST_ATTR(src, value, lj_hbond_hdis);

    return true;
  }
};

template <typename Int>
struct type_caster<
    tmol::score::ljlk::potentials::LKBallWaterGenTypeParams<Int>> {
 public:
  typedef tmol::score::ljlk::potentials::LKBallWaterGenTypeParams<Int> T;

  PYBIND11_TYPE_CASTER(T, _<T>());

  bool load(handle src, bool convert) {
    nvtx_range_function();

    CAST_ATTR(src, value, is_acceptor);
    CAST_ATTR(src, value, acceptor_hybridization);
    CAST_ATTR(src, value, is_donor);
    CAST_ATTR(src, value, is_hydrogen);

    return true;
  }
};

template <typename Real, int MAX_WATER>
struct type_caster<tmol::score::ljlk::potentials::
                       LKBallWaterGenGlobalParams<Real, MAX_WATER>> {
 public:
  typedef tmol::score::ljlk::potentials::
      LKBallWaterGenGlobalParams<Real, MAX_WATER>
          T;

  PYBIND11_TYPE_CASTER(T, _<T>());

  bool load(handle src, bool convert) {
    nvtx_range_function();

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

}  // namespace detail
}  // namespace pybind11
