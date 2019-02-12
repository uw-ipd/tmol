#pragma once

#include <pybind11/eigen.h>
#include <tmol/utility/tensor/pybind.h>
#include "water.hh"

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

}  // namespace detail
}  // namespace pybind11
