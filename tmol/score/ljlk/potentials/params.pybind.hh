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
struct type_caster<tmol::score::ljlk::potentials::LJTypeParams<Real>> {
 public:
  typedef tmol::score::ljlk::potentials::LJTypeParams<Real> T;

  PYBIND11_TYPE_CASTER(T, _<T>());

  bool load(handle src, bool convert) {
    nvtx_range_function();

    CAST_ATTR(src, value, lj_radius);
    CAST_ATTR(src, value, lj_wdepth);
    CAST_ATTR(src, value, is_donor);
    CAST_ATTR(src, value, is_acceptor);
    CAST_ATTR(src, value, is_hydroxyl);
    CAST_ATTR(src, value, is_polarh);

    return true;
  }
};

template <typename Real, tmol::Device D>
struct type_caster<tmol::score::ljlk::potentials::LJTypeParamTensors<Real, D>> {
 public:
  typedef tmol::score::ljlk::potentials::LJTypeParamTensors<Real, D> T;

  PYBIND11_TYPE_CASTER(T, _<T>());

  bool load(handle src, bool convert) {
    nvtx_range_function();

    CAST_ATTR(src, value, lj_radius);
    CAST_ATTR(src, value, lj_wdepth);
    CAST_ATTR(src, value, is_donor);
    CAST_ATTR(src, value, is_acceptor);
    CAST_ATTR(src, value, is_hydroxyl);
    CAST_ATTR(src, value, is_polarh);

    return true;
  }
};

template <typename Real>
struct type_caster<tmol::score::ljlk::potentials::LKTypeParams<Real>> {
 public:
  typedef tmol::score::ljlk::potentials::LKTypeParams<Real> T;

  PYBIND11_TYPE_CASTER(T, _<T>());

  bool load(handle src, bool convert) {
    nvtx_range_function();

    CAST_ATTR(src, value, lj_radius);
    CAST_ATTR(src, value, lk_dgfree);
    CAST_ATTR(src, value, lk_lambda);
    CAST_ATTR(src, value, lk_volume);
    CAST_ATTR(src, value, is_donor);
    CAST_ATTR(src, value, is_acceptor);
    CAST_ATTR(src, value, is_hydroxyl);
    CAST_ATTR(src, value, is_polarh);

    return true;
  }
};

template <typename Real, tmol::Device D>
struct type_caster<tmol::score::ljlk::potentials::LKTypeParamTensors<Real, D>> {
 public:
  typedef tmol::score::ljlk::potentials::LKTypeParamTensors<Real, D> T;

  PYBIND11_TYPE_CASTER(T, _<T>());

  bool load(handle src, bool convert) {
    nvtx_range_function();

    CAST_ATTR(src, value, lj_radius);
    CAST_ATTR(src, value, lk_dgfree);
    CAST_ATTR(src, value, lk_lambda);
    CAST_ATTR(src, value, lk_volume);
    CAST_ATTR(src, value, is_donor);
    CAST_ATTR(src, value, is_acceptor);
    CAST_ATTR(src, value, is_hydroxyl);
    CAST_ATTR(src, value, is_polarh);

    return true;
  }
};

template <typename Real>
struct type_caster<tmol::score::ljlk::potentials::LJGlobalParams<Real>> {
 public:
  typedef tmol::score::ljlk::potentials::LJGlobalParams<Real> T;

  PYBIND11_TYPE_CASTER(T, _<T>());

  bool load(handle src, bool convert) {
    nvtx_range_function();

    CAST_ATTR(src, value, lj_dlin_sigma_factor);
    CAST_ATTR(src, value, lj_hbond_dis);
    CAST_ATTR(src, value, lj_hbond_OH_donor_dis);
    CAST_ATTR(src, value, lj_hbond_hdis);

    return true;
  }
};

template <typename Real, tmol::Device D>
struct type_caster<
    tmol::score::ljlk::potentials::LJGlobalParamTensors<Real, D>> {
 public:
  typedef tmol::score::ljlk::potentials::LJGlobalParamTensors<Real, D> T;

  PYBIND11_TYPE_CASTER(T, _<T>());

  bool load(handle src, bool convert) {
    nvtx_range_function();

    CAST_ATTR(src, value, lj_dlin_sigma_factor);
    CAST_ATTR(src, value, lj_hbond_dis);
    CAST_ATTR(src, value, lj_hbond_OH_donor_dis);
    CAST_ATTR(src, value, lj_hbond_hdis);

    return true;
  }
};

#undef CAST_ATTR

}  // namespace detail
}  // namespace pybind11
