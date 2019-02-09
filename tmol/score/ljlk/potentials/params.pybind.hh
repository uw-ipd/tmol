#pragma once

#include <pybind11/pybind11.h>
#include <tmol/utility/tensor/pybind.h>

#include "params.hh"

namespace pybind11 {
namespace detail {

template <typename Real>
struct type_caster<tmol::score::ljlk::potentials::LJTypeParams<Real>> {
 public:
  typedef tmol::score::ljlk::potentials::LJTypeParams<Real> T;

  PYBIND11_TYPE_CASTER(T, _<T>());

  bool load(handle src, bool convert) {
    using namespace pybind11::detail;

    value = T{
        src.attr("lj_radius").cast<decltype(T::lj_radius)>(),
        src.attr("lj_wdepth").cast<decltype(T::lj_wdepth)>(),
        src.attr("is_donor").cast<decltype(T::is_donor)>(),
        src.attr("is_hydroxyl").cast<decltype(T::is_hydroxyl)>(),
        src.attr("is_polarh").cast<decltype(T::is_polarh)>(),
        src.attr("is_acceptor").cast<decltype(T::is_acceptor)>(),
    };

    return true;
  }
};

template <typename Real, tmol::Device D>
struct type_caster<tmol::score::ljlk::potentials::LJTypeParamTensors<Real, D>> {
 public:
  typedef tmol::score::ljlk::potentials::LJTypeParamTensors<Real, D> T;

  PYBIND11_TYPE_CASTER(T, _<T>());

  bool load(handle src, bool convert) {
    using namespace pybind11::detail;

    value = T{
        src.attr("lj_radius").cast<decltype(T::lj_radius)>(),
        src.attr("lj_wdepth").cast<decltype(T::lj_wdepth)>(),
        src.attr("is_donor").cast<decltype(T::is_donor)>(),
        src.attr("is_hydroxyl").cast<decltype(T::is_hydroxyl)>(),
        src.attr("is_polarh").cast<decltype(T::is_polarh)>(),
        src.attr("is_acceptor").cast<decltype(T::is_acceptor)>(),
    };

    return true;
  }
};

template <typename Real>
struct type_caster<tmol::score::ljlk::potentials::LKTypeParams<Real>> {
 public:
  typedef tmol::score::ljlk::potentials::LKTypeParams<Real> T;

  PYBIND11_TYPE_CASTER(T, _<T>());

  bool load(handle src, bool convert) {
    using namespace pybind11::detail;

    value = T{
        src.attr("lj_radius").cast<decltype(T::lj_radius)>(),
        src.attr("lk_dgfree").cast<decltype(T::lk_dgfree)>(),
        src.attr("lk_lambda").cast<decltype(T::lk_lambda)>(),
        src.attr("lk_volume").cast<decltype(T::lk_volume)>(),
        src.attr("is_donor").cast<decltype(T::is_donor)>(),
        src.attr("is_hydroxyl").cast<decltype(T::is_hydroxyl)>(),
        src.attr("is_polarh").cast<decltype(T::is_polarh)>(),
        src.attr("is_acceptor").cast<decltype(T::is_acceptor)>(),
    };

    return true;
  }
};

template <typename Real, tmol::Device D>
struct type_caster<tmol::score::ljlk::potentials::LKTypeParamTensors<Real, D>> {
 public:
  typedef tmol::score::ljlk::potentials::LKTypeParamTensors<Real, D> T;

  PYBIND11_TYPE_CASTER(T, _<T>());

  bool load(handle src, bool convert) {
    using namespace pybind11::detail;

    value = T{
        src.attr("lj_radius").cast<decltype(T::lj_radius)>(),
        src.attr("lk_dgfree").cast<decltype(T::lk_dgfree)>(),
        src.attr("lk_lambda").cast<decltype(T::lk_lambda)>(),
        src.attr("lk_volume").cast<decltype(T::lk_volume)>(),
        src.attr("is_donor").cast<decltype(T::is_donor)>(),
        src.attr("is_hydroxyl").cast<decltype(T::is_hydroxyl)>(),
        src.attr("is_polarh").cast<decltype(T::is_polarh)>(),
        src.attr("is_acceptor").cast<decltype(T::is_acceptor)>(),
    };

    return true;
  }
};

template <typename Real>
struct type_caster<tmol::score::ljlk::potentials::LJGlobalParams<Real>> {
 public:
  typedef tmol::score::ljlk::potentials::LJGlobalParams<Real> T;

  PYBIND11_TYPE_CASTER(T, _<T>());

  bool load(handle src, bool convert) {
    using namespace pybind11::detail;

    value = T{
        src.attr("lj_hbond_dis").cast<decltype(T::lj_hbond_dis)>(),
        src.attr("lj_hbond_OH_donor_dis")
            .cast<decltype(T::lj_hbond_OH_donor_dis)>(),
        src.attr("lj_hbond_hdis").cast<decltype(T::lj_hbond_hdis)>(),
    };

    return true;
  }
};

}  // namespace detail
}  // namespace pybind11
