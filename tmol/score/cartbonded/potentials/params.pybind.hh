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
    return false;                                                \
  }

template <typename Real>
struct type_caster<tmol::score::ljlk::potentials::CartBondedLengthParams<Int>> {
 public:
  typedef tmol::score::ljlk::potentials::CartBondedLengthParams<Int> T;

  PYBIND11_TYPE_CASTER(T, _<T>());

  bool load(handle src, bool convert) {
    CAST_ATTR(src, value, atom_index_i);
    CAST_ATTR(src, value, atom_index_j);
    CAST_ATTR(src, value, param_index);

    return true;
  }
};

template <typename Real, tmol::Device D>
struct type_caster<tmol::score::ljlk::potentials::CartBondedAngleParams<Int>> {
 public:
  typedef tmol::score::ljlk::potentials::CartBondedAngleParams<Int> T;

  PYBIND11_TYPE_CASTER(T, _<T>());

  bool load(handle src, bool convert) {
    CAST_ATTR(src, value, atom_index_i);
    CAST_ATTR(src, value, atom_index_j);
    CAST_ATTR(src, value, atom_index_k);
    CAST_ATTR(src, value, param_index);

    return true;
  }
};

template <typename Real, tmol::Device D>
struct type_caster<
    tmol::score::ljlk::potentials::CartBondedTorsionParams<Int>> {
 public:
  typedef tmol::score::ljlk::potentials::CartBondedTorsionParams<Int> T;

  PYBIND11_TYPE_CASTER(T, _<T>());

  bool load(handle src, bool convert) {
    CAST_ATTR(src, value, atom_index_i);
    CAST_ATTR(src, value, atom_index_j);
    CAST_ATTR(src, value, atom_index_k);
    CAST_ATTR(src, value, atom_index_l);
    CAST_ATTR(src, value, param_index);

    return true;
  }
};

template <typename Real>
struct type_caster<
    tmol::score::ljlk::potentials::CartBondedHarmonicTypeParams<Real>> {
 public:
  typedef tmol::score::ljlk::potentials::CartBondedHarmonicTypeParams<Real> T;

  PYBIND11_TYPE_CASTER(T, _<T>());

  bool load(handle src, bool convert) {
    CAST_ATTR(src, value, K);
    CAST_ATTR(src, value, x0);

    return true;
  }
};

template <typename Real>
struct type_caster<
    tmol::score::ljlk::potentials::CartBondedPeriodicTypeParams<Real>> {
 public:
  typedef tmol::score::ljlk::potentials::CartBondedPeriodicTypeParams<Real> T;

  PYBIND11_TYPE_CASTER(T, _<T>());

  bool load(handle src, bool convert) {
    CAST_ATTR(src, value, K);
    CAST_ATTR(src, value, x0);
    CAST_ATTR(src, value, period);

    return true;
  }
};

template <typename Real>
struct type_caster<
    tmol::score::ljlk::potentials::CartBondedSinusoidalTypeParams<Real>> {
 public:
  typedef tmol::score::ljlk::potentials::CartBondedSinusoidalTypeParams<Real> T;

  PYBIND11_TYPE_CASTER(T, _<T>());

  bool load(handle src, bool convert) {
    CAST_ATTR(src, value, k1);
    CAST_ATTR(src, value, k2);
    CAST_ATTR(src, value, k3);
    CAST_ATTR(src, value, phi1);
    CAST_ATTR(src, value, phi2);
    CAST_ATTR(src, value, phi3);

    return true;
  }
};

#undef CAST_ATTR

}  // namespace detail
}  // namespace pybind11
