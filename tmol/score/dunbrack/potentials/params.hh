#pragma once

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>

namespace tmol {
namespace score {
namespace dunbrack {
namespace potentials {

template <int BINS_PER_DIH, typename T>
constexpr T MAXROTBINS(T exponent) {
  T retval = BINS_PER_DIH;
  for (int i = 1; i < exponent; ++i) {
    retval *= BINS_PER_DIH;
  }
  return retval;
}

// ROTAMERIC table metadata
template <typename Real, int MAXBB>
struct RotamericTableParams {
  Real bbstarts[MAXBB];
  Real bbsteps[MAXBB];
};

// SEMIROTAMERIC table metadata
template <typename Real, int MAXBB>
struct SemirotamericTableParams {
  Real bbstarts[MAXBB + 1];
  Real bbsteps[MAXBB + 1];
};

// per-RESIDUE parameters
template <typename Int, int MAXBB, int MAXCHI>
struct DunResParameters {
  Int bb_indices[MAXBB][4];
  Int chi_indices[MAXCHI][4];
  Int aa_index;
};

// per-AA-TYPE parameters
template <typename Int, int MAXCHI>
struct DunTableLookupParams {
  // map rotamer ids -> rotameric prob table
  Int rotidx2probtableidx[MAXROTBINS<3>(MAXCHI)];

  // map rotamer ids -> rotameric mean/stdev table
  // note this is defined for both semirot and rot
  Int rotidx2meantableidx[MAXROTBINS<3>(MAXCHI)];

  // map semirotamer ids -> rotameric prob table
  Int semirotidx2probtableidx[MAXROTBINS<3>(MAXCHI)];

  Int nrotchi;     // # rotameric chis
  Int semirotchi;  // # chis
};

}  // namespace potentials
}  // namespace dunbrack
}  // namespace score
}  // namespace tmol

namespace tmol {

template <typename Real, int MAXBB>
struct enable_tensor_view<
    score::dunbrack::potentials::RotamericTableParams<Real, MAXBB>> {
  static const bool enabled = enable_tensor_view<Real>::enabled;
  static const at::ScalarType scalar_type =
      enable_tensor_view<Real>::scalar_type;

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0) ? sizeof(score::dunbrack::potentials::
                                 RotamericTableParams<Real, MAXBB>)
                          / sizeof(Real)
                    : 0;
  }

  typedef typename enable_tensor_view<Real>::PrimitiveType PrimitiveType;
};

template <typename Real, int MAXBB>
struct enable_tensor_view<
    score::dunbrack::potentials::SemirotamericTableParams<Real, MAXBB>> {
  static const bool enabled = enable_tensor_view<Real>::enabled;
  static const at::ScalarType scalar_type =
      enable_tensor_view<Real>::scalar_type;

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0) ? sizeof(score::dunbrack::potentials::
                                 SemirotamericTableParams<Real, MAXBB>)
                          / sizeof(Real)
                    : 0;
  }

  typedef typename enable_tensor_view<Real>::PrimitiveType PrimitiveType;
};

template <typename Int, int MAXBB, int MAXCHI>
struct enable_tensor_view<
    score::dunbrack::potentials::DunResParameters<Int, MAXBB, MAXCHI>> {
  static const bool enabled = enable_tensor_view<Int>::enabled;
  static const at::ScalarType scalar_type =
      enable_tensor_view<Int>::scalar_type;

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0) ? sizeof(score::dunbrack::potentials::
                                 DunResParameters<Int, MAXBB, MAXCHI>)
                          / sizeof(Int)
                    : 0;
  }

  typedef typename enable_tensor_view<Int>::PrimitiveType PrimitiveType;
};

template <typename Int, int MAXCHI>
struct enable_tensor_view<
    score::dunbrack::potentials::DunTableLookupParams<Int, MAXCHI>> {
  static const bool enabled = enable_tensor_view<Int>::enabled;
  static const at::ScalarType scalar_type =
      enable_tensor_view<Int>::scalar_type;

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0) ? sizeof(score::dunbrack::potentials::
                                 DunTableLookupParams<Int, MAXCHI>)
                          / sizeof(Int)
                    : 0;
  }

  typedef typename enable_tensor_view<Int>::PrimitiveType PrimitiveType;
};

}  // namespace tmol
