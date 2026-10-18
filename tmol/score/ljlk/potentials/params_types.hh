#pragma once

// Plain parameter structs, free of the torch tensor layer.
//
// Split out of params.hh so that device-side scalar math can include the
// struct definitions without dragging in TensorUtil.h (and through it Eigen
// and torch), which exist only for the enable_tensor_view specializations.
// A Metal kernel receives these as bound buffers and needs the layout, not
// the tensor machinery. See tmol/tests/utility/portable/.

#include <tmol/utility/portable/math.hh>

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template <typename Real>
struct LJTypeParams {
  Real lj_radius;
  Real lj_sqrt_wdepth;
  Real is_donor;
  Real is_hydroxyl;
  Real is_polarh;
  Real is_acceptor;
};

template <typename Real>
struct LKTypeParams {
  Real lj_radius;
  Real lk_dgfree;
  Real lk_lambda;
  Real lk_volume;
  Real is_donor;
  Real is_hydroxyl;
  Real is_polarh;
  Real is_acceptor;
  Real is_carbon_lk;
  Real lk_coeff;
  Real lk_inv_lambda2;
};

template <typename Real>
struct LJLKTypeParams {
  Real lj_radius;
  Real lj_sqrt_wdepth;
  Real lk_dgfree;
  Real lk_lambda;
  Real lk_volume;
  Real is_donor;
  Real is_hydroxyl;
  Real is_polarh;
  Real is_acceptor;
  Real is_carbon_lk;
  Real is_hydrogen;
  Real lk_coeff;
  Real lk_inv_lambda2;

  // The threadgroup overloads exist because these parameters are read out of
  // the workgroup's shared tile buffer, and in MSL the address space is part
  // of the implicit `this` pointer's type: a thread-space member function
  // cannot be called on a threadgroup object. They are the same code; only
  // the object's address space differs. See matrix.hh for the same pattern on
  // Mat's accessors.
  LJTypeParams<Real> TMOL_HD lj_params() const {
    return LJTypeParams<Real>(
        {lj_radius,
         lj_sqrt_wdepth,
         is_donor,
         is_hydroxyl,
         is_polarh,
         is_acceptor});
  }

#if TMOL_PORTABLE_METAL
  LJTypeParams<Real> TMOL_HD lj_params() const threadgroup {
    return LJTypeParams<Real>(
        {lj_radius,
         lj_sqrt_wdepth,
         is_donor,
         is_hydroxyl,
         is_polarh,
         is_acceptor});
  }
#endif

  LKTypeParams<Real> TMOL_HD lk_params() const {
    return LKTypeParams<Real>(
        {lj_radius,
         lk_dgfree,
         lk_lambda,
         lk_volume,
         is_donor,
         is_hydroxyl,
         is_polarh,
         is_acceptor,
         is_carbon_lk,
         lk_coeff,
         lk_inv_lambda2});
  }

#if TMOL_PORTABLE_METAL
  LKTypeParams<Real> TMOL_HD lk_params() const threadgroup {
    return LKTypeParams<Real>(
        {lj_radius,
         lk_dgfree,
         lk_lambda,
         lk_volume,
         is_donor,
         is_hydroxyl,
         is_polarh,
         is_acceptor,
         is_carbon_lk,
         lk_coeff,
         lk_inv_lambda2});
  }
#endif
};

template <typename Real>
struct LJGlobalParams {
  Real max_dis;
  Real lj_dlin_sigma_factor;
  Real lj_hbond_dis;
  Real lj_hbond_OH_donor_dis;
  Real lj_hbond_hdis;
};

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
