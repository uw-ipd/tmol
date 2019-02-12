#pragma once

#include <tmol/utility/tensor/TensorAccessor.h>

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {
#define _STR(v) #v

template <typename Real>
struct LJTypeParams {
  Real lj_radius;
  Real lj_wdepth;
  bool is_donor;
  bool is_hydroxyl;
  bool is_polarh;
  bool is_acceptor;
};

template <typename Real, tmol::Device D>
struct LJTypeParamTensors {
  TView<Real, 1, D> lj_radius;
  TView<Real, 1, D> lj_wdepth;
  TView<bool, 1, D> is_donor;
  TView<bool, 1, D> is_hydroxyl;
  TView<bool, 1, D> is_polarh;
  TView<bool, 1, D> is_acceptor;

  template <typename Idx>
  auto operator[](Idx i) const {
    return LJTypeParams<Real>{lj_radius[i],
                              lj_wdepth[i],
                              is_donor[i],
                              is_hydroxyl[i],
                              is_polarh[i],
                              is_acceptor[i]};
  }
};

template <typename Real>
struct LKTypeParams {
  Real lj_radius;
  Real lk_dgfree;
  Real lk_lambda;
  Real lk_volume;
  bool is_donor;
  bool is_hydroxyl;
  bool is_polarh;
  bool is_acceptor;
};

template <typename Real, tmol::Device D>
struct LKTypeParamTensors {
  TView<Real, 1, D> lj_radius;
  TView<Real, 1, D> lk_dgfree;
  TView<Real, 1, D> lk_lambda;
  TView<Real, 1, D> lk_volume;
  TView<bool, 1, D> is_donor;
  TView<bool, 1, D> is_hydroxyl;
  TView<bool, 1, D> is_polarh;
  TView<bool, 1, D> is_acceptor;

  template <typename Idx>
  auto operator[](Idx i) const {
    return LKTypeParams<Real>{lj_radius[i],
                              lk_dgfree[i],
                              lk_lambda[i],
                              lk_volume[i],
                              is_donor[i],
                              is_hydroxyl[i],
                              is_polarh[i],
                              is_acceptor[i]};
  }
};

template <typename Real>
struct LJGlobalParams {
  Real lj_hbond_dis;
  Real lj_hbond_OH_donor_dis;
  Real lj_hbond_hdis;
};

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
