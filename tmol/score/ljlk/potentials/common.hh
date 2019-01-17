#pragma once

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template <typename Real>
struct LJTypeParams {
  Real lj_radius;
  Real lj_wdepth;
  bool is_donor;
  bool is_hydroxyl;
  bool is_polarh;
  bool is_acceptor;
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

template <typename Real>
struct LJGlobalParams {
  Real lj_hbond_dis;
  Real lj_hbond_OH_donor_dis;
  Real lj_hbond_hdis;
};

template <typename Real, typename Int>
Real connectivity_weight(Int bonded_path_length) {
  if (bonded_path_length > 4) {
    return 1.0;
  } else if (bonded_path_length == 4) {
    return 0.2;
  } else {
    return 0.0;
  }
}

template <typename Real, typename TypeParams, typename GlobalParams>
Real lj_sigma(
    TypeParams i, TypeParams j, GlobalParams global) {
  if ((i.is_donor && !i.is_hydroxyl && j.is_acceptor)
      || (j.is_donor && !j.is_hydroxyl && i.is_acceptor)) {
    // standard donor/acceptor pair
    return global.lj_hbond_dis;
  } else if (
      (i.is_donor && i.is_hydroxyl && j.is_acceptor)
      || (j.is_donor && j.is_hydroxyl && i.is_acceptor)) {
    // hydroxyl donor/acceptor pair
    return global.lj_hbond_OH_donor_dis;
  } else if ((i.is_polarh && j.is_acceptor) || (j.is_polarh && i.is_acceptor)) {
    // hydrogen/acceptor pair
    return global.lj_hbond_hdis;
  } else {
    // standard lj
    return i.lj_radius + j.lj_radius;
  }
}

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
