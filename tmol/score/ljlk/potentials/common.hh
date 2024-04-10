#pragma once

#include <Eigen/src/Core/util/Macros.h>
#include "params.hh"

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

#define def auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE

template <typename Real, typename Int>
def connectivity_weight(Int bonded_path_length)->Real {
  if (bonded_path_length > 4) {
    return 1.0;
  } else if (bonded_path_length == 4) {
    return 0.2;
  } else {
    return 0.0;
  }
}

template <typename Real, typename TypeParams, typename GlobalParams>
def lj_sigma(TypeParams i, TypeParams j, GlobalParams global)->Real {
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

#undef def

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
