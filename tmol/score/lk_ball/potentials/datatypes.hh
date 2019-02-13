#pragma once

#include <tmol/utility/tensor/TensorPack.h>

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

template <typename Real, tmol::Device D>
struct LKBallGlobalParameters {
  Real lkb_water_dist;
  Real lkb_water_angle_sp2;
  Real lkb_water_angle_sp3;
  Real lkb_water_angle_ring;
  TView<Real, 1, D> lkb_water_tors_sp2;
  TView<Real, 1, D> lkb_water_tors_sp3;
  TView<Real, 1, D> lkb_water_tors_ring;
};

template <tmol::Device D>
struct AtomTypes {
  TView<bool, 1, D> is_acceptor;
  TView<int32_t, 1, D> acceptor_hybridization;
  TView<bool, 1, D> is_donor;
  TView<bool, 1, D> is_hydrogen;
};

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
