#pragma once

// Plain parameter structs, free of the torch tensor layer.
//
// Split out of params.hh so that device-side scalar math can include the
// struct definitions without dragging in TensorUtil.h (and through it Eigen
// and torch), which exist only for the enable_tensor_view specializations.
// A Metal kernel receives these as bound buffers and needs the layout, not
// the tensor machinery. See tmol/tests/utility/portable/.

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

// same (for now) as LKTypeParams
template <typename Real>
struct LKBallTypeParams {
  Real lj_radius;
  Real lk_coeff;
  Real lk_inv_lambda2;
  Real lk_volume;
  Real is_donor;
  Real is_hydroxyl;
  Real is_polarh;
  Real is_acceptor;
  Real is_carbon_lk;
};

template <typename Real>
struct LKBallGlobalParams {
  Real lj_hbond_dis;
  Real lj_hbond_OH_donor_dis;
  Real lj_hbond_hdis;
  Real lkb_water_dist;      // needed by both watergen and scoring
  Real distance_threshold;  // maximum distance two heavy-atoms can be to have a
                            // non-zero interaction energy
};

template <typename Int>
struct LKBallWaterGenTypeParams {
  Int is_acceptor;
  Int acceptor_hybridization;
  Int is_donor;
  Int is_hydrogen;
};

template <typename Real>
struct LKBallWaterGenGlobalParams {
  Real lkb_water_dist;
  Real lkb_water_angle_sp2;
  Real lkb_water_angle_sp3;
  Real lkb_water_angle_ring;
};

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
