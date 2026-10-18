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
namespace elec {
namespace potentials {

template <typename Real>
struct ElecGlobalParams {
  Real D;
  Real D0;
  Real S;
  Real min_dis;
  Real max_dis;
  Real cutoff_offset;
  Real min_score;
  Real low_score;
  Real low_deriv;
  Real high_score;
  Real high_deriv;
};

}  // namespace potentials
}  // namespace elec
}  // namespace score
}  // namespace tmol
