#pragma once

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
