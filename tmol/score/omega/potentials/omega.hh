#pragma once

#include <Eigen/Core>
#include <cmath>

#include <tmol/score/common/cubic_hermite_polynomial.hh>
#include <tmol/score/common/tuple.hh>

namespace tmol {
namespace score {
namespace omega {
namespace potentials {

#define def auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE

using namespace tmol::score::common;

#undef def

}  // namespace potentials
}  // namespace omega
}  // namespace score
}  // namespace tmol
