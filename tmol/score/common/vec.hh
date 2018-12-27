#pragma once

#include <Eigen/Core>

namespace tmol {
namespace score {
namespace common {

template <int N, typename Real>
using Vec = Eigen::Matrix<Real, N, 1>;

}
}
}
