#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/score/common/tuple.hh>

#include <tmol/score/common/diamond_macros.hh>

namespace tmol {
namespace score {
namespace common {

template <typename Int>
TMOL_DEVICE_FUNC tuple<Int, Int> get_connection_spanning_subgraph_indices(
    Int index) {
  static const int CON_PATH_INDICES[][2] = {// Length
                                            {0, 0},

                                            // Angles
                                            {0, 1},
                                            {0, 2},
                                            {0, 3},
                                            {1, 0},
                                            {2, 0},
                                            {3, 0},

                                            // Torsions
                                            {0, 4},
                                            {0, 5},
                                            {0, 6},
                                            {0, 7},
                                            {0, 8},
                                            {0, 9},
                                            {0, 10},
                                            {0, 11},
                                            {0, 12},

                                            {1, 1},
                                            {1, 2},
                                            {1, 3},
                                            {2, 1},
                                            {2, 2},
                                            {2, 3},
                                            {3, 1},
                                            {3, 2},
                                            {3, 3},

                                            {4, 0},
                                            {5, 0},
                                            {6, 0},
                                            {7, 0},
                                            {8, 0},
                                            {9, 0},
                                            {10, 0},
                                            {11, 0},
                                            {12, 0}};
  return {CON_PATH_INDICES[index][0], CON_PATH_INDICES[index][1]};
}

}  // namespace common
}  // namespace score
}  // namespace tmol
