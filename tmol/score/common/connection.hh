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

const int MAX_PATHS_FROM_CONN = 13;
const int NUM_INTER_RES_PATHS = 34;

// Given an integer, returns a tuple of two indices into the
// atom_paths_from_conn tensor. Using these these indices on two connected
// blocks will give you all combinations of paths that form lengths, angles, and
// torsions. Chose to make this a lookup table because it is clearer and faster
// to simply enumerate the possibilities.
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
