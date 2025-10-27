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

enum subgraph_type {
  subgraph_length = 0,
  subgraph_angle,
  subgraph_torsion
};

// Given an integer, returns a tuple of three integers, a type and two indices into the
// atom_paths_from_conn tensor. Using these indices on two connected
// blocks will give you all combinations of paths that form lengths, angles, and
// torsions. Chose to make this a lookup table because it is clearer and faster
// to simply enumerate the possibilities.
template <typename Int>
TMOL_DEVICE_FUNC tuple<Int, Int, Int> get_connection_spanning_subgraph_indices(
    Int index) {
  static const int CON_PATH_INDICES[][3] =
  {// Length
    {Int(subgraph_length), 0, 0},

    // Angles
    {Int(subgraph_angle), 0, 1},
    {Int(subgraph_angle), 0, 2},
    {Int(subgraph_angle), 0, 3},
    {Int(subgraph_angle), 1, 0},
    {Int(subgraph_angle), 2, 0},
    {Int(subgraph_angle), 3, 0},

    // Torsions
    {Int(subgraph_torsion), 0, 4},
    {Int(subgraph_torsion), 0, 5},
    {Int(subgraph_torsion), 0, 6},
    {Int(subgraph_torsion), 0, 7},
    {Int(subgraph_torsion), 0, 8},
    {Int(subgraph_torsion), 0, 9},
    {Int(subgraph_torsion), 0, 10},
    {Int(subgraph_torsion), 0, 11},
    {Int(subgraph_torsion), 0, 12},

    {Int(subgraph_torsion), 1, 1},
    {Int(subgraph_torsion), 1, 2},
    {Int(subgraph_torsion), 1, 3},
    {Int(subgraph_torsion), 2, 1},
    {Int(subgraph_torsion), 2, 2},
    {Int(subgraph_torsion), 2, 3},
    {Int(subgraph_torsion), 3, 1},
    {Int(subgraph_torsion), 3, 2},
    {Int(subgraph_torsion), 3, 3},

    {Int(subgraph_torsion), 4, 0},
    {Int(subgraph_torsion), 5, 0},
    {Int(subgraph_torsion), 6, 0},
    {Int(subgraph_torsion), 7, 0},
    {Int(subgraph_torsion), 8, 0},
    {Int(subgraph_torsion), 9, 0},
    {Int(subgraph_torsion), 10, 0},
    {Int(subgraph_torsion), 11, 0},
    {Int(subgraph_torsion), 12, 0}
  };
  return {CON_PATH_INDICES[index][0], CON_PATH_INDICES[index][1], CON_PATH_INDICES[index][2]};
}

TMOL_DEVICE_FUNC int get_n_connection_spanning_subgraphs(
    subgraph_type st)
{
  /// Return the number of connection-spanning (i.e. inter-residue) subgraphs
  /// for each type of subgraph for the enumeration above, which is based
  /// on the assumption that each atom has at most 4 connections.
  switch (st) {
    case subgraph_length: {return 1;}
    case subgraph_angle: {return 6;}
    case subgraph_torsion: {return 27;}
  };
}

TMOL_DEVICE_FUNC int get_connection_spanning_subgraphs_offset(
    subgraph_type st)
{
  switch (st) {
    case subgraph_length: {return 0;}
    case subgraph_angle: {return 1;}
    case subgraph_torsion: {return 6;}
  };
}

}  // namespace common
}  // namespace score
}  // namespace tmol
