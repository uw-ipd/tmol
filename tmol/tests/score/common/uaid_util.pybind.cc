#pragma once

#include <tmol/utility/tensor/pybind.h>
#include <tmol/score/common/uaid_util.hh>
#include <tmol/score/unresolved_atom.pybind.hh>

namespace tmol {
namespace test {
namespace score {
namespace common {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <typename Int, tmol::Device D>
at::Tensor resolve_uaids(
    TView<UnresolvedAtomID<Int>, 1, D> uaids,
    TView<Int, 1, D> block_inds,
    TView<Int, 1, D> pose_inds,
    TView<Int, 2, D> pose_stack_block_coord_offset,
    TView<Int, 2, D> pose_stack_block_type,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Int, 3, D> block_type_atom_downstream_of_conn) {
  int const n_uaids = uaids.size(0);
  int const n_poses = pose_stack_inter_block_connections.size(0);
  int const max_n_blocks = pose_stack_block_coord_offset.size(1);
  int const max_n_conn = pose_stack_inter_block_connections.size(2);
  int const n_block_types = block_type_atom_downstream_of_conn.size(0);

  assert(block_inds.size(0) == n_uaids);
  assert(pose_inds.size(0) == n_uaids);
  assert(pose_stack_block_coord_offset.size(0) == n_poses);
  assert(pose_stack_block_type.size(0) == n_poses);
  assert(pose_stack_block_type.size(1) == max_n_blocks);
  assert(pose_stack_inter_block_connections.size(1) == max_n_blocks);
  assert(block_type_atom_downstream_of_conn.size(1) == max_n_conn);

  auto outputs_t = TPack<Int, 1, D>::zeros({n_uaids});
  auto outputs = outputs_t.view;
  for (int i = 0; i < n_uaids; ++i) {
    outputs[i] = tmol::score::common::resolve_atom_from_uaid(
        uaids[i],
        block_inds[i],
        pose_inds[i],
        pose_stack_block_coord_offset,
        pose_stack_block_type,
        pose_stack_inter_block_connections,
        block_type_atom_downstream_of_conn);
  }

  return outputs_t.tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;

  m.def(
      "resolve_uaids",
      &resolve_uaids<int32_t, tmol::Device::CPU>,
      "uaids"_a,
      "block_inds"_a,
      "pose_inds"_a,
      "pose_stack_block_coord_offset"_a,
      "pose_stack_block_type"_a,
      "pose_stack_inter_block_connections"_a,
      "block_type_atom_downstream_of_conn"_a);
}

}  // namespace common
}  // namespace score
}  // namespace test
}  // namespace tmol
