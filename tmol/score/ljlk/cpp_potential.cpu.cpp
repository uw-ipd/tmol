#include <ATen/ScalarTypeUtils.h>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <torch/torch.h>
#include "Eigen/Geometry"
#include "lj.hh"

namespace tmol {
namespace score {
namespace lj {

template <typename Real, typename Int>
at::Tensor block_interaction_table(
    at::Tensor coords_t, Real max_dis, Int block_size) {
  typedef Eigen::AlignedBox<Real, 3> Box;
  typedef Eigen::Matrix<Real, 3, 1> Vector;

  auto coords = tmol::view_tensor<Vector, 2>(coords_t);

  AT_ASSERTM(
      coords_t.size(0) % block_size == 0,
      "Coordinate size must be even multiple of target block size.");
  int64_t num_blocks = coords_t.size(0) / block_size;

  auto out_t = at::empty({num_blocks, num_blocks}, coords_t.type());
  auto out = tmol::view_tensor<Real, 2>(out_t);

  static_assert(sizeof(Box) == sizeof(Real) * 6, "");
  auto box_t = at::zeros({num_blocks, 6}, coords_t.type());
  auto boxes = tmol::view_tensor<Box, 2>(box_t);

  for (int bi = 0; bi < num_blocks; ++bi) {
    int bsi = bi * block_size;
    Box block_box;
    for (int i = bsi; i < bsi + block_size; ++i) {
      block_box.extend(coords[i][0]);
    }

    boxes[bi][0] = block_box;
  }

  for (int bi = 0; bi < num_blocks; ++bi) {
    Box block_box = boxes[bi][0];
    block_box.extend(block_box.max() + Vector(max_dis, max_dis, max_dis));
    block_box.extend(block_box.min() - Vector(max_dis, max_dis, max_dis));

    for (int bj = bi; bj < num_blocks; ++bj) {
      out[bi][bj] = block_box.intersects(boxes[bj][0]);
    }
  }

  return out_t;
}

template <typename Real, typename Int>
at::Tensor block_interaction_lists(
    at::Tensor coords_t, Real max_dis, Int block_size) {
  typedef Eigen::AlignedBox<Real, 3> Box;
  typedef Eigen::Matrix<Real, 3, 1> Vector;

  auto coords = tmol::view_tensor<Vector, 2>(coords_t);

  AT_ASSERTM(
      coords_t.size(0) % block_size == 0,
      "Coordinate size must be even multiple of target block size.");
  int64_t num_blocks = coords_t.size(0) / block_size;

  at::Tensor block_lists_t = at::empty(
      {num_blocks, num_blocks},
      at::TensorOptions(coords_t).dtype(at::CTypeToScalarType<Int>::to()));
  auto block_lists = tmol::view_tensor<Int, 2>(block_lists_t);

  at::Tensor block_list_lengths_t = at::zeros(
      {num_blocks + 1},
      at::TensorOptions(coords_t).dtype(at::CTypeToScalarType<Int>::to()));
  auto block_list_lengths =
      tmol::view_tensor<Int, 1>(block_list_lengths_t.slice(0, 1));

  static_assert(sizeof(Box) == sizeof(Real) * 6, "");
  auto box_t = at::zeros({num_blocks, 6}, coords_t.type());
  auto boxes = tmol::view_tensor<Box, 2>(box_t);

  for (int bi = 0; bi < num_blocks; ++bi) {
    int bsi = bi * block_size;
    Box block_box;
    for (int i = bsi; i < bsi + block_size; ++i) {
      block_box.extend(coords[i][0]);
    }

    boxes[bi][0] = block_box;
  }

  for (int bi = 0; bi < num_blocks; ++bi) {
    Box block_box = boxes[bi][0];
    block_box.extend(block_box.max() + Vector(max_dis, max_dis, max_dis));
    block_box.extend(block_box.min() - Vector(max_dis, max_dis, max_dis));

    for (int bj = 0; bj < num_blocks; ++bj) {
      if (block_box.intersects(boxes[bj][0])) {
        block_lists[bi][block_list_lengths[bi]] = bj;
        block_list_lengths[bi]++;
      }
    }
  }

  at::Tensor block_spans_t = block_list_lengths_t.cumsum(0);
  auto block_spans = tmol::view_tensor<Int, 1>(block_spans_t);

  at::Tensor result_t =
      at::empty({block_spans[num_blocks], 2}, block_spans_t.type());
  auto result = tmol::view_tensor<Int, 2>(result_t);
  for (int bi = 0; bi < num_blocks; ++bi) {
    for (int r = 0; r < block_list_lengths[bi]; ++r) {
      result[block_spans[bi] + r][0] = bi;
      result[block_spans[bi] + r][1] = block_lists[bi][r];
    }
  }

  return result_t;
}

template <typename Real, typename Int, typename PathLength>
at::Tensor lj_intra_block(
    at::Tensor coords_t,
    at::Tensor block_interactions_t,
    Int block_size,
    at::Tensor types_t,
    LJ_PARAM_ARGS) {
  typedef Eigen::AlignedBox<Real, 3> Box;
  typedef Eigen::Matrix<Real, 3, 1> Vector;

  auto coords = tmol::view_tensor<Vector, 2>(coords_t);
  auto block_interactions = tmol::view_tensor<Int, 2>(block_interactions_t);
  auto types = tmol::view_tensor<Int, 1>(types_t);

  LJ_PARAM_UNPACK

  int64_t num_blocks = block_interactions_t.size(0);

  at::Tensor out_t =
      at::empty({num_blocks, block_size, block_size}, coords_t.type());
  auto out = tmol::view_tensor<Real, 3>(out_t);

  for (int bp = 0; bp < num_blocks; ++bp) {
    int bi = block_interactions[bp][0];
    int bj = block_interactions[bp][1];
    int bsi = bi * block_size;
    int bsj = bj * block_size;

    for (int i = 0; i < block_size; ++i) {
      auto aidx = i + bsi;
      auto a = coords[aidx][0];
      auto at = types[aidx];

      for (int j = 0; j < block_size; ++j) {
        auto bidx = j + bsj;
        auto b = coords[bidx][0];
        auto bt = types[bidx];

        if (at == -1 || bt == -1 || bidx < aidx) {
          out[bp][i][j] = 0;
        } else {
          Vector delta = a - b;
          auto dist = std::sqrt(delta.dot(delta));

          out[bp][i][j] =
              lj(dist,
                 bonded_path_length[aidx][bidx],
                 lj_sigma[at][bt],
                 lj_switch_slope[at][bt],
                 lj_switch_intercept[at][bt],
                 lj_coeff_sigma12[at][bt],
                 lj_coeff_sigma6[at][bt],
                 lj_spline_y0[at][bt],
                 lj_spline_dy0[at][bt],
                 lj_switch_dis2sigma[0],
                 spline_start[0],
                 max_dis[0]);
        }
      }
    }
  }

  return out_t;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;
  m.def("lj", &lj<float, uint8_t>, "LJ potential.", "dist"_a, LJ_PARAM_PYARGS);

  m.def(
      "d_lj_d_dist",
      &d_lj_d_dist<float, uint8_t>,
      "LJ potential derivative.",
      "dist"_a,
      LJ_PARAM_PYARGS);

  m.def(
      "block_interaction_table",
      &block_interaction_table<float, int64_t>,
      "Calculate coordinate-block interaction table.",
      "coords"_a,
      "max_dis"_a,
      "block_size"_a);

  m.def(
      "block_interaction_lists",
      &block_interaction_lists<float, int64_t>,
      "Calculate coordinate-block interaction lists.",
      "coords"_a,
      "max_dis"_a,
      "block_size"_a);

  m.def(
      "lj_intra_block",
      &lj_intra_block<float, int64_t, uint8_t>,
      "LJ intra-coordinate score.",
      "coords"_a,
      "block_iteractions"_a,
      "block_size"_a,
      "types"_a,
      LJ_PARAM_PYARGS);
}

}  // namespace lj
}  // namespace score
}  // namespace tmol
