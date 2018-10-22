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

  auto coords = tmol::reinterpret_tensor<Vector, Real, 2>(coords_t);

  AT_ASSERTM(
      coords_t.size(0) % block_size == 0,
      "Coordinate size must be even multiple of target block size.");
  int64_t num_blocks = coords_t.size(0) / block_size;

  auto out_t = at::zeros({num_blocks, num_blocks}, coords_t.type());
  auto out = out_t.accessor<Real, 2>();

  static_assert(sizeof(Box) == sizeof(Real) * 6, "");
  auto box_t = at::zeros({num_blocks, 6}, coords_t.type());
  auto boxes = tmol::reinterpret_tensor<Box, Real, 2>(box_t);

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

template <typename Real, typename AtomType, typename Int>
at::Tensor lj_intra_block(
    at::Tensor coords_t,
    at::Tensor out_t,
    at::Tensor block_table_t,
    at::Tensor types_t,
    LJ_PARAM_ARGS) {
  typedef Eigen::AlignedBox<Real, 3> Box;
  typedef Eigen::Matrix<Real, 3, 1> Vector;

  auto coords = tmol::reinterpret_tensor<Vector, Real, 2>(coords_t);
  auto block_table = tmol::reinterpret_tensor<Real, Real, 2>(block_table_t);
  auto types = types_t.accessor<AtomType, 1>();

  auto out = out_t.accessor<Real, 2>();

  LJ_PARAM_UNPACK

  AT_ASSERTM(
      block_table_t.size(0) == block_table_t.size(1),
      "block_table must be square.");

  AT_ASSERTM(
      coords_t.size(0) % block_table_t.size(0) == 0,
      "coords size must be even multiple block_table size.");

  int64_t num_blocks = block_table_t.size(0);
  Int block_size = coords_t.size(0) / block_table_t.size(0);

  AT_ASSERTM(
      out_t.size(0) == coords_t.size(0) && out_t.size(1) == coords_t.size(0),
      "Output buffer must be of correct size.");

  AT_ASSERTM(
      block_table_t.size(0) == num_blocks
          && block_table_t.size(1) == num_blocks,
      "Block interation table did not match expected shape.");

  for (int bi = 0; bi < num_blocks; ++bi) {
    for (int bj = bi; bj < num_blocks; ++bj) {
      int bsi = bi * block_size;
      int bsj = bj * block_size;

      if (!block_table[bi][bj] > 0) {
        continue;
      }

      for (int i = bsi; i < bsi + block_size; ++i) {
        auto at = types[i];
        if (at == -1) {
          continue;
        }

        auto a = coords[i][0];

        for (int j = bsj; j < bsj + block_size; ++j) {
          auto bt = types[j];
          if (bt == -1 || j < i) {
            continue;
          }

          auto b = coords[j][0];

          Vector delta = a - b;
          auto dist = std::sqrt(delta.dot(delta));

          out[i][j] =
              lj(dist,
                 bonded_path_length[i][j],
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

template <typename Real, typename AtomType, typename Int>
at::Tensor lj_intra_naive(
    at::Tensor coords_t, at::Tensor types_t, LJ_PARAM_ARGS) {
  typedef Eigen::AlignedBox<Real, 3> Box;
  typedef Eigen::Matrix<Real, 3, 1> Vector;

  auto out_t = at::zeros({coords_t.size(0), coords_t.size(0)}, coords_t.type());

  auto coords = tmol::reinterpret_tensor<Vector, Real, 2>(coords_t);
  auto types = types_t.accessor<AtomType, 1>();

  auto out = out_t.accessor<Real, 2>();

  LJ_PARAM_UNPACK
  for (int i = i; i < coords.size(0); ++i) {
    auto a = coords[i][0];
    auto at = types[i];

    if (at == -1) {
      continue;
    }

    for (int j = i; j < coords.size(0); ++j) {
      auto b = coords[j][0];
      auto bt = types[j];
      if (bt == -1) {
        continue;
      }

      Vector delta = a - b;
      auto dist = std::sqrt(delta.dot(delta));

      out[i][j] =
          lj(dist,
             bonded_path_length[i][j],
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
      "lj_intra_block",
      &lj_intra_block<float, int64_t, uint8_t>,
      "LJ intra-coordinate score.",
      "coords"_a,
      "out"_a,
      "block_table"_a,
      "types"_a,
      LJ_PARAM_PYARGS);

  m.def(
      "lj_intra_naive",
      &lj_intra_naive<float, int64_t, uint8_t>,
      "LJ intra-coordinate score.",
      "coords"_a,
      "types"_a,
      LJ_PARAM_PYARGS);
}

}  // namespace lj
}  // namespace score
}  // namespace tmol
