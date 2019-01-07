#include <ATen/ATen.h>
#include <ATen/ScalarTypeUtils.h>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include "lj.hh"

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace tmol {
namespace score {
namespace lj {

__global__ int foocuda(int a) { return 2*a; }

template <typename Real>
__global__ void block_aabb_kernel(
    tmol::TView<Real, 2, RestrictPtrTraits> coords,
    tmol::TView<Real, 2, RestrictPtrTraits> box_out) {
  static const int BLOCK_SIZE = 8;

  namespace cg = cooperative_groups;

  cg::thread_block tb = cg::this_thread_block();
  cg::thread_block_tile<32> tile = cg::tiled_partition<32>(tb);

  unsigned int block_index =
      (tb.group_index().x * (tb.size() / 32)) + (tb.thread_rank() / 32);
  auto bsi = block_index * BLOCK_SIZE;

  if (bsi >= coords.size(0)) {
    return;
  }

  auto coor = tile.thread_rank() % BLOCK_SIZE;
  auto dim = tile.thread_rank() / BLOCK_SIZE;

  Real min;
  Real max;
  if (dim < 3) {
    max = coords[bsi + coor][dim];
    min = max;
  }

#define FULL_MASK 0xffffffff
  for (int offset = 4; offset > 0; offset /= 2) {
    Real other_min = __shfl_down_sync(FULL_MASK, min, offset);
    min = fminf(min, other_min);
    Real other_max = __shfl_down_sync(FULL_MASK, max, offset);
    max = fmaxf(max, other_max);
  }

  if (coor == 0 && dim < 3) {
    box_out[block_index][dim] = min;
    box_out[block_index][dim + 3] = max;
  }
}

template <typename Real>
at::Tensor calc_block_aabb(at::Tensor coords_t) {
  static const int BLOCK_SIZE = 8;

  AT_ASSERTM(coords_t.size(0) % BLOCK_SIZE == 0,
             "Coordinate size must be even multiple of target block size.");
  int64_t num_blocks = coords_t.size(0) / BLOCK_SIZE;

  static const int WARPS_PER_BLOCK = 8;

  dim3 threads(32 * WARPS_PER_BLOCK);
  dim3 blocks(((num_blocks) / WARPS_PER_BLOCK) + 1);

  at::Tensor aabb_t = at::empty(
      {num_blocks, 6},
      at::TensorOptions(coords_t).dtype(at::CTypeToScalarType<Real>::to()));

  block_aabb_kernel<Real><<<blocks, threads>>>(
      tmol::view_tensor<Real, 2, RestrictPtrTraits>(coords_t),
      tmol::view_tensor<Real, 2, RestrictPtrTraits>(aabb_t));

  return aabb_t;
};

template <int BLOCK_SIZE, typename Real, typename Int, typename PathLength>
__global__ void blocked_lj_kernel(
    tmol::TView<int32_t, 1, RestrictPtrTraits> out_block_count,
    tmol::TView<Int, 2, RestrictPtrTraits> out_block_inds,
    tmol::TView<Real, 3, RestrictPtrTraits> out_block_lj,
    tmol::TView<Eigen::AlignedBox<Real, 3>, 2, RestrictPtrTraits>
        coord_block_aabb,
    tmol::TView<Eigen::Matrix<Real, 3, 1>, 2, RestrictPtrTraits> coords,
    tmol::TView<Int, 1, RestrictPtrTraits> types, LJ_PARAM_VIEW_ARGS) {
  typedef Eigen::AlignedBox<Real, 3> Box;
  typedef Eigen::Matrix<Real, 3, 1> Vector;

  namespace cg = cooperative_groups;

  cg::thread_block tb = cg::this_thread_block();
  cg::thread_block_tile<32> tile = cg::tiled_partition<32>(tb);

  unsigned int warps_per_block = tb.size() / 32;
  unsigned int tile_idx =
      (tb.group_index().x * warps_per_block) + (tb.thread_rank() / 32);
  auto bi = tile_idx % coord_block_aabb.size(0);

  Box block_box(coord_block_aabb[bi][0].min() -
                    Vector(max_dis[0], max_dis[0], max_dis[0]),
                coord_block_aabb[bi][0].max() +
                    Vector(max_dis[0], max_dis[0], max_dis[0]));

  for (auto bj = tile_idx / coord_block_aabb.size(0);
       bj < coord_block_aabb.size(0); bj += warps_per_block) {
    Box other_box = coord_block_aabb[bj][0];
    bool block_interacts = block_box.intersects(other_box);
    if (!block_interacts) {
      continue;
    }

    int32_t out_block;
    if (tile.thread_rank() == 0) {
      out_block = atomicAdd(&out_block_count[0], 1);
      out_block_inds[out_block][0] = bi;
      out_block_inds[out_block][1] = bj;
    }
    out_block = tile.shfl(out_block, 0);

    auto tj = (tile.thread_rank() % BLOCK_SIZE);
    auto j = (bj * BLOCK_SIZE) + tj;
    Vector b = coords[j][0];
    auto bt = types[j];

    for (auto ti = (tile.thread_rank() / BLOCK_SIZE); ti < BLOCK_SIZE;
         ti += 4) {
      auto i = (bi * BLOCK_SIZE) + ti;
      Vector a = coords[i][0];
      auto at = types[i];

      if (at != -1 && bt != -1 && i < j) {
        Vector delta = a - b;
        auto dist = std::sqrt(delta.dot(delta));

        out_block_lj[out_block][ti][tj] =
            lj(dist, bonded_path_length[i][j], lj_sigma[at][bt],
               lj_switch_slope[at][bt], lj_switch_intercept[at][bt],
               lj_coeff_sigma12[at][bt], lj_coeff_sigma6[at][bt],
               lj_spline_y0[at][bt], lj_spline_dy0[at][bt],
               lj_switch_dis2sigma[0], spline_start[0], max_dis[0]);
      } else {
        out_block_lj[out_block][ti][tj] = 0;
      }
    }
  }

}  // namespace lj

template <int BLOCK_SIZE, typename Real, typename Int, typename PathLength>
std::tuple<at::Tensor, at::Tensor, at::Tensor> lj_intra_block(
    at::Tensor coords_t, at::Tensor types_t, LJ_PARAM_ARGS) {
  typedef Eigen::AlignedBox<Real, 3> Box;
  typedef Eigen::Matrix<Real, 3, 1> Vector;

  AT_ASSERTM(coords_t.size(0) % BLOCK_SIZE == 0,
             "Coordinate size must be even multiple of target block size.");
  int64_t num_blocks = coords_t.size(0) / BLOCK_SIZE;

  static const int WARPS_PER_BLOCK = 8;
  dim3 threads(32 * WARPS_PER_BLOCK);
  dim3 blocks(num_blocks);

  at::Tensor aabb_t = at::empty(
      {num_blocks, 6},
      at::TensorOptions(coords_t).dtype(at::CTypeToScalarType<Real>::to()));

  block_aabb_kernel<Real><<<blocks, threads>>>(
      tmol::view_tensor<Real, 2, RestrictPtrTraits>(coords_t),
      tmol::view_tensor<Real, 2, RestrictPtrTraits>(aabb_t));

  at::Tensor block_table_idx_t = at::zeros(
      {1},
      at::TensorOptions(coords_t).dtype(at::CTypeToScalarType<int32_t>::to()));

  at::Tensor block_index_t = at::empty(
      {num_blocks * num_blocks, 2},
      at::TensorOptions(coords_t).dtype(at::CTypeToScalarType<Int>::to()));

  at::Tensor block_result_t = at::empty(
      {num_blocks * num_blocks, BLOCK_SIZE, BLOCK_SIZE},
      at::TensorOptions(coords_t).dtype(at::CTypeToScalarType<Real>::to()));

  blocked_lj_kernel<BLOCK_SIZE, Real, Int, PathLength><<<blocks, threads>>>(
      tmol::view_tensor<int32_t, 1, RestrictPtrTraits>(block_table_idx_t),
      tmol::view_tensor<Int, 2, RestrictPtrTraits>(block_index_t),
      tmol::view_tensor<Real, 3, RestrictPtrTraits>(block_result_t),
      tmol::view_tensor<Box, 2, RestrictPtrTraits>(aabb_t),
      tmol::view_tensor<Vector, 2, RestrictPtrTraits>(coords_t),
      tmol::view_tensor<Int, 1, RestrictPtrTraits>(types_t),
      LJ_PARAM_UNPACK_ARGS);

  return std::tuple<at::Tensor, at::Tensor, at::Tensor>(
      block_table_idx_t, block_index_t, block_result_t);
};

template std::tuple<at::Tensor, at::Tensor, at::Tensor>
lj_intra_block<8, float, int64_t, uint8_t>(at::Tensor coords_t,
                                           at::Tensor types_t, LJ_PARAM_ARGS);

}  // namespace lj
}  // namespace score
}  // namespace tmol
