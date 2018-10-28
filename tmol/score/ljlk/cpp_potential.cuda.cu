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

template <int BLOCK_SIZE, typename Real, typename Int>
__device__ inline bool block_interaction_check(
    tmol::TView<Eigen::Matrix<Real, 3, 1>, 2, RestrictPtrTraits> coords,
    Real max_dis,
    cooperative_groups::thread_block_tile<32> tile,
    Int bi,
    Int bj) {
  namespace cg = cooperative_groups;
  auto i = (bi * BLOCK_SIZE) + (tile.thread_rank() % BLOCK_SIZE);
  auto j = (bj * BLOCK_SIZE) + (tile.thread_rank() / BLOCK_SIZE);

  typedef Eigen::AlignedBox<Real, 3> Box;
  typedef Eigen::Matrix<Real, 3, 1> Vector;

  Box box(coords[i][0]);
  box.extend(box.max() + Vector(max_dis, max_dis, max_dis));
  box.extend(box.min() - Vector(max_dis, max_dis, max_dis));

  if (tile.any(box.contains(coords[j][0]))) {
    return true;
  } else if (tile.any(box.contains(coords[j + (BLOCK_SIZE / 2)][0]))) {
    return true;
  } else {
    return false;
  }
}

template <int BLOCK_SIZE, typename Real, typename Int, typename PathLength>
__global__ void blocked_lj_kernel(
    tmol::TView<Eigen::Matrix<Real, 3, 1>, 2, RestrictPtrTraits> coords,
    tmol::TView<Int, 1, RestrictPtrTraits> types,
    tmol::TView<int32_t, 1, RestrictPtrTraits> block_out_idx,
    tmol::TView<Int, 2, RestrictPtrTraits> block_ind_out,
    tmol::TView<Real, 3, RestrictPtrTraits> block_lj_out,
    LJ_PARAM_VIEW_ARGS) {
  typedef Eigen::AlignedBox<Real, 3> Box;
  typedef Eigen::Matrix<Real, 3, 1> Vector;

  namespace cg = cooperative_groups;

  cg::thread_block tb = cg::this_thread_block();

  cg::thread_block_tile<32> tile = cg::tiled_partition<32>(tb);
  unsigned int num_blocks = coords.size(0) / BLOCK_SIZE;

  unsigned int tile_idx =
      (tb.group_index().x * (tb.size() / 32)) + (tb.thread_rank() / 32);

  auto bi = tile_idx % num_blocks;
  auto bj = tile_idx / num_blocks;

  if(!(bi <= bj)){
    return;
  }

  if (!block_interaction_check<8>(coords, max_dis[0], tile, bi, bj)) {
    return;
  }

  int32_t block_idx;
  if (tile.thread_rank() == 0) {
    block_idx = atomicAdd(&block_out_idx[0], 1);
    block_ind_out[block_idx][0] = bi;
    block_ind_out[block_idx][1] = bj;
  }

  block_idx = tile.shfl(block_idx, 0);

  auto ti = (tile.thread_rank() / BLOCK_SIZE);
  auto i = (bi * BLOCK_SIZE) + ti;
  Vector a = coords[i][0];
  auto at = types[i];

  auto tj = (tile.thread_rank() % BLOCK_SIZE);
  auto j = (bj * BLOCK_SIZE) + tj;
  Vector b = coords[j][0];
  auto bt = types[j];

  if (at != -1 && bt != -1 && i < j) {
    Vector delta = a - b;
    auto dist = std::sqrt(delta.dot(delta));

    block_lj_out[block_idx][ti][tj] =
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
  } else {
    block_lj_out[block_idx][ti][tj] = 0;
  }

  ti += 4;
  i = (bi * BLOCK_SIZE) + ti;
  a = coords[i][0];
  at = types[i];

  if (at != -1 && bt != -1 && i < j) {
    Vector delta = a - b;
    auto dist = std::sqrt(delta.dot(delta));

    block_lj_out[block_idx][ti][tj] =
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
  } else {
    block_lj_out[block_idx][ti][tj] = 0;
  }
}  // namespace lj

template <int BLOCK_SIZE, typename Real, typename Int, typename PathLength>
std::tuple<at::Tensor, at::Tensor, at::Tensor> lj_intra_block(
    at::Tensor coords_t, at::Tensor types_t, LJ_PARAM_ARGS) {
  typedef Eigen::AlignedBox<Real, 3> Box;
  typedef Eigen::Matrix<Real, 3, 1> Vector;

  AT_ASSERTM(
      coords_t.size(0) % BLOCK_SIZE == 0,
      "Coordinate size must be even multiple of target block size.");
  int64_t num_blocks = coords_t.size(0) / BLOCK_SIZE;

  static_assert(sizeof(Box) == sizeof(Real) * 6, "");

  static const int WARPS_PER_BLOCK = 4;
  dim3 threads(32 * WARPS_PER_BLOCK);
  dim3 blocks((num_blocks * num_blocks) / WARPS_PER_BLOCK);

  std::cerr << "threads: " << threads.x << std::endl;
  std::cerr << "blocks: " << blocks.x << std::endl;

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
      tmol::view_tensor<Vector, 2, RestrictPtrTraits>(coords_t),
      tmol::view_tensor<Int, 1, RestrictPtrTraits>(types_t),
      tmol::view_tensor<int32_t, 1, RestrictPtrTraits>(block_table_idx_t),
      tmol::view_tensor<Int, 2, RestrictPtrTraits>(block_index_t),
      tmol::view_tensor<Real, 3, RestrictPtrTraits>(block_result_t),
      LJ_PARAM_UNPACK_ARGS);

  return std::tuple<at::Tensor, at::Tensor, at::Tensor>(
      block_table_idx_t, block_index_t, block_result_t);
};

template std::tuple<at::Tensor, at::Tensor, at::Tensor>
lj_intra_block<8, float, int64_t, uint8_t>(
    at::Tensor coords_t, at::Tensor types_t, LJ_PARAM_ARGS);

}  // namespace lj
}  // namespace score
}  // namespace tmol
