#include <ATen/ATen.h>
#include <ATen/ScalarTypeUtils.h>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <Eigen/Geometry>

namespace tmol {
namespace score {
namespace blocked {

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

  AT_ASSERTM(
      coords_t.size(0) % BLOCK_SIZE == 0,
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

template at::Tensor calc_block_aabb<float>(at::Tensor coords_t);

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

template <typename Real, typename Int, int BLOCK_SIZE>
__global__ void compute_block_table(
    tmol::TView<Eigen::AlignedBox<Real, 3>, 2, RestrictPtrTraits> block_aabb,
    tmol::TView<Int, 2, RestrictPtrTraits> block_table,
    Real max_dis) {
  typedef Eigen::AlignedBox<Real, 3> Box;
  typedef Eigen::Matrix<Real, 3, 1> Vector;

  namespace cg = cooperative_groups;

  cg::thread_block tb = cg::this_thread_block();
  cg::thread_block_tile<32> tile = cg::tiled_partition<32>(tb);

  unsigned int warps_per_block = tb.size() / 32;
  unsigned int tile_idx =
      (tb.group_index().x * warps_per_block) + (tb.thread_rank() / 32);
  auto bi = tile_idx % block_table.size(0);

  Box block_box(
      block_aabb[bi][0].min() - Vector(max_dis, max_dis, max_dis),
      block_aabb[bi][0].max() + Vector(max_dis, max_dis, max_dis));

  bool block_interacts = false;
  for (auto bj = tile_idx / block_table.size(0); bj < block_table.size(1);
       bj += warps_per_block) {
    Box other_box = block_aabb[bj][0];
    bool block_interacts = block_box.intersects(other_box);
    if (tile.thread_rank() == 0) {
      block_table[bi][bj] = block_interacts;
    }
  }
}

template <typename Real, typename Int, int BLOCK_SIZE>
at::Tensor block_interaction_table(at::Tensor coords_t, Real max_dis) {
  typedef Eigen::AlignedBox<Real, 3> Box;
  typedef Eigen::Matrix<Real, 3, 1> Vector;

  AT_ASSERTM(
      coords_t.size(0) % BLOCK_SIZE == 0,
      "Coordinate size must be even multiple of target block size.");
  int64_t num_blocks = coords_t.size(0) / BLOCK_SIZE;

  static const int WARPS_PER_BLOCK = 8;

  dim3 threads(32 * WARPS_PER_BLOCK);
  dim3 blocks(num_blocks);

  at::Tensor aabb_t = at::empty(
      {num_blocks, 6},
      at::TensorOptions(coords_t).dtype(at::CTypeToScalarType<Real>::to()));


  at::Tensor block_table_t = at::empty(
      {num_blocks, num_blocks},
      at::TensorOptions(coords_t).dtype(at::CTypeToScalarType<Int>::to()));

  block_aabb_kernel<Real><<<blocks, threads>>>(
      tmol::view_tensor<Real, 2, RestrictPtrTraits>(coords_t),
      tmol::view_tensor<Real, 2, RestrictPtrTraits>(aabb_t));

  compute_block_table<Real, Int, BLOCK_SIZE><<<blocks, threads>>>(
      tmol::view_tensor<Box, 2, RestrictPtrTraits>(aabb_t),
      tmol::view_tensor<Int, 2, RestrictPtrTraits>(block_table_t),
      max_dis);

  return block_table_t;
};

template <typename Real, typename Int, int BLOCK_SIZE>
__global__ void compute_block_list(
    tmol::TView<Eigen::Matrix<Real, 3, 1>, 2, RestrictPtrTraits> coords,
    tmol::TView<Int, 2, RestrictPtrTraits> block_table,
    tmol::TView<Int, 1, RestrictPtrTraits> block_table_idx,
    Real max_dis) {
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

  bool block_interacts =
      block_interaction_check<8>(coords, max_dis, tile, bi, bj);

  if (tile.thread_rank() == 0 && block_interacts) {
    Int block_idx = atomicAdd(&block_table_idx[0], 1);
    block_table[block_idx][0] = bi;
    block_table[block_idx][1] = bj;
  }
}

template <typename Real, typename Int, int BLOCK_SIZE>
std::tuple<at::Tensor, at::Tensor> block_interaction_list(
    at::Tensor coords_t, Real max_dis) {
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

  at::Tensor block_table_t = at::empty(
      {num_blocks * num_blocks, 2},
      at::TensorOptions(coords_t).dtype(at::CTypeToScalarType<Int>::to()));

  at::Tensor block_table_idx_t = at::zeros(
      {1}, at::TensorOptions(coords_t).dtype(at::CTypeToScalarType<Int>::to()));

  compute_block_list<Real, Int, BLOCK_SIZE><<<blocks, threads>>>(
      tmol::view_tensor<Vector, 2, RestrictPtrTraits>(coords_t),
      tmol::view_tensor<Int, 2, RestrictPtrTraits>(block_table_t),
      tmol::view_tensor<Int, 1, RestrictPtrTraits>(block_table_idx_t),
      max_dis);

  return std::tuple<at::Tensor, at::Tensor>(block_table_t, block_table_idx_t);
};

template at::Tensor block_interaction_table<float, int32_t, 8>(
    at::Tensor coord_t, float max_dis);

template std::tuple<at::Tensor, at::Tensor>
block_interaction_list<float, int32_t, 8>(at::Tensor coords_t, float max_dis);

}  // namespace blocked
}  // namespace score
}  // namespace tmol
