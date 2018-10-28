#include <ATen/ATen.h>
#include <ATen/ScalarTypeUtils.h>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace tmol {
namespace score {
namespace blocked {

static const int WARPS_PER_BLOCK = 4;

__device__ inline auto grid_x() -> decltype(threadIdx.x) {
  return (blockIdx.x * blockDim.x) + threadIdx.x;
}

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
    tmol::TView<Eigen::Matrix<Real, 3, 1>, 2, RestrictPtrTraits> coords,
    tmol::TView<Int, 2, RestrictPtrTraits> block_table,
    Real max_dis) {
  typedef Eigen::AlignedBox<Real, 3> Box;
  typedef Eigen::Matrix<Real, 3, 1> Vector;

  namespace cg = cooperative_groups;

  cg::thread_block tb = cg::this_thread_block();

  cg::thread_block_tile<32> tile = cg::tiled_partition<32>(tb);

  unsigned int tile_idx =
      (tb.group_index().x * (tb.size() / 32)) + (tb.thread_rank() / 32);
  auto bi = tile_idx % block_table.size(0);
  auto bj = tile_idx / block_table.size(0);

  bool block_interacts = block_interaction_check<8>(coords, max_dis, tile, bi, bj);

  if (tile.thread_rank() == 0) {
    block_table[bi][bj] = block_interacts;
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

  static_assert(sizeof(Box) == sizeof(Real) * 6, "");

  dim3 threads(32 * WARPS_PER_BLOCK);
  dim3 blocks((num_blocks * num_blocks) / WARPS_PER_BLOCK);

  at::Tensor block_table_t = at::empty(
      {num_blocks, num_blocks},
      at::TensorOptions(coords_t).dtype(at::CTypeToScalarType<Int>::to()));

  compute_block_table<Real, Int, BLOCK_SIZE><<<blocks, threads>>>(
      tmol::view_tensor<Vector, 2, RestrictPtrTraits>(coords_t),
      tmol::view_tensor<Int, 2, RestrictPtrTraits>(block_table_t),
      max_dis);

  return block_table_t;
};

template at::Tensor block_interaction_table<float, int32_t, 8>(
    at::Tensor coords_t, float max_dis);

}  // namespace blocked
}  // namespace score
}  // namespace tmol
