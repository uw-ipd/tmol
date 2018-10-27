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

__device__ inline auto grid_x() -> decltype(threadIdx.x) {
  return (blockIdx.x * blockDim.x) + threadIdx.x;
}

template <typename Real, typename Int, int BLOCK_SIZE>
__global__ void compute_block_table(
    tmol::TView<Eigen::Matrix<Real, 3, 1>, 2, RestrictPtrTraits> coords,
    tmol::TView<Int, 2, RestrictPtrTraits> block_table,
    Real max_dis) {
  typedef Eigen::AlignedBox<Real, 3> Box;
  typedef Eigen::Matrix<Real, 3, 1> Vector;

  namespace cg = cooperative_groups;

  auto const bi = grid_x() / block_table.size(0);
  auto const bj = grid_x() % block_table.size(0);

  Int count = 0;
  for (auto i = bi * BLOCK_SIZE; i < (bi + 1) * BLOCK_SIZE; ++i) {
    Box box(coords[i][0]);
    box.extend(box.max() + Vector(max_dis, max_dis, max_dis));
    box.extend(box.min() - Vector(max_dis, max_dis, max_dis));

    for (auto j = bj * BLOCK_SIZE; j < (bj + 1) * BLOCK_SIZE; ++j) {
      count += box.contains(coords[j][0]) ? 1 : 0;
    }
  }

  block_table[bi][bj] = count;
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

  dim3 threads(128);
  dim3 blocks((num_blocks * num_blocks) / 128 + 1);

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
