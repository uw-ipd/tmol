#include <ATen/ATen.h>
#include <ATen/ScalarTypeUtils.h>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include "lj.hh"

#include <cuda.h>
#include <cuda_runtime.h>

namespace tmol {
namespace score {
namespace lj {


__device__ inline auto grid_x() -> decltype(threadIdx.x){
  return (blockIdx.x * blockDim.x) + threadIdx.x;
}

__device__ inline auto grid_y() -> decltype(threadIdx.y){
  return (blockIdx.y * blockDim.y) + threadIdx.y;
}

__device__ inline auto grid_z() -> decltype(threadIdx.z){
  return (blockIdx.y * blockDim.y) + threadIdx.y;
}

template <typename Real>
__global__ void compute_coord_boxes(
  tmol::TView<Eigen::Matrix<Real, 3, 1>, 2> coords,
  tmol::TView<Eigen::AlignedBox<Real, 3>, 2> boxes,
  unsigned int block_size
)
{
  typedef Eigen::AlignedBox<Real, 3> Box;

  auto num_blocks = boxes.size(0);
  auto bi = grid_x();
  if (bi >= num_blocks){
    return;
  }

  auto bsi = bi * block_size;
  Box block_box(coords[bsi][0]);
  for (auto i = bsi + 1; i < bsi + block_size; ++i) {
    block_box.extend(coords[i][0]);
  }

  boxes[bi][0] = block_box;

  return;
}

template <typename Real, typename Int>
void __global__ compute_box_lists(
  tmol::TView<Eigen::AlignedBox<Real, 3>, 2> boxes,
  tmol::TView<Int, 2> block_lists,
  tmol::TView<Int, 1> block_list_lengths,
  Real max_dis
)
{
  typedef Eigen::AlignedBox<Real, 3> Box;
  typedef Eigen::Matrix<Real, 3, 1> Vector;

  auto num_blocks = boxes.size(0);
  auto bi = grid_x();
  if (bi >= num_blocks){
    return;
  }

  Box block_box = boxes[bi][0];

  block_box.extend(block_box.max() + Vector(max_dis, max_dis, max_dis));
  block_box.extend(block_box.min() - Vector(max_dis, max_dis, max_dis));

  for (unsigned int bj = 0; bj < num_blocks; ++bj) {
    if (block_box.intersects(boxes[bj][0])) {
      block_lists[bi][block_list_lengths[bi]] = bj;
      block_list_lengths[bi]++;
    }
  }

  return;
}

template <typename Int>
void __global__ pack_box_lists(
  tmol::TView<Int, 2> block_lists,
  tmol::TView<Int, 1> block_spans,
  tmol::TView<Int, 2> result
)
{
  auto num_blocks = block_lists.size(0);
  auto bi = grid_x();
  if (bi >= num_blocks){
    return;
  }

  int span_start = block_spans[bi];
  int span_end = block_spans[bi+1];

  for (int i = span_start; i < span_end; ++i) {
    result[i][0] = bi;
    result[i][1] = block_lists[bi][i - span_start];
  }
}


template <typename Real, typename Int>
at::Tensor block_interaction_lists(
    at::Tensor coords_t, Real max_dis, Int block_size){
  typedef Eigen::AlignedBox<Real, 3> Box;
  typedef Eigen::Matrix<Real, 3, 1> Vector;

  AT_ASSERTM(
      coords_t.size(0) % block_size == 0,
      "Coordinate size must be even multiple of target block size.");
  int64_t num_blocks = coords_t.size(0) / block_size;

  static_assert(sizeof(Box) == sizeof(Real) * 6, "");
  at::Tensor boxes_t = at::empty({num_blocks, 6}, at::TensorOptions(coords_t));

  dim3 threads(128, 1);
  dim3 blocks((num_blocks / threads.x) + 1, 1);

  compute_coord_boxes<Real><<<blocks, threads>>>(
    tmol::view_tensor<Vector, 2>(coords_t),
    tmol::view_tensor<Box, 2>(boxes_t),
    block_size
  );

  at::Tensor block_lists_t = at::empty(
      {num_blocks, num_blocks},
      at::TensorOptions(coords_t).dtype(at::CTypeToScalarType<Int>::to()));

  at::Tensor block_list_lengths_t = at::zeros(
      {num_blocks + 1},
      at::TensorOptions(coords_t).dtype(at::CTypeToScalarType<Int>::to()));


  compute_box_lists<Real, Int><<<blocks, threads>>>(
    tmol::view_tensor<Box, 2>(boxes_t),
    tmol::view_tensor<Int, 2>(block_lists_t),
    tmol::view_tensor<Int, 1>(block_list_lengths_t.slice(0, 1)),
    max_dis
  );

  at::Tensor block_spans_t = block_list_lengths_t.cumsum(0);
  at::Tensor result_t = at::empty({block_spans_t[num_blocks].toCLong(), 2}, block_spans_t.type());

  pack_box_lists<Int><<<blocks, threads>>>(
    tmol::view_tensor<Int, 2>(block_lists_t),
    tmol::view_tensor<Int, 1>(block_spans_t),
    tmol::view_tensor<Int, 2>(result_t)
  );

  return result_t;
};

template at::Tensor block_interaction_lists<float, int64_t>(
    at::Tensor coords_t, float max_dis, int64_t block_size);


template <typename Real, typename Int, typename PathLength>
__global__ void lj_intra_kernel(
    tmol::TView<Eigen::Matrix<Real, 3, 1>, 2> coords,
    tmol::TView<Int, 2> block_interactions,
    Int block_size,
    tmol::TView<Int, 1> types,
    tmol::TView<Real, 3> out,
    LJ_PARAM_VIEW_ARGS
)
{
  typedef Eigen::Matrix<Real, 3, 1> Vector;

  auto num_lines = block_interactions.size(0) * block_size;
  auto li = grid_x();
  unsigned int bp = li / block_size;
  unsigned int i = li % block_size;

  if (li >= num_lines){
    return;
  }

  int bsi = block_interactions[bp][0] * block_size;
  int bsj = block_interactions[bp][1] * block_size;

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


template <typename Real, typename Int, typename PathLength>
at::Tensor lj_intra_block(
    at::Tensor coords_t,
    at::Tensor block_interactions_t,
    Int block_size,
    at::Tensor types_t,
    LJ_PARAM_ARGS) {
  typedef Eigen::Matrix<Real, 3, 1> Vector;

  int64_t num_lines = block_interactions_t.size(0) * block_size;

  dim3 threads(128, 1);
  dim3 blocks((num_lines / threads.x) + 1, 1);

  at::Tensor out_t = at::empty({block_interactions_t.size(0), block_size, block_size}, at::TensorOptions(coords_t));

  lj_intra_kernel<Real, Int, PathLength><<<blocks, threads>>>(
      tmol::view_tensor<Vector, 2>(coords_t),
      tmol::view_tensor<Int, 2>(block_interactions_t),
      block_size,
      tmol::view_tensor<Int, 1>(types_t),
      tmol::view_tensor<Real, 3>(out_t),
      LJ_PARAM_UNPACK_ARGS
  );

  return out_t;

};

template at::Tensor lj_intra_block<float, int64_t, uint8_t>(
    at::Tensor coords_t,
    at::Tensor block_interactions_t,
    int64_t block_size,
    at::Tensor types_t,
    LJ_PARAM_ARGS);
}
}
}

