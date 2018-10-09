#include <ATen/ATen.h>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include "lj.hh"

#include <cuda.h>
#include <cuda_runtime.h>

namespace tmol {
namespace score {
namespace lj {

void __global__ lj_intra_kernel(
    tmol::PackedTensorAccessor<Eigen::Vector3f, 2> coords,
    tmol::PackedTensorAccessor<int64_t, 1> types,
    tmol::PackedTensorAccessor<uint8_t, 2> bonded_path_length,
    tmol::PackedTensorAccessor<float, 2> out,
    tmol::PackedTensorAccessor<float, 2> lj_sigma,
    tmol::PackedTensorAccessor<float, 2> lj_switch_slope,
    tmol::PackedTensorAccessor<float, 2> lj_switch_intercept,
    tmol::PackedTensorAccessor<float, 2> lj_coeff_sigma12,
    tmol::PackedTensorAccessor<float, 2> lj_coeff_sigma6,
    tmol::PackedTensorAccessor<float, 2> lj_spline_y0,
    tmol::PackedTensorAccessor<float, 2> lj_spline_dy0,
    tmol::PackedTensorAccessor<float, 1> lj_switch_dis2sigma,
    tmol::PackedTensorAccessor<float, 1> spline_start,
    tmol::PackedTensorAccessor<float, 1> max_dis) {

  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (i >= coords.size(0) || j >= coords.size(0)) {
    return;
  }

  if (i > j) {
    return;
  }
  auto at = types[i];
  if (at == -1) {
    return;
  }

  auto bt = types[j];
  if (bt == -1) {
    return;
  }

  auto a = coords[i][0];
  auto b = coords[j][0];

  Eigen::Vector3f delta = a - b;
  auto dist = std::sqrt(delta.dot(delta));

  out[i][j] = lj(
      dist,
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
  return;
}

at::Tensor lj_intra_cuda(
    at::Tensor coords_t,
    at::Tensor types_t,
    at::Tensor bonded_path_length_t,
    at::Tensor lj_sigma_t,
    at::Tensor lj_switch_slope_t,
    at::Tensor lj_switch_intercept_t,
    at::Tensor lj_coeff_sigma12_t,
    at::Tensor lj_coeff_sigma6_t,
    at::Tensor lj_spline_y0_t,
    at::Tensor lj_spline_dy0_t,
    at::Tensor lj_switch_dis2sigma_t,
    at::Tensor spline_start_t,
    at::Tensor max_dis_t) {
  auto out_t = coords_t.type().zeros({coords_t.size(0), coords_t.size(0)});
  auto out = tmol::reinterpret_tensor<float, float, 2>(out_t);

  auto coords = tmol::reinterpret_tensor<Eigen::Vector3f, float, 2>(coords_t);
  auto types = tmol::reinterpret_tensor<int64_t, int64_t, 1>(types_t);
  auto bonded_path_length = tmol::reinterpret_tensor<uint8_t, uint8_t, 2>(bonded_path_length_t);

  auto lj_sigma = tmol::reinterpret_tensor<float, float, 2>(lj_sigma_t);
  auto lj_switch_slope = tmol::reinterpret_tensor<float, float, 2>(lj_switch_slope_t);
  auto lj_switch_intercept = tmol::reinterpret_tensor<float, float, 2>(lj_switch_intercept_t);
  auto lj_coeff_sigma12 = tmol::reinterpret_tensor<float, float, 2>(lj_coeff_sigma12_t);
  auto lj_coeff_sigma6 = tmol::reinterpret_tensor<float, float, 2>(lj_coeff_sigma6_t);
  auto lj_spline_y0 = tmol::reinterpret_tensor<float, float, 2>(lj_spline_y0_t);
  auto lj_spline_dy0 = tmol::reinterpret_tensor<float, float, 2>(lj_spline_dy0_t);


  // Reshape globals into 1d, accessor cast of 0-d tensors segfaults.
  auto lj_switch_dis2sigma = tmol::reinterpret_tensor<float, float, 1>(lj_switch_dis2sigma_t.reshape(1));
  auto spline_start = tmol::reinterpret_tensor<float, float, 1>(spline_start_t.reshape(1));
  auto max_dis = tmol::reinterpret_tensor<float, float, 1>(max_dis_t.reshape(1));

  dim3 threads(8, 8);
  dim3 blocks(coords.size(0) / threads.x, coords.size(0) / threads.y);

  lj_intra_kernel<<<blocks, threads>>>(
     coords,
     types,
     bonded_path_length,
     out,
     lj_sigma,
     lj_switch_slope,
     lj_switch_intercept,
     lj_coeff_sigma12,
     lj_coeff_sigma6,
     lj_spline_y0,
     lj_spline_dy0,
     lj_switch_dis2sigma,
     spline_start,
     max_dis);

  return out_t;
}

}
}
}
