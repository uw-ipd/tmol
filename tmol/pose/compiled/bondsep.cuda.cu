#include "bondsep.hh"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace tmol::pose {

__global__ void gather_block_bondsep_kernel(
    int64_t size,
    int64_t blocks,
    int64_t ports,
    int64_t nodes,
    int32_t const* distances,
    int64_t const* offsets,
    int32_t const* counts,
    int32_t* output,
    int32_t sentinel) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < size;
       i += int64_t(blockDim.x) * gridDim.x)
    output[i] = block_bondsep_value(
        i, blocks, ports, nodes, distances, offsets, counts, sentinel);
}

template <>
void gather_block_bondsep<Device::CUDA>(
    torch::Tensor distances,
    torch::Tensor offsets,
    torch::Tensor counts,
    torch::Tensor output,
    int32_t sentinel) {
  if (!output.numel()) return;
  c10::cuda::CUDAGuard guard(distances.device());
  auto const grid = std::min<int64_t>((output.numel() + 255) / 256, 65535);
  gather_block_bondsep_kernel<<<
      grid,
      256,
      0,
      c10::cuda::getCurrentCUDAStream()>>>(
      output.numel(),
      counts.size(1),
      output.size(3),
      distances.size(1),
      distances.data_ptr<int32_t>(),
      offsets.data_ptr<int64_t>(),
      counts.data_ptr<int32_t>(),
      output.data_ptr<int32_t>(),
      sentinel);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace tmol::pose
