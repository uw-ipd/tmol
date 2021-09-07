#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>

tmol::TPack<float, 1, tmol::Device::CUDA> warp_stride_reduce_gpu(
    tmol::TView<float, 1, tmol::Device::CUDA> values, int stride);

tmol::TPack<float, 1, tmol::Device::CUDA> warp_stride_reduce_gpu2(
    tmol::TView<float, 1, tmol::Device::CUDA> values, int stride);
