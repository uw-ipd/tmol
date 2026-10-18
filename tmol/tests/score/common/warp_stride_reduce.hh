#include <tmol/utility/portable/matrix.hh>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>

using tmol::portable::Vec;

tmol::TPack<float, 1, tmol::Device::CUDA> gpu_warp_stride_reduce_full(
    tmol::TView<float, 1, tmol::Device::CUDA> values, int stride);

tmol::TPack<Vec<float, 3>, 1, tmol::Device::CUDA>
gpu_warp_stride_reduce_full_vec3(
    tmol::TView<Vec<float, 3>, 1, tmol::Device::CUDA> values, int stride);

tmol::TPack<float, 1, tmol::Device::CUDA> gpu_warp_stride_reduce_partial(
    tmol::TView<float, 1, tmol::Device::CUDA> values, int stride);
