#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>

tmol::TPack<float, 1, tmol::Device::CUDA> warp_segreduce_gpu(
    tmol::TView<float, 1, tmol::Device::CUDA> values,
    tmol::TView<int, 1, tmol::Device::CUDA> flags);

tmol::TPack<float, 1, tmol::Device::CUDA> warp_segreduce_gpu2(
    tmol::TView<float, 1, tmol::Device::CUDA> values,
    tmol::TView<int, 1, tmol::Device::CUDA> flags);
