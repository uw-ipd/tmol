#include <tmol/utility/portable/matrix.hh>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <Eigen/Core>

using tmol::portable::Vec;

tmol::TPack<float, 1, tmol::Device::CUDA> gpu_warp_segreduce_full(
    tmol::TView<float, 1, tmol::Device::CUDA> values,
    tmol::TView<int, 1, tmol::Device::CUDA> flags);

tmol::TPack<Vec<float, 3>, 1, tmol::Device::CUDA> gpu_warp_segreduce_full_vec3(
    tmol::TView<Vec<float, 3>, 1, tmol::Device::CUDA> values,
    tmol::TView<int, 1, tmol::Device::CUDA> flags);

tmol::TPack<Vec<float, 3>, 1, tmol::Device::CUDA>
gpu_warp_segreduce_vec3_benchmark(
    tmol::TView<Vec<float, 3>, 1, tmol::Device::CUDA> values,
    tmol::TView<int, 1, tmol::Device::CUDA> flags,
    int n_repeats);

tmol::TPack<float, 1, tmol::Device::CUDA> gpu_warp_segreduce_partial(
    tmol::TView<float, 1, tmol::Device::CUDA> values,
    tmol::TView<int, 1, tmol::Device::CUDA> flags);
