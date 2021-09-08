#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <Eigen/Core>

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

tmol::TPack<float, 1, tmol::Device::CUDA> gpu_warp_segreduce_full(
    tmol::TView<float, 1, tmol::Device::CUDA> values,
    tmol::TView<int, 1, tmol::Device::CUDA> flags);

tmol::TPack<Vec<float, 3>, 1, tmol::Device::CUDA> gpu_warp_segreduce_full_vec3(
    tmol::TView<Vec<float, 3>, 1, tmol::Device::CUDA> values,
    tmol::TView<int, 1, tmol::Device::CUDA> flags);

tmol::TPack<float, 1, tmol::Device::CUDA> gpu_warp_segreduce_partial(
    tmol::TView<float, 1, tmol::Device::CUDA> values,
    tmol::TView<int, 1, tmol::Device::CUDA> flags);
