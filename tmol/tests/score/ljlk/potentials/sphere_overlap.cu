#include <pybind11/pybind11.h>
#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/tensor/pybind.h>

#include <tmol/score/ljlk/potentials/sphere_overlap.cuda.cuh>

#include <moderngpu/launch_box.hxx>

// This file moves in more recent versions of Torch
#include <c10/cuda/CUDAStream.h>

namespace tmol {
namespace tests {
namespace score {
namespace ljlk {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <typename Real>

struct compute_block_spheres {
  static TPack<Real, 3, tmol::Device::CUDA> f(
      TView<Vec<Real, 3>, 2, tmol::Device::CUDA> coords,
      TView<int32_t, 2, tmol::Device::CUDA> pose_stack_block_coord_offset,
      TView<int32_t, 2, tmol::Device::CUDA> pose_stack_block_type,
      TView<int32_t, 1, tmol::Device::CUDA> block_type_n_atoms) {
    typedef mgpu::launch_box_t<
        mgpu::arch_20_cta<32, 1>,
        mgpu::arch_35_cta<32, 1>,
        mgpu::arch_52_cta<32, 1>>
        // mgpu::arch_70_cta<32, 1>,
        // mgpu::arch_75_cta<32, 1>>
        launch_t;

    int const n_poses = coords.size(0);
    int const max_n_blocks = pose_stack_block_type.size(1);

    auto block_spheres_t =
        TPack<Real, 3, tmol::Device::CUDA>::zeros({n_poses, max_n_blocks, 4});
    auto block_spheres = block_spheres_t.view;

    at::cuda::CUDAStream wrapped_stream = at::cuda::getDefaultCUDAStream();
    mgpu::standard_context_t context(wrapped_stream.stream());

    tmol::score::ljlk::potentials::launch_compute_block_spheres<
        tmol::Device::CUDA,
        Real,
        int32_t,
        launch_t>(

        coords,
        pose_stack_block_coord_offset,
        pose_stack_block_type,
        block_type_n_atoms,
        block_spheres,
        context);
    return block_spheres_t;
  }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;
  using namespace tmol::score::ljlk::potentials;

  m.def(
      "compute_block_spheres_float",
      &compute_block_spheres<float>::f,
      "coords"_a,
      "pose_stack_block_coord_offset"_a,
      "pose_stack_block_type"_a,
      "block_type_n_atoms"_a);
}

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tests
}  // namespace tmol
