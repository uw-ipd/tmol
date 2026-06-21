#pragma once

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/extern/moderngpu/loadstore.hxx>
#include <tmol/extern/moderngpu/meta.hxx>

namespace tmol {
namespace score {
namespace common {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <class Real>
MGPU_DEVICE inline void coalesced_read_of_32_coords_into_shared(
    TensorAccessor<Vec<Real, 3>, 1, tmol::Device::CUDA> global_coords,
    int offset,
    Real* shared_coords_array,
    int tid) {
  // coords is a tensor of max_n_natoms x 3 and contains
  // data stored in global memory

  // for (int ii = 0; ii < 3; ++ii) {
  //   int ii_ind = ii * 32 + tid;
  //   int local_atomind = ii_ind / 3;
  //   int atid = local_atomind + offset;
  //   int dim = ii_ind % 3;
  //   if (atid < global_coords.size(0)) {
  //     shared_coords_array[ii_ind] = global_coords[atid][dim];
  //   }
  // }

  using namespace mgpu;

  mem_to_shared<32, 3, 3>(
      &global_coords[offset][0], tid, 32 * 3, shared_coords_array, false);
}

}  // namespace common
}  // namespace score
}  // namespace tmol
