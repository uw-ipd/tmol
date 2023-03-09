#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/tuple.hh>

#include "omega.hh"
#include "params.hh"
// #include "rotamer_pair_energy_lj.hh"

namespace tmol {
namespace score {
namespace omega {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device>
    class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
struct OmegaPoseScoreDispatch {
  static auto f(
      TView<Vec<Real, 3>, 2, D> coords,
      TView<Int, 2, D> pose_stack_block_coord_offset,
      TView<Int, 2, D> pose_stack_block_type,
      TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
      TView<Int, 3, D> block_type_omega_quad_uaids,
      TView<Int, 3, D> block_type_atom_downstream_of_conn,

      TView<OmegaGlobalParams<Real>, 1, D> global_params,

      bool compute_derivs

      ) -> std::tuple<TPack<Real, 2, D>, TPack<Vec<Real, 3>, 3, D>>;
};

}  // namespace potentials
}  // namespace omega
}  // namespace score
}  // namespace tmol
