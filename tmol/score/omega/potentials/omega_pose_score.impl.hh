#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/count_pair.hh>
#include <tmol/score/common/data_loading.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/sphere_overlap.impl.hh>
#include <tmol/score/common/tile_atom_pair_evaluation.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/warp_segreduce.hh>
#include <tmol/score/common/warp_stride_reduce.hh>

#include <tmol/score/omega/potentials/omega_pose_score.hh>

// Operator definitions; safe for CPU compilation
#include <moderngpu/operators.hxx>

#include <chrono>

// The maximum number of inter-residue chemical bonds
#define MAX_N_CONN 4
#define TILE_SIZE 32

namespace tmol {
namespace score {
namespace omega {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto OmegaPoseScoreDispatch<DeviceDispatch, D, Real, Int>::f(
    TView<Vec<Real, 3>, 2, D> coords,
    TView<Int, 2, D> pose_stack_block_coord_offset,
    TView<Int, 2, D> pose_stack_block_type,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Int, 3, D> block_type_omega_quad_uaids,

    TView<OmegaGlobalParams<Real>, 1, D> global_params,

    bool compute_derivs

    ) -> std::tuple<TPack<Real, 1, D>, TPack<Vec<Real, 3>, 2, D>> {
  int const n_poses = coords.size(0);
  auto V_t = TPack<Real, 1, D>::zeros({n_poses});
  auto dV_dx_t = TPack<Vec<Real, 3>, 2, D>::zeros({n_poses, coords.size(1)});

  auto V = V_t.view;
  auto dV_dx = dV_dx_t.view;

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;
  // Define nt and reduce_t
  CTA_REAL_REDUCE_T_TYPEDEF;

  auto func = ([=] TMOL_DEVICE_FUNC(int cta) {
    printf("%f\n", global_params[0].K);
    auto omega_a = block_type_omega_quad_uaids;
    for (int ii = 0; ii < omega_a.size(0); ++ii) {
      printf("%i: ", ii);
      for (int jj = 0; jj < omega_a.size(1); ++jj) {
        for (int kk = 0; kk < omega_a.size(2); ++kk) {
          printf("%i ", omega_a[ii][jj][kk]);
        }
        printf("  ");
      }
      printf("\n");
    }
  });

  int num_Vs = coords.size(2);
  DeviceDispatch<D>::template foreach_workgroup<launch_t>(1, func);

  return {V_t, dV_dx_t};
}  // namespace potentials

}  // namespace potentials
}  // namespace omega
}  // namespace score
}  // namespace tmol
