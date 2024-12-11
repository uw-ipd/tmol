#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/connection.hh>
#include <tmol/score/common/count_pair.hh>
#include <tmol/score/common/data_loading.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/hash_util.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/uaid_util.hh>
#include <tmol/score/common/warp_segreduce.hh>
#include <tmol/score/common/warp_stride_reduce.hh>

#include <tmol/score/constraint/potentials/constraint_score.hh>

// Operator definitions; safe for CPU compilation
#include <moderngpu/operators.hxx>

#include <chrono>

namespace tmol {
namespace score {
namespace constraint {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;
template <typename Real>
using CoordQuad = Eigen::Matrix<Real, 4, 3>;
#define Real3 Vec<Real, 3>

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real>
auto GetTorsionAngleDispatch<DeviceDispatch, D, Real>::forward(
    TView<Vec<Real, 3>, 2, D> coords)
    -> std::tuple<TPack<Real, 1, D>, TPack<Vec<Real, 3>, 2, D>> {
  using tmol::score::common::accumulate;

  int const n_angles = coords.size(0);

  TPack<Real, 1, D> V_t = TPack<Real, 1, D>::zeros({n_angles});

  auto dV_dx_t = TPack<Vec<Real, 3>, 2, D>::zeros({n_angles, 4});

  auto V = V_t.view;
  auto dV_dx = dV_dx_t.view;

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;

  auto func = ([=] TMOL_DEVICE_FUNC(int angle_index) {
    Real score = 0;

    Real3 atm1 = coords[angle_index][0];
    Real3 atm2 = coords[angle_index][1];
    Real3 atm3 = coords[angle_index][2];
    Real3 atm4 = coords[angle_index][3];

    auto torsion =
        score::common::dihedral_angle<Real>::V_dV(atm1, atm2, atm3, atm4);

    accumulate<D, Real>::add(V[angle_index], torsion.V);
    accumulate<D, Vec<Real, 3>>::add(dV_dx[angle_index][0], torsion.dV_dI);
    accumulate<D, Vec<Real, 3>>::add(dV_dx[angle_index][1], torsion.dV_dJ);
    accumulate<D, Vec<Real, 3>>::add(dV_dx[angle_index][2], torsion.dV_dK);
    accumulate<D, Vec<Real, 3>>::add(dV_dx[angle_index][3], torsion.dV_dL);
  });

  DeviceDispatch<D>::template forall<launch_t>(n_angles, func);

  return {V_t, dV_dx_t};
}

}  // namespace potentials
}  // namespace constraint
}  // namespace score
}  // namespace tmol
