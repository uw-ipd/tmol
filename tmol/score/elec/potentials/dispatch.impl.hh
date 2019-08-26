#pragma once

#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <tuple>

#include <tmol/score/common/accumulate.hh>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/score/common/tuple.hh>

#include <tmol/score/elec/potentials/potentials.hh>

#include "params.hh"

namespace tmol {
namespace score {
namespace elec {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device>
    class SingleDispatch,
    template <tmol::Device>
    class PairDispatch,
    tmol::Device Dev,
    typename Real,
    typename Int>
struct ElecDispatch {
  static auto f(
      TView<Vec<Real, 3>, 2, Dev> coords_i,
      TView<Real, 2, Dev> e_i,
      TView<Vec<Real, 3>, 2, Dev> coords_j,
      TView<Real, 2, Dev> e_j,
      TView<Real, 3, Dev> bonded_path_lengths,
      TView<ElecGlobalParams<float>, 1, Dev> global_params)
      -> std::tuple<
          TPack<float, 1, Dev>,
          TPack<Vec<Real, 3>, 2, Dev>,
          TPack<Vec<Real, 3>, 2, Dev>> {
    int nstacks = coords_i.size(0);
    auto Vs_t = TPack<float, 1, Dev>::zeros({nstacks});
    auto Vs_accum_t = TPack<float, 2, Dev>::zeros({nstacks, 2});
    //auto Vs_i_ats_t = TPack<Real, 3, Dev>::zeros({nstacks, coords_i.size(1), 2});
    //auto Vs_j_ats_t = TPack<Real, 3, Dev>::zeros({nstacks, coords_j.size(1), 2});
    auto dVs_dI_t =
        TPack<Vec<Real, 3>, 2, Dev>::zeros({nstacks, coords_i.size(1)});
    auto dVs_dJ_t =
        TPack<Vec<Real, 3>, 2, Dev>::zeros({nstacks, coords_j.size(1)});

    auto Vs = Vs_t.view;
    auto Vs_accum = Vs_accum_t.view;
    //auto Vs_i_ats = Vs_i_ats_t.view;
    //auto Vs_j_ats = Vs_j_ats_t.view;
    auto dVs_dI = dVs_dI_t.view;
    auto dVs_dJ = dVs_dJ_t.view;

    Real threshold_distance = 6.0;  // fd  make this a parameter...

    PairDispatch<Dev>::forall_stacked_pairs(
        threshold_distance,
        coords_i,
        coords_j,
        [=] EIGEN_DEVICE_FUNC(int stack, int i, int j) {
          auto dist_r =
              distance<Real>::V_dV(coords_i[stack][i], coords_j[stack][j]);
          auto& dist = dist_r.V;
          auto& ddist_dI = dist_r.dV_dA;
          auto& ddist_dJ = dist_r.dV_dB;

          Real V, dV_dDist;
          tie(V, dV_dDist) = elec_delec_ddist(
              dist,
              e_i[stack][i],
              e_j[stack][j],
              bonded_path_lengths[stack][i][j],
              global_params[0].D,
              global_params[0].D0,
              global_params[0].S,
              global_params[0].min_dis,
              global_params[0].max_dis);

	  // Kahan summation to reduce numerical noise
	  accumulate_kahan<Dev, float>::add(&Vs_accum[stack][0], V);

	  // after accumulating, copy over the result into the output
	  // tensor; the last thread to complete this will have it right
	  Vs[stack] = Vs_accum[stack][0];
          accumulate<Dev, Vec<Real, 3>>::add(
              dVs_dI[stack][i], dV_dDist * ddist_dI);
          accumulate<Dev, Vec<Real, 3>>::add(
              dVs_dJ[stack][j], dV_dDist * ddist_dJ);
        });

    return {Vs_t, dVs_dI_t, dVs_dJ_t};
  }
};

}  // namespace potentials
}  // namespace elec
}  // namespace score
}  // namespace tmol
