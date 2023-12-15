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

#include "lj.dispatch.hh"
#include "lj.hh"

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto LJDispatch<Dispatch, D, Real, Int>::f(
    TView<Vec<Real, 3>, 2, D> coords_i,
    TView<Int, 2, D> atom_type_i,

    TView<Vec<Real, 3>, 2, D> coords_j,
    TView<Int, 2, D> atom_type_j,

    TView<Real, 3, D> bonded_path_lengths,
    TView<LJTypeParams<Real>, 1, D> type_params,
    TView<LJGlobalParams<Real>, 1, D> global_params)
    -> std::tuple<
        TPack<Real, 1, D>,
        TPack<Vec<Real, 3>, 2, D>,
        TPack<Vec<Real, 3>, 2, D>> {
  NVTXRange _function(__FUNCTION__);

  nvtx_range_push("dispatch::score");
  assert(coords_i.size(0) == coords_j.size(0));
  int const nstacks = coords_i.size(0);

  auto V_t = TPack<Real, 1, D>::zeros({nstacks});
  auto dV_dI_t = TPack<Vec<Real, 3>, 2, D>::zeros({nstacks, coords_i.size(1)});
  auto dV_dJ_t = TPack<Vec<Real, 3>, 2, D>::zeros({nstacks, coords_j.size(1)});

  auto V = V_t.view;
  auto dV_dI = dV_dI_t.view;
  auto dV_dJ = dV_dJ_t.view;

  // auto wcts = std::chrono::system_clock::now();
  // clock_t start_time = clock();
  // auto count_t = TPack<int, 1, D>::zeros({1});
  // auto count = count_t.view;
  nvtx_range_pop();

  nvtx_range_push("dispatch::score");
  Real threshold_distance = 6.0;
  Dispatch<D>::forall_stacked_pairs(
      threshold_distance,
      coords_i,
      coords_j,
      [=] EIGEN_DEVICE_FUNC(int stack, int i, int j) {
        Int ati = atom_type_i[stack][i];
        Int atj = atom_type_j[stack][j];

        auto dist_r =
            distance<Real>::V_dV(coords_i[stack][i], coords_j[stack][j]);
        auto& dist = dist_r.V;
        auto& ddist_dI = dist_r.dV_dA;
        auto& ddist_dJ = dist_r.dV_dB;

        auto lj = lj_score<Real>::V_dV(
            dist,
            bonded_path_lengths[stack][i][j],
            type_params[ati],
            type_params[atj],
            global_params[0]);

        Real ljV =
            lj.Vatr + lj.Vrep;  // fd no need to split since this is going away
        Real ljdVddist = lj.dVatr_ddist + lj.dVrep_ddist;
        accumulate<D, Real>::add_one_dst(V, stack, ljV);
        accumulate<D, Vec<Real, 3>>::add(dV_dI[stack][i], ljdVddist * ddist_dI);
        accumulate<D, Vec<Real, 3>>::add(dV_dJ[stack][j], ljdVddist * ddist_dJ);
        // accumulate<D, int>::add(count[0], 1);
      });
  nvtx_range_pop();

  nvtx_range_pop();

  // #ifdef __CUDACC__
  //   int count_total;
  //   cudaMemcpy(&count_total, &count[0], sizeof(int), cudaMemcpyDeviceToHost);
  //
  //   clock_t stop_time = clock();
  //
  //   std::chrono::duration<double> wctduration =
  //       (std::chrono::system_clock::now() - wcts);
  //
  //   std::cout << nstacks << " " << coords_i.size(1) << " " <<
  //   coords_j.size(1)
  //             << " ";
  //   std::cout << nstacks * coords_i.size(1) * coords_j.size(1) << " ";
  //   std::cout << "runtime? " << ((double)stop_time - start_time) /
  //   CLOCKS_PER_SEC
  //             << " wall time: " << wctduration.count()
  //             << " count= " << count_total << std::endl;
  // #endif

  return {V_t, dV_dI_t, dV_dJ_t};
};

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
