#include <tmol/score/common/simple_dispatch.cuda.impl.cuh>

// TEMPT !! #include "lk_isotropic.dispatch.impl.hh"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/dist_pairs.cuda.impl.cuh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/tuple.hh>

#include "lk_isotropic.dispatch.hh"
#include "lk_isotropic.hh"

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

// TEMP !! #define declare_dispatch(Real, Int)    \
// TEMP !!                                        \
// TEMP !!   template struct LKIsotropicDispatch< \
// TEMP !!       AABBDispatch,                    \
// TEMP !!       tmol::Device::CUDA,              \
// TEMP !!       Real,                            \
// TEMP !!       Int>;                            \
// TEMP !!   template struct LKIsotropicDispatch< \
// TEMP !!       AABBTriuDispatch,                \
// TEMP !!       tmol::Device::CUDA,              \
// TEMP !!       Real,                            \
// TEMP !!       Int>;
// TEMP !!
// TEMP !! declare_dispatch(float, int64_t);
// TEMP !! declare_dispatch(double, int64_t);
// TEMP !!
// TEMP !! #undef declare_dispatch

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto LKIsotropicDispatch<Dispatch, D, Real, Int>::f(
    TView<Vec<Real, 3>, 1, D> coords_i,
    TView<Int, 1, D> atom_type_i,

    TView<Vec<Real, 3>, 1, D> coords_j,
    TView<Int, 1, D> atom_type_j,

    TView<Real, 2, D> bonded_path_lengths,

    TView<LKTypeParams<Real>, 1, D> type_params,
    TView<LJGlobalParams<Real>, 1, D> global_params)
    -> std::tuple<
        TPack<Real, 1, D>,
        TPack<Vec<Real, 3>, 1, D>,
        TPack<Vec<Real, 3>, 1, D>> {
  NVTXRange _function(__FUNCTION__);

  nvtx_range_push("alloc");
  auto V_t = TPack<Real, 1, D>::zeros({1});
  auto dV_dI_t = TPack<Vec<Real, 3>, 1, D>::zeros({coords_i.size(0)});
  auto dV_dJ_t = TPack<Vec<Real, 3>, 1, D>::zeros({coords_j.size(0)});

  auto V = V_t.view;
  auto dV_dI = dV_dI_t.view;
  auto dV_dJ = dV_dJ_t.view;

  int const natoms = coords_i.size(0);
  int npairs = (natoms * (natoms - 1)) / 2;

  auto nearby_t = TPack<Int, 1, D>::empty({npairs});
  auto nearby_scan_t = TPack<Int, 1, D>::empty({npairs});
  auto nearby_pair_i_t = TPack<Int, 1, D>::empty({npairs});
  auto nearby_pair_j_t = TPack<Int, 1, D>::empty({npairs});
  // auto nearby_pair_scores_t = TPack<Real, 1, D>::empty({npairs});

  auto nearby = nearby_t.view;
  auto nearby_scan = nearby_scan_t.view;
  auto nearby_pair_i = nearby_pair_i_t.view;
  auto nearby_pair_j = nearby_pair_j_t.view;
  // auto nearby_pair_scores = nearby_pair_scores_t.view;
  nvtx_range_pop();

  nvtx_range_push("lk::thresh");
  clock_t start = clock();
  int n_nearby = TriuDistanceCutoff<Real, Int, 16, 64>::f(
      coords_i, nearby, nearby_scan, nearby_pair_i, nearby_pair_j, 6.0);
  nvtx_range_pop();
  clock_t stop = clock();
  printf("distpair total: %f\n", ((double)stop - start) / CLOCKS_PER_SEC);

  nvtx_range_push("lk::calc");
  start = clock();
  mgpu::standard_context_t context;
  mgpu::transform(
      ([=] MGPU_DEVICE(int idx) {
        Int i = nearby_pair_i[idx];
        Int j = nearby_pair_j[idx];
        Int ati = atom_type_i[i];
        Int atj = atom_type_j[j];

        auto dist_r = distance<Real>::V_dV(coords_i[i], coords_j[j]);
        auto& dist = dist_r.V;
        auto& ddist_dI = dist_r.dV_dA;
        auto& ddist_dJ = dist_r.dV_dB;

        auto lk = lk_isotropic_score<Real>::V_dV(
            dist,
            bonded_path_lengths[i][j],
            type_params[ati],
            type_params[atj],
            global_params[0]);
        // if using V
        // accumulate<D, Real>::add(V[0], lk);

        // if using V_dV
        accumulate<D, Real>::add(V[0], lk.V);
        accumulate<D, Vec<Real, 3>>::add(dV_dI[i], lk.dV_ddist * ddist_dI);
        accumulate<D, Vec<Real, 3>>::add(dV_dJ[j], lk.dV_ddist * ddist_dJ);
      }),
      n_nearby,
      context);

  // nvtx_range_push("dispatch::score");
  // Real threshold_distance = 6.0;
  // Dispatch<D>::forall_pairs(
  //    threshold_distance,
  //    coords_i,
  //    coords_j,
  //    [=] EIGEN_DEVICE_FUNC(int i, int j) {
  //      Int ati = atom_type_i[i];
  //      Int atj = atom_type_j[j];
  //
  //      auto dist_r = distance<Real>::V_dV(coords_i[i], coords_j[j]);
  //      auto& dist = dist_r.V;
  //      auto& ddist_dI = dist_r.dV_dA;
  //      auto& ddist_dJ = dist_r.dV_dB;
  //
  //      auto lk = lk_isotropic_score<Real>::V_dV(
  //          dist,
  //          bonded_path_lengths[i][j],
  //          type_params[ati],
  //          type_params[atj],
  //          global_params[0]);
  //
  //      accumulate<D, Real>::add(V[0], lk.V);
  //      accumulate<D, Vec<Real, 3>>::add(dV_dI[i], lk.dV_ddist * ddist_dI);
  //      accumulate<D, Vec<Real, 3>>::add(dV_dJ[j], lk.dV_ddist * ddist_dJ);
  //    });
  // nvtx_range_pop();

  nvtx_range_pop();
  float lk0;
  cudaMemcpy(&lk0, &V[0], sizeof(int), cudaMemcpyDeviceToHost);

  stop = clock();
  printf(
      "lk eval took: %f, computed score %f\n",
      ((double)stop - start) / CLOCKS_PER_SEC,
      lk0);

  return {V_t, dV_dI_t, dV_dJ_t};
};

#define declare_dispatch(Real, Int)    \
                                       \
  template struct LKIsotropicDispatch< \
      AABBDispatch,                    \
      tmol::Device::CUDA,              \
      Real,                            \
      Int>;                            \
  template struct LKIsotropicDispatch< \
      AABBTriuDispatch,                \
      tmol::Device::CUDA,              \
      Real,                            \
      Int>;

declare_dispatch(float, int64_t);
declare_dispatch(double, int64_t);

#undef declare_dispatch

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
