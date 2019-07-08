#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <ATen/cuda/CUDAStream.h>

#include <tmol/utility/cuda/stream.hh>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/tuple.hh>

#include "lk_isotropic.dispatch.hh"
#include "lk_isotropic.hh"

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device>
    class SingleDispatch,
    template <tmol::Device>
    class PairDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto LKIsotropicDispatch<SingleDispatch, PairDispatch, D, Real, Int>::f(
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

  // clock_t start1 = clock();
  // if (D == tmol::Device::CUDA) {
  //   int orig = std::cout.precision();
  //   std::cout.precision(16);
  //   std::cout << "lk_start " << (double)start / CLOCKS_PER_SEC * 1000000
  //             << std::endl;
  //   std::cout.precision(orig);
  // }
  NVTXRange _function(__FUNCTION__);

  NVTXRange _allocate("allocate");

  auto stream = utility::cuda::get_cuda_stream_from_pool();
  utility::cuda::set_current_cuda_stream(stream);

  auto V_t = TPack<Real, 1, D>::empty({1});
  auto dV_dI_t = TPack<Vec<Real, 3>, 1, D>::empty({coords_i.size(0)});
  auto dV_dJ_t = TPack<Vec<Real, 3>, 1, D>::empty({coords_j.size(0)});

  auto V = V_t.view;
  auto dV_dI = dV_dI_t.view;
  auto dV_dJ = dV_dJ_t.view;
  
  _allocate.exit();
  NVTXRange _zero("zero");
  auto zero = [=] EIGEN_DEVICE_FUNC (int i) {
    if (i < 3) {
      V[i] = 0;
    }
    if (i < dV_dI.size(0)) {
      for (int j = 0; j < 3; ++j) {
	dV_dI[i](j) = 0;
      }
    }
    if (i < dV_dJ.size(0)) {
      for (int j = 0; j < 3; ++j) {
	dV_dJ[i](j) = 0;
      }
    }
  };
  int largest = std::max(3, (int)std::max(coords_i.size(0), coords_j.size(0)));
  SingleDispatch<D>::forall(largest, zero, stream);
  _zero.exit();
  
  clock_t start2 = clock();

  NVTXRange _score("score");
  Real threshold_distance = 6.0;
  PairDispatch<D>::forall_pairs(
      threshold_distance,
      coords_i,
      coords_j,
      [=] EIGEN_DEVICE_FUNC(int i, int j) {
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

        accumulate<D, Real>::add(V[0], lk.V);
        accumulate<D, Vec<Real, 3>>::add(dV_dI[i], lk.dV_ddist * ddist_dI);
        accumulate<D, Vec<Real, 3>>::add(dV_dJ[j], lk.dV_ddist * ddist_dJ);
      },
      stream);
  _score.exit();
  
  // clock_t start3 = clock();
  // if (D == tmol::Device::CUDA) {
  //   int orig = std::cout.precision();
  //   std::cout.precision(16);
  //   std::cout << "lk "
  //   //<< " a " << ((double)start2 - start1) / CLOCKS_PER_SEC * 1000000
  //   //<< " b " << ((double)start3 - start2) / CLOCKS_PER_SEC * 1000000
  //   //<< " c "
  //   << ((double)start3 - start1) / CLOCKS_PER_SEC * 1000000
  //   << "\n";
  //   std::cout.precision(orig);
  // }

  // restore the global stream to default before leaving
  utility::cuda::set_default_cuda_stream();

  return {V_t, dV_dI_t, dV_dJ_t};
};

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
