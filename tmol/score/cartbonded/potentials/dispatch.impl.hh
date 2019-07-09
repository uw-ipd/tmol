#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/cuda/stream.hh>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/dispatch.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/tuple.hh>

#include <tmol/score/ljlk/potentials/params.hh>

#include "params.hh"
#include "potentials.hh"

namespace tmol {
namespace score {
namespace cartbonded {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct CartBondedLengthDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, D> coords,
      TView<CartBondedLengthParams<Int>, 1, D> atom_indices,
      TView<CartBondedHarmonicTypeParams<Real>, 1, D> param_table)
      -> std::tuple<TPack<Real, 1, D>, TPack<Vec<Real, 3>, 1, D>> {
    auto num_Vs = atom_indices.size(0);

    auto stream = utility::cuda::get_cuda_stream_from_pool();
    utility::cuda::set_current_cuda_stream(stream);

    auto V_t = TPack<Real, 1, D>::zeros({1});
    auto dV_dx_t = TPack<Vec<Real, 3>, 1, D>::zeros({coords.size(0)});

    auto V = V_t.view;
    auto dV_dx = dV_dx_t.view;

    auto f_i = ([=] EIGEN_DEVICE_FUNC(int i) {
      Int ati = atom_indices[i].atom_index_i;
      Int atj = atom_indices[i].atom_index_j;
      Int pari = atom_indices[i].param_index;
      auto cblength = cblength_V_dV(
          coords[ati], coords[atj], param_table[pari].K, param_table[pari].x0);

      accumulate<D, Real>::add(V[0], common::get<0>(cblength));
      accumulate<D, Vec<Real, 3>>::add(dV_dx[ati], common::get<1>(cblength));
      accumulate<D, Vec<Real, 3>>::add(dV_dx[atj], common::get<2>(cblength));
    });

    Dispatch<D>::forall(num_Vs, f_i, stream);
    // restore the global stream to default before leaving
    utility::cuda::set_default_cuda_stream();

    return {V_t, dV_dx_t};
  }
};

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct CartBondedAngleDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, D> coords,
      TView<CartBondedAngleParams<Int>, 1, D> atom_indices,
      TView<CartBondedHarmonicTypeParams<Real>, 1, D> param_table)
      -> std::tuple<TPack<Real, 1, D>, TPack<Vec<Real, 3>, 1, D>> {
    auto num_Vs = atom_indices.size(0);

    auto stream = utility::cuda::get_cuda_stream_from_pool();
    utility::cuda::set_current_cuda_stream(stream);

    auto V_t = TPack<Real, 1, D>::zeros({1});
    auto dV_dx_t = TPack<Vec<Real, 3>, 1, D>::zeros({coords.size(0)});

    auto V = V_t.view;
    auto dV_dx = dV_dx_t.view;

    auto f_i = ([=] EIGEN_DEVICE_FUNC(int i) {
      Int ati = atom_indices[i].atom_index_i;
      Int atj = atom_indices[i].atom_index_j;
      Int atk = atom_indices[i].atom_index_k;
      Int pari = atom_indices[i].param_index;
      auto cbangle = cbangle_V_dV(
          coords[ati],
          coords[atj],
          coords[atk],
          param_table[pari].K,
          param_table[pari].x0);

      accumulate<D, Real>::add(V[0], common::get<0>(cbangle));
      accumulate<D, Vec<Real, 3>>::add(dV_dx[ati], common::get<1>(cbangle));
      accumulate<D, Vec<Real, 3>>::add(dV_dx[atj], common::get<2>(cbangle));
      accumulate<D, Vec<Real, 3>>::add(dV_dx[atk], common::get<3>(cbangle));
    });

    Dispatch<D>::forall(num_Vs, f_i, stream);

    // restore the global stream to default before leaving
    utility::cuda::set_default_cuda_stream();
    
    return {V_t, dV_dx_t};
  }
};

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct CartBondedTorsionDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, D> coords,
      TView<CartBondedTorsionParams<Int>, 1, D> atom_indices,
      TView<CartBondedPeriodicTypeParams<Real>, 1, D> param_table)
      -> std::tuple<TPack<Real, 1, D>, TPack<Vec<Real, 3>, 1, D>> {
    auto num_Vs = atom_indices.size(0);

    auto stream = utility::cuda::get_cuda_stream_from_pool();
    utility::cuda::set_current_cuda_stream(stream);

    auto V_t = TPack<Real, 1, D>::zeros({1});
    auto dV_dx_t = TPack<Vec<Real, 3>, 1, D>::zeros({coords.size(0)});

    auto V = V_t.view;
    auto dV_dx = dV_dx_t.view;

    auto f_i = ([=] EIGEN_DEVICE_FUNC(int i) {
      Int ati = atom_indices[i].atom_index_i;
      Int atj = atom_indices[i].atom_index_j;
      Int atk = atom_indices[i].atom_index_k;
      Int atl = atom_indices[i].atom_index_l;
      Int pari = atom_indices[i].param_index;
      auto cbtorsion = cbtorsion_V_dV(
          coords[ati],
          coords[atj],
          coords[atk],
          coords[atl],
          param_table[pari].K,
          param_table[pari].x0,
          param_table[pari].period);

      accumulate<D, Real>::add(V[0], common::get<0>(cbtorsion));
      accumulate<D, Vec<Real, 3>>::add(dV_dx[ati], common::get<1>(cbtorsion));
      accumulate<D, Vec<Real, 3>>::add(dV_dx[atj], common::get<2>(cbtorsion));
      accumulate<D, Vec<Real, 3>>::add(dV_dx[atk], common::get<3>(cbtorsion));
      accumulate<D, Vec<Real, 3>>::add(dV_dx[atl], common::get<4>(cbtorsion));
    });

    Dispatch<D>::forall(num_Vs, f_i, stream);

    // restore the global stream to default before leaving
    utility::cuda::set_default_cuda_stream();

    return {V_t, dV_dx_t};
  }
};

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct CartBondedHxlTorsionDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, D> coords,
      TView<CartBondedTorsionParams<Int>, 1, D> atom_indices,
      TView<CartBondedSinusoidalTypeParams<Real>, 1, D> param_table)
      -> std::tuple<TPack<Real, 1, D>, TPack<Vec<Real, 3>, 1, D>> {
    auto num_Vs = atom_indices.size(0);

    auto stream = utility::cuda::get_cuda_stream_from_pool();
    utility::cuda::set_current_cuda_stream(stream);

    auto V_t = TPack<Real, 1, D>::zeros({1});
    auto dV_dx_t = TPack<Vec<Real, 3>, 1, D>::zeros({coords.size(0)});

    auto V = V_t.view;
    auto dV_dx = dV_dx_t.view;

    auto f_i = ([=] EIGEN_DEVICE_FUNC(int i) {
      Int ati = atom_indices[i].atom_index_i;
      Int atj = atom_indices[i].atom_index_j;
      Int atk = atom_indices[i].atom_index_k;
      Int atl = atom_indices[i].atom_index_l;
      Int pari = atom_indices[i].param_index;
      auto cbhxltorsion = cbhxltorsion_V_dV(
          coords[ati],
          coords[atj],
          coords[atk],
          coords[atl],
          param_table[pari].k1,
          param_table[pari].k2,
          param_table[pari].k3,
          param_table[pari].phi1,
          param_table[pari].phi2,
          param_table[pari].phi3);

      accumulate<D, Real>::add(V[0], common::get<0>(cbhxltorsion));
      accumulate<D, Vec<Real, 3>>::add(
          dV_dx[ati], common::get<1>(cbhxltorsion));
      accumulate<D, Vec<Real, 3>>::add(
          dV_dx[atj], common::get<2>(cbhxltorsion));
      accumulate<D, Vec<Real, 3>>::add(
          dV_dx[atk], common::get<3>(cbhxltorsion));
      accumulate<D, Vec<Real, 3>>::add(
          dV_dx[atl], common::get<4>(cbhxltorsion));
    });

    Dispatch<D>::forall(num_Vs, f_i, stream);

    // restore the global stream to default before leaving
    utility::cuda::set_default_cuda_stream();
    
    return {V_t, dV_dx_t};
  }
};

}  // namespace potentials
}  // namespace cartbonded
}  // namespace score
}  // namespace tmol
