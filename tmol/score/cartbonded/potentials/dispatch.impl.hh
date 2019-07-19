#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

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
struct CartBondedDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, D> coords,
      TView<CartBondedLengthParams<Int>, 1, D> cbl_atoms,
      TView<CartBondedAngleParams<Int>, 1, D> cba_atoms,
      TView<CartBondedTorsionParams<Int>, 1, D> cbt_atoms,
      TView<CartBondedTorsionParams<Int>, 1, D> cbi_atoms,
      TView<CartBondedTorsionParams<Int>, 1, D> cbhxl_atoms,
      TView<CartBondedHarmonicTypeParams<Real>, 1, D> cbl_params,
      TView<CartBondedHarmonicTypeParams<Real>, 1, D> cba_params,
      TView<CartBondedPeriodicTypeParams<Real>, 1, D> cbt_params,
      TView<CartBondedPeriodicTypeParams<Real>, 1, D> cbi_params,
      TView<CartBondedSinusoidalTypeParams<Real>, 1, D> cbhxl_params)
      -> std::tuple<TPack<Real, 1, D>, TPack<Vec<Real, 3>, 2, D>> {
    auto V_t = TPack<Real, 1, D>::zeros({5});
    auto dV_dx_t = TPack<Vec<Real, 3>, 2, D>::zeros({coords.size(0), 5});
    auto V = V_t.view;
    auto dV_dx = dV_dx_t.view;

    // length
    auto cbl_score_i = ([=] EIGEN_DEVICE_FUNC(int i) {
      Int ati = cbl_atoms[i].atom_index_i;
      Int atj = cbl_atoms[i].atom_index_j;
      Int pari = cbl_atoms[i].param_index;
      auto cblength = cblength_V_dV(
          coords[ati], coords[atj], cbl_params[pari].K, cbl_params[pari].x0);

      accumulate<D, Real>::add(V[0], common::get<0>(cblength));
      accumulate<D, Vec<Real, 3>>::add(dV_dx[ati][0], common::get<1>(cblength));
      accumulate<D, Vec<Real, 3>>::add(dV_dx[atj][0], common::get<2>(cblength));
    });
    Dispatch<D>::forall(cbl_atoms.size(0), cbl_score_i);

    // angle
    auto cba_score_i = ([=] EIGEN_DEVICE_FUNC(int i) {
      Int ati = cba_atoms[i].atom_index_i;
      Int atj = cba_atoms[i].atom_index_j;
      Int atk = cba_atoms[i].atom_index_k;
      Int pari = cba_atoms[i].param_index;
      auto cbangle = cbangle_V_dV(
          coords[ati],
          coords[atj],
          coords[atk],
          cba_params[pari].K,
          cba_params[pari].x0);

      accumulate<D, Real>::add(V[1], common::get<0>(cbangle));
      accumulate<D, Vec<Real, 3>>::add(dV_dx[ati][1], common::get<1>(cbangle));
      accumulate<D, Vec<Real, 3>>::add(dV_dx[atj][1], common::get<2>(cbangle));
      accumulate<D, Vec<Real, 3>>::add(dV_dx[atk][1], common::get<3>(cbangle));
    });
    Dispatch<D>::forall(cba_atoms.size(0), cba_score_i);

    // torsion
    auto cbt_score_i = ([=] EIGEN_DEVICE_FUNC(int i) {
      Int ati = cbt_atoms[i].atom_index_i;
      Int atj = cbt_atoms[i].atom_index_j;
      Int atk = cbt_atoms[i].atom_index_k;
      Int atl = cbt_atoms[i].atom_index_l;
      Int pari = cbt_atoms[i].param_index;
      auto cbtorsion = cbtorsion_V_dV(
          coords[ati],
          coords[atj],
          coords[atk],
          coords[atl],
          cbt_params[pari].K,
          cbt_params[pari].x0,
          cbt_params[pari].period);

      accumulate<D, Real>::add(V[2], common::get<0>(cbtorsion));
      accumulate<D, Vec<Real, 3>>::add(
          dV_dx[ati][2], common::get<1>(cbtorsion));
      accumulate<D, Vec<Real, 3>>::add(
          dV_dx[atj][2], common::get<2>(cbtorsion));
      accumulate<D, Vec<Real, 3>>::add(
          dV_dx[atk][2], common::get<3>(cbtorsion));
      accumulate<D, Vec<Real, 3>>::add(
          dV_dx[atl][2], common::get<4>(cbtorsion));
    });
    Dispatch<D>::forall(cbt_atoms.size(0), cbt_score_i);

    // improper torsion
    auto cbi_score_i = ([=] EIGEN_DEVICE_FUNC(int i) {
      Int ati = cbi_atoms[i].atom_index_i;
      Int atj = cbi_atoms[i].atom_index_j;
      Int atk = cbi_atoms[i].atom_index_k;
      Int atl = cbi_atoms[i].atom_index_l;
      Int pari = cbi_atoms[i].param_index;
      auto cbimproper = cbtorsion_V_dV(
          coords[ati],
          coords[atj],
          coords[atk],
          coords[atl],
          cbi_params[pari].K,
          cbi_params[pari].x0,
          cbi_params[pari].period);

      accumulate<D, Real>::add(V[3], common::get<0>(cbimproper));
      accumulate<D, Vec<Real, 3>>::add(
          dV_dx[ati][3], common::get<1>(cbimproper));
      accumulate<D, Vec<Real, 3>>::add(
          dV_dx[atj][3], common::get<2>(cbimproper));
      accumulate<D, Vec<Real, 3>>::add(
          dV_dx[atk][3], common::get<3>(cbimproper));
      accumulate<D, Vec<Real, 3>>::add(
          dV_dx[atl][3], common::get<4>(cbimproper));
    });
    Dispatch<D>::forall(cbi_atoms.size(0), cbi_score_i);

    // hydroxyl torsion
    auto cbhxl_score_i = ([=] EIGEN_DEVICE_FUNC(int i) {
      Int ati = cbhxl_atoms[i].atom_index_i;
      Int atj = cbhxl_atoms[i].atom_index_j;
      Int atk = cbhxl_atoms[i].atom_index_k;
      Int atl = cbhxl_atoms[i].atom_index_l;
      Int pari = cbhxl_atoms[i].param_index;
      auto cbhxltorsion = cbhxltorsion_V_dV(
          coords[ati],
          coords[atj],
          coords[atk],
          coords[atl],
          cbhxl_params[pari].k1,
          cbhxl_params[pari].k2,
          cbhxl_params[pari].k3,
          cbhxl_params[pari].phi1,
          cbhxl_params[pari].phi2,
          cbhxl_params[pari].phi3);

      accumulate<D, Real>::add(V[4], common::get<0>(cbhxltorsion));
      accumulate<D, Vec<Real, 3>>::add(
          dV_dx[ati][4], common::get<1>(cbhxltorsion));
      accumulate<D, Vec<Real, 3>>::add(
          dV_dx[atj][4], common::get<2>(cbhxltorsion));
      accumulate<D, Vec<Real, 3>>::add(
          dV_dx[atk][4], common::get<3>(cbhxltorsion));
      accumulate<D, Vec<Real, 3>>::add(
          dV_dx[atl][4], common::get<4>(cbhxltorsion));
    });
    Dispatch<D>::forall(cbhxl_atoms.size(0), cbhxl_score_i);

    return {V_t, dV_dx_t};
  }
};

}  // namespace potentials
}  // namespace cartbonded
}  // namespace score
}  // namespace tmol
