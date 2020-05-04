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
      TView<Vec<Real, 3>, 2, D> coords,
      TView<CartBondedLengthParams<Int>, 2, D> cbl_atoms,
      TView<CartBondedAngleParams<Int>, 2, D> cba_atoms,
      TView<CartBondedTorsionParams<Int>, 2, D> cbt_atoms,
      TView<CartBondedTorsionParams<Int>, 2, D> cbi_atoms,
      TView<CartBondedTorsionParams<Int>, 2, D> cbhxl_atoms,
      TView<CartBondedHarmonicTypeParams<Real>, 1, D> cbl_params,
      TView<CartBondedHarmonicTypeParams<Real>, 1, D> cba_params,
      TView<CartBondedPeriodicTypeParams<Real>, 1, D> cbt_params,
      TView<CartBondedPeriodicTypeParams<Real>, 1, D> cbi_params,
      TView<CartBondedSinusoidalTypeParams<Real>, 1, D> cbhxl_params)
      -> std::tuple<TPack<Real, 2, D>, TPack<Vec<Real, 3>, 3, D>> {
    auto nstacks = coords.size(0);
    auto V_t = TPack<Real, 2, D>::zeros({nstacks, 5});
    auto dV_dx_t =
        TPack<Vec<Real, 3>, 3, D>::zeros({nstacks, coords.size(1), 5});
    auto V = V_t.view;
    auto dV_dx = dV_dx_t.view;

    // length
    auto cbl_score_i = ([=] EIGEN_DEVICE_FUNC(int stack, int i) {
      Int ati = cbl_atoms[stack][i].atom_index_i;
      Int atj = cbl_atoms[stack][i].atom_index_j;
      Int pari = cbl_atoms[stack][i].param_index;

      // Negative indices are sentinel values for "no bond to score"
      if (ati < 0 || atj < 0) {
        return;
      }

      auto cblength = cblength_V_dV(
          coords[stack][ati],
          coords[stack][atj],
          cbl_params[pari].K,
          cbl_params[pari].x0);

      accumulate<D, Real>::add(V[stack][0], common::get<0>(cblength));
      accumulate<D, Vec<Real, 3>>::add(
          dV_dx[stack][ati][0], common::get<1>(cblength));
      accumulate<D, Vec<Real, 3>>::add(
          dV_dx[stack][atj][0], common::get<2>(cblength));
    });
    Dispatch<D>::forall_stacks(
        cbl_atoms.size(0), cbl_atoms.size(1), cbl_score_i);

    // angle
    auto cba_score_i = ([=] EIGEN_DEVICE_FUNC(int stack, int i) {
      Int ati = cba_atoms[stack][i].atom_index_i;
      Int atj = cba_atoms[stack][i].atom_index_j;
      Int atk = cba_atoms[stack][i].atom_index_k;

      // Negative atom indices are sentinel values for "no angle to score"
      if (ati < 0 || atj < 0 || atk < 0) {
        return;
      }

      Int pari = cba_atoms[stack][i].param_index;
      auto cbangle = cbangle_V_dV(
          coords[stack][ati],
          coords[stack][atj],
          coords[stack][atk],
          cba_params[pari].K,
          cba_params[pari].x0);

      accumulate<D, Real>::add(V[stack][1], common::get<0>(cbangle));
      accumulate<D, Vec<Real, 3>>::add(
          dV_dx[stack][ati][1], common::get<1>(cbangle));
      accumulate<D, Vec<Real, 3>>::add(
          dV_dx[stack][atj][1], common::get<2>(cbangle));
      accumulate<D, Vec<Real, 3>>::add(
          dV_dx[stack][atk][1], common::get<3>(cbangle));
    });
    Dispatch<D>::forall_stacks(
        cba_atoms.size(0), cba_atoms.size(1), cba_score_i);

    // torsion
    auto cbt_score_i = ([=] EIGEN_DEVICE_FUNC(int stack, int i) {
      Int ati = cbt_atoms[stack][i].atom_index_i;
      Int atj = cbt_atoms[stack][i].atom_index_j;
      Int atk = cbt_atoms[stack][i].atom_index_k;
      Int atl = cbt_atoms[stack][i].atom_index_l;
      Int pari = cbt_atoms[stack][i].param_index;

      // Negative atom indices are sentinel values for "no torsion to score"
      if (ati < 0 || atj < 0 || atk < 0 || atl < 0) {
        return;
      }

      auto cbtorsion = cbtorsion_V_dV(
          coords[stack][ati],
          coords[stack][atj],
          coords[stack][atk],
          coords[stack][atl],
          cbt_params[pari].K,
          cbt_params[pari].x0,
          cbt_params[pari].period);

      accumulate<D, Real>::add(V[stack][2], common::get<0>(cbtorsion));
      accumulate<D, Vec<Real, 3>>::add(
          dV_dx[stack][ati][2], common::get<1>(cbtorsion));
      accumulate<D, Vec<Real, 3>>::add(
          dV_dx[stack][atj][2], common::get<2>(cbtorsion));
      accumulate<D, Vec<Real, 3>>::add(
          dV_dx[stack][atk][2], common::get<3>(cbtorsion));
      accumulate<D, Vec<Real, 3>>::add(
          dV_dx[stack][atl][2], common::get<4>(cbtorsion));
    });
    Dispatch<D>::forall_stacks(
        cbt_atoms.size(0), cbt_atoms.size(1), cbt_score_i);

    // improper torsion
    auto cbi_score_i = ([=] EIGEN_DEVICE_FUNC(int stack, int i) {
      Int ati = cbi_atoms[stack][i].atom_index_i;
      Int atj = cbi_atoms[stack][i].atom_index_j;
      Int atk = cbi_atoms[stack][i].atom_index_k;
      Int atl = cbi_atoms[stack][i].atom_index_l;
      Int pari = cbi_atoms[stack][i].param_index;

      // Negative atom indices are sentinel values for "no torsion to score"
      if (ati < 0 || atj < 0 || atk < 0 || atl < 0) {
        return;
      }

      auto cbimproper = cbtorsion_V_dV(
          coords[stack][ati],
          coords[stack][atj],
          coords[stack][atk],
          coords[stack][atl],
          cbi_params[pari].K,
          cbi_params[pari].x0,
          cbi_params[pari].period);

      accumulate<D, Real>::add(V[stack][3], common::get<0>(cbimproper));
      accumulate<D, Vec<Real, 3>>::add(
          dV_dx[stack][ati][3], common::get<1>(cbimproper));
      accumulate<D, Vec<Real, 3>>::add(
          dV_dx[stack][atj][3], common::get<2>(cbimproper));
      accumulate<D, Vec<Real, 3>>::add(
          dV_dx[stack][atk][3], common::get<3>(cbimproper));
      accumulate<D, Vec<Real, 3>>::add(
          dV_dx[stack][atl][3], common::get<4>(cbimproper));
    });
    Dispatch<D>::forall_stacks(
        cbi_atoms.size(0), cbi_atoms.size(1), cbi_score_i);

    // hydroxyl torsion
    auto cbhxl_score_i = ([=] EIGEN_DEVICE_FUNC(int stack, int i) {
      Int ati = cbhxl_atoms[stack][i].atom_index_i;
      Int atj = cbhxl_atoms[stack][i].atom_index_j;
      Int atk = cbhxl_atoms[stack][i].atom_index_k;
      Int atl = cbhxl_atoms[stack][i].atom_index_l;
      Int pari = cbhxl_atoms[stack][i].param_index;

      // Negative atom indices are sentinel values for "no torsion to score"
      if (ati < 0 || atj < 0 || atk < 0 || atl < 0) {
        return;
      }

      auto cbhxltorsion = cbhxltorsion_V_dV(
          coords[stack][ati],
          coords[stack][atj],
          coords[stack][atk],
          coords[stack][atl],
          cbhxl_params[pari].k1,
          cbhxl_params[pari].k2,
          cbhxl_params[pari].k3,
          cbhxl_params[pari].phi1,
          cbhxl_params[pari].phi2,
          cbhxl_params[pari].phi3);

      accumulate<D, Real>::add(V[stack][4], common::get<0>(cbhxltorsion));
      accumulate<D, Vec<Real, 3>>::add(
          dV_dx[stack][ati][4], common::get<1>(cbhxltorsion));
      accumulate<D, Vec<Real, 3>>::add(
          dV_dx[stack][atj][4], common::get<2>(cbhxltorsion));
      accumulate<D, Vec<Real, 3>>::add(
          dV_dx[stack][atk][4], common::get<3>(cbhxltorsion));
      accumulate<D, Vec<Real, 3>>::add(
          dV_dx[stack][atl][4], common::get<4>(cbhxltorsion));
    });
    Dispatch<D>::forall_stacks(
        cbhxl_atoms.size(0), cbhxl_atoms.size(1), cbhxl_score_i);

    return {V_t, dV_dx_t};
  }
};

}  // namespace potentials
}  // namespace cartbonded
}  // namespace score
}  // namespace tmol
