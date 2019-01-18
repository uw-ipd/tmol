#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>

#include <tmol/score/common/dispatch.hh>
#include <tmol/score/common/geom.hh>

#include "lj.hh"
#include "lk_isotropic.hh"

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

using std::tie;
using std::tuple;

template <typename Dispatch, typename Real, typename Int>
auto lj_dispatch(
    TView<Vec<Real, 3>, 1> coords_i,
    TView<Int, 1> atom_type_i,

    TView<Vec<Real, 3>, 1> coords_j,
    TView<Int, 1> atom_type_j,

    TView<Real, 2> bonded_path_lengths,

    LJTypeParams_targs(1),
    LJGlobalParams_args()) -> tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> {
  using tmol::new_tensor;

  Dispatch dispatcher(6.0, coords_i.size(0), coords_j.size(0));
  auto num_Vs = dispatcher.scan(coords_i, coords_j);

  at::Tensor inds_t;
  TView<int64_t, 2> inds;
  tie(inds_t, inds) = new_tensor<int64_t, 2>({num_Vs, 2});

  at::Tensor Vs_t;
  TView<Real, 1> Vs;
  tie(Vs_t, Vs) = new_tensor<Real, 1>(num_Vs);

  at::Tensor dV_dIs_t;
  TView<Vec<Real, 3>, 1> dV_dIs;
  tie(dV_dIs_t, dV_dIs) = new_tensor<Vec<Real, 3>, 1>(num_Vs);

  at::Tensor dV_dJs_t;
  TView<Vec<Real, 3>, 1> dV_dJs;
  tie(dV_dJs_t, dV_dJs) = new_tensor<Vec<Real, 3>, 1>(num_Vs);

  dispatcher.score([=](int o, int i, int j) mutable {
    Int ati = atom_type_i[i];
    Int atj = atom_type_j[j];

    Real D;
    Vec<Real, 3> dD_dI, dD_dJ;
    tie(D, dD_dI, dD_dJ) = distance_V_dV(coords_i[i], coords_j[j]);

    Real V, dV_dD;
    tie(V, dV_dD) = lj_score_V_dV(
        D,
        bonded_path_lengths[i][j],
        LJTypeParams_struct(, [ati]),
        LJTypeParams_struct(, [atj]),
        LJGlobalParams_struct());

    inds[o][0] = i;
    inds[o][1] = j;

    Vs[o] = V;
    dV_dIs[o] = dV_dD * dD_dI;
    dV_dJs[o] = dV_dD * dD_dJ;
  });

  return {inds_t, Vs_t, dV_dIs_t, dV_dJs_t};
}

template <typename Dispatch, typename Real, typename Int>
auto lk_isotropic_dispatch(
    TView<Vec<Real, 3>, 1> coords_i,
    TView<Int, 1> atom_type_i,

    TView<Vec<Real, 3>, 1> coords_j,
    TView<Int, 1> atom_type_j,

    TView<Real, 2> bonded_path_lengths,

    LKTypeParams_targs(1),
    LJGlobalParams_args()) -> tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> {
  using tmol::new_tensor;

  Dispatch dispatcher(6.0, coords_i.size(0), coords_j.size(0));
  auto num_Vs = dispatcher.scan(coords_i, coords_j);

  at::Tensor inds_t;
  TView<int64_t, 2> inds;
  tie(inds_t, inds) = new_tensor<int64_t, 2>({num_Vs, 2});

  at::Tensor Vs_t;
  TView<Real, 1> Vs;
  tie(Vs_t, Vs) = new_tensor<Real, 1>(num_Vs);

  at::Tensor dV_dIs_t;
  TView<Vec<Real, 3>, 1> dV_dIs;
  tie(dV_dIs_t, dV_dIs) = new_tensor<Vec<Real, 3>, 1>(num_Vs);

  at::Tensor dV_dJs_t;
  TView<Vec<Real, 3>, 1> dV_dJs;
  tie(dV_dJs_t, dV_dJs) = new_tensor<Vec<Real, 3>, 1>(num_Vs);

  dispatcher.score([=](int o, int i, int j) mutable {
    Int ati = atom_type_i[i];
    Int atj = atom_type_j[j];

    Real D;
    Vec<Real, 3> dD_dI, dD_dJ;
    tie(D, dD_dI, dD_dJ) = distance_V_dV(coords_i[i], coords_j[j]);

    Real V, dV_dD;
    tie(V, dV_dD) = lk_isotropic_score_V_dV(
        D,
        bonded_path_lengths[i][j],
        LKTypeParams_struct(, [ati]),
        LKTypeParams_struct(, [atj]),
        LJGlobalParams_struct());

    inds[o][0] = i;
    inds[o][1] = j;

    Vs[o] = V;
    dV_dIs[o] = dV_dD * dD_dI;
    dV_dJs[o] = dV_dD * dD_dJ;
  });

  return {inds_t, Vs_t, dV_dIs_t, dV_dJs_t};
}
}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
