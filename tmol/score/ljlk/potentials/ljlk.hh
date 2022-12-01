#pragma once

#include <tmol/utility/tensor/TensorAccessor.h>

#include <tmol/score/common/data_loading.hh>
#include <tmol/score/ljlk/potentials/common.hh>
#include <tmol/score/ljlk/potentials/lj.hh>
#include <tmol/score/ljlk/potentials/lk_isotropic.hh>

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template <typename Real>
class LJLKSingleResData {
 public:
  int block_type;
  int block_coord_offset;
  int n_atoms;
  int n_conn;
  int n_heavy;
  Real *coords;
  LJLKTypeParams<Real> *params;
  unsigned char *heavy_inds;
  unsigned char *path_dist;
};

template <typename Real>
class LJLKScoringData {
 public:
  int pose_ind;
  LJLKSingleResData<Real> r1;
  LJLKSingleResData<Real> r2;
  int max_important_bond_separation;
  int min_separation;
  bool in_count_pair_striking_dist;
  unsigned char *conn_seps;
  LJGlobalParams<Real> global_params;
  Real total_lj;
  Real total_lk;
};

template <typename Real, tmol::Device D>
TMOL_DEVICE_FUNC Real lj_atom_energy_and_derivs_full(
    int atom_tile_ind1,
    int atom_tile_ind2,
    int start_atom1,
    int start_atom2,
    LJLKScoringData<Real> const &score_dat,
    int cp_separation,
    TView<Eigen::Matrix<Real, 3, 1>, 3, D> dV_dcoords) {
  using Real3 = Eigen::Matrix<Real, 3, 1>;

  Real3 coord1 = coord_from_shared(score_dat.r1.coords, atom_tile_ind1);
  Real3 coord2 = coord_from_shared(score_dat.r2.coords, atom_tile_ind2);

  auto dist_r = distance<Real>::V_dV(coord1, coord2);
  auto &dist = dist_r.V;
  auto &ddist_dat1 = dist_r.dV_dA;
  auto &ddist_dat2 = dist_r.dV_dB;
  auto lj = lj_score<Real>::V_dV(
      dist,
      cp_separation,
      score_dat.r1.params[atom_tile_ind1].lj_params(),
      score_dat.r2.params[atom_tile_ind2].lj_params(),
      score_dat.global_params);

  // all threads accumulate derivatives for atom 1 to global memory
  Vec<Real, 3> lj_dxyz_at1 = lj.dV_ddist * ddist_dat1;
  for (int j = 0; j < 3; ++j) {
    if (lj_dxyz_at1[j] != 0) {
      accumulate<D, Real>::add(
          dV_dcoords[0][score_dat.pose_ind]
                    [score_dat.r1.block_coord_offset + atom_tile_ind1
                     + start_atom1][j],
          lj_dxyz_at1[j]);
    }
  }

  // all threads accumulate derivatives for atom 2 to global memory
  Vec<Real, 3> lj_dxyz_at2 = lj.dV_ddist * ddist_dat2;
  for (int j = 0; j < 3; ++j) {
    if (lj_dxyz_at2[j] != 0) {
      accumulate<D, Real>::add(
          dV_dcoords[0][score_dat.pose_ind]
                    [score_dat.r2.block_coord_offset + atom_tile_ind2
                     + start_atom2][j],
          lj_dxyz_at2[j]);
    }
  }
  return lj.V;
}

template <typename Real, tmol::Device D>
TMOL_DEVICE_FUNC Real lk_atom_energy_and_derivs_full(
    int atom_tile_ind1,
    int atom_tile_ind2,
    int start_atom1,
    int start_atom2,
    LJLKScoringData<Real> const &score_dat,
    int cp_separation,
    TView<Eigen::Matrix<Real, 3, 1>, 3, D> dV_dcoords) {
  using Real3 = Eigen::Matrix<Real, 3, 1>;
  Real3 coord1 = coord_from_shared(score_dat.r1.coords, atom_tile_ind1);
  Real3 coord2 = coord_from_shared(score_dat.r2.coords, atom_tile_ind2);

  auto dist_r = distance<Real>::V_dV(coord1, coord2);
  auto &dist = dist_r.V;
  auto &ddist_dat1 = dist_r.dV_dA;
  auto &ddist_dat2 = dist_r.dV_dB;
  auto lk = lk_isotropic_score<Real>::V_dV(
      dist,
      cp_separation,
      score_dat.r1.params[atom_tile_ind1].lk_params(),
      score_dat.r2.params[atom_tile_ind2].lk_params(),
      score_dat.global_params);

  Vec<Real, 3> lk_dxyz_at1 = lk.dV_ddist * ddist_dat1;
  for (int j = 0; j < 3; ++j) {
    if (lk_dxyz_at1[j] != 0) {
      accumulate<D, Real>::add(
          dV_dcoords[1][score_dat.pose_ind]
                    [score_dat.r1.block_coord_offset + atom_tile_ind1
                     + start_atom1][j],
          lk_dxyz_at1[j]);
    }
  }

  Vec<Real, 3> lk_dxyz_at2 = lk.dV_ddist * ddist_dat2;
  for (int j = 0; j < 3; ++j) {
    if (lk_dxyz_at2[j] != 0) {
      accumulate<D, Real>::add(
          dV_dcoords[1][score_dat.pose_ind]
                    [score_dat.r2.block_coord_offset + atom_tile_ind2
                     + start_atom2][j],
          lk_dxyz_at2[j]);
    }
  }
  return lk.V;
}

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
