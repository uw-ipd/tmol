#pragma once

#include <math.h>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/score/common/geom.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/tuple_operators.hh>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/numeric/bspline_compiled/bspline.hh>

namespace tmol {
namespace score {
namespace dunbrack {
namespace potentials {

using namespace tmol::score::common;

#define def auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

#define CoordQuad Eigen::Matrix<Real, 4, 3>

constexpr float operator"" _2rad(long double deg) {
  return float(M_PI * deg / 180.);
}

template <typename Real, typename Int, tmol::Device D>
def measure_dihedrals_V_dV(
    TView<Eigen::Matrix<Real, 3, 1>, 1, D> coordinates,
    int i,
    TView<Vec<Int, 4>, 1, D> dihedral_atom_inds,
    TView<Real, 1, D> dihedrals,
    TView<Eigen::Matrix<Real, 4, 3>, 1, D> ddihe_dxyz)
    ->void {
  Int at1 = dihedral_atom_inds[i][0];
  Int at2 = dihedral_atom_inds[i][1];
  Int at3 = dihedral_atom_inds[i][2];
  Int at4 = dihedral_atom_inds[i][3];
  if (at1 >= 0) {
    auto dihe = dihedral_angle<Real>::V_dV(
        coordinates[at1], coordinates[at2], coordinates[at3], coordinates[at4]);
    dihedrals[i] = dihe.V;

    // CUDA required me to unroll these loops instead of
    // using "ddihe_dxyz[i].row(0) = dihe.dV_dI;"
    // I got errors looking like:
    //
    // /home/andrew/rosetta/GIT/tmol/tmol/extern/Eigen/src/Core/AssignEvaluator.h(720):
    // warning: calling a __host__ function from a __host__ __device__ function
    // is not allowed
    //
    // and I don't know why this error shows up here in dunbrack, but not in
    // rama.
    //
    ddihe_dxyz[i](0, 0) = dihe.dV_dI(0);
    ddihe_dxyz[i](0, 1) = dihe.dV_dI(1);
    ddihe_dxyz[i](0, 2) = dihe.dV_dI(2);
    ddihe_dxyz[i](1, 0) = dihe.dV_dJ(0);
    ddihe_dxyz[i](1, 1) = dihe.dV_dJ(1);
    ddihe_dxyz[i](1, 2) = dihe.dV_dJ(2);
    ddihe_dxyz[i](2, 0) = dihe.dV_dK(0);
    ddihe_dxyz[i](2, 1) = dihe.dV_dK(1);
    ddihe_dxyz[i](2, 2) = dihe.dV_dK(2);
    ddihe_dxyz[i](3, 0) = dihe.dV_dL(0);
    ddihe_dxyz[i](3, 1) = dihe.dV_dL(1);
    ddihe_dxyz[i](3, 2) = dihe.dV_dL(2);
  } else if (at1 == -1) {
    // -1 is sentinel value for undefined phi dihedral
    dihedrals[i] = -60.0 * M_PI / 180;
  } else if (at1 == -2) {
    // -2 is sentinel value for undefined psi dihedral
    dihedrals[i] = 60.0 * M_PI / 180;
  }
}

template <typename Real, tmol::Device D>
def classify_rotamer(
    TView<Real, 1, D> dihedrals, int n_rotameric_chi, int dihe_offset)
    ->int {
  // Input dihedral value must be in the range [-pi,+pi)
  // three bins: g+ (  0--120) = 0
  //             t  (120--240) = 1
  //             g- (240--360) = 2
  int rotamer_ind = 0;

  for (int ii = 0; ii < n_rotameric_chi; ++ii) {
    Real iidihe = dihedrals[dihe_offset + ii];
    rotamer_ind *= 3;
    if (iidihe < -120.0_2rad) {
      // dihedral between -180 and -120: trans
      rotamer_ind += 1;
    } else if (iidihe < 0) {
      // dihedral between -120 and 0: g-
      rotamer_ind += 2;
    } else if (iidihe < 120.0_2rad) {
      // dihedral between 0 and +120: g+
      rotamer_ind += 0;
    } else {
      // dihedral between +120 and +180: trans
      rotamer_ind += 1;
    }
  }
  return rotamer_ind;
}

// will also template later on the number of interpolated backbone dihedrals;
// for now this number is 2. Same logic for interpolating
// rotameric means as for interpolating rotameric stdev.
template <typename Real, typename Int, tmol::Device D>
def interpolate_rotameric_table(
    TView<TView<Real, 2, D>, 1, D> rotameric_value_tables_view,
    TView<Vec<Real, 2>, 1, D> rotameric_bb_start,
    TView<Vec<Real, 2>, 1, D> rotameric_bb_step,
    TView<Vec<Real, 2>, 1, D> rotameric_bb_periodicity,
    int residue_ind,
    int residue_nchi,
    int chi_dihe_for_residue,
    TView<Real, 1, D> dihedrals,
    TView<Int, 1, D> dihedral_offset_for_res,
    TView<Int, 1, D> rottable_set_for_res,
    TView<Int, 1, D> rotmean_table_offset_for_residue,
    TView<Int, 1, D> rottable_assignment)
    ->tuple<Real, Eigen::Matrix<Real, 2, 1> > {
  Int rottableind_for_set = rottable_assignment[residue_ind];

  Int tableind = rotmean_table_offset_for_residue[residue_ind]
                 + residue_nchi * rottableind_for_set + chi_dihe_for_residue;
  Int res_dihedral_offset = dihedral_offset_for_res[residue_ind];

  Int table_set = rottable_set_for_res[residue_ind];

  Eigen::Matrix<Real, 2, 1> bbdihe, bbstep;
  for (int ii = 0; ii < 2; ++ii) {
    Real wrap_iidihe =
        dihedrals[res_dihedral_offset + ii] - rotameric_bb_start[table_set][ii];
    while (wrap_iidihe < 0) {
      wrap_iidihe += 2 * M_PI;
    }
    Real ii_period = rotameric_bb_periodicity[table_set][ii];
    while (wrap_iidihe > ii_period) {
      wrap_iidihe -= ii_period;
    }

    bbstep[ii] = rotameric_bb_step[table_set][ii];
    bbdihe[ii] = wrap_iidihe / bbstep[ii];
  }

  Real V;
  Eigen::Matrix<Real, 2, 1> dVdbb;
  tie(V, dVdbb) =
      tmol::numeric::bspline::ndspline<2, 3, D, Real, Int>::interpolate(
          rotameric_value_tables_view[tableind], bbdihe);
  for (int ii = 0; ii < 2; ++ii) {
    dVdbb[ii] /= bbstep[ii];
  }
  return {V, dVdbb};
}

template <typename Real, typename Int, tmol::Device D>
def chi_deviation_penalty(
    TView<TView<Real, 2, D>, 1, D> rotameric_mean_tables_view,
    TView<TView<Real, 2, D>, 1, D> rotameric_sdev_tables_view,
    TView<Vec<Real, 2>, 1, D> const& rotameric_bb_start,
    TView<Vec<Real, 2>, 1, D> const& rotameric_bb_step,
    TView<Vec<Real, 2>, 1, D> const& rotameric_bb_periodicity,
    int residue_ind,
    int residue_nchi,
    int chi_dihe_for_residue,
    TView<Real, 1, D> dihedrals,
    TView<Int, 1, D> dihedral_offset_for_res,
    TView<Int, 1, D> rottable_set_for_res,
    TView<Int, 1, D> rotmean_table_set_offset,
    TView<Int, 1, D> rottable_assignment)
    ->tuple<Real, Real, Eigen::Matrix<Real, 2, 1> > {
  Real mean, sdev;
  Eigen::Matrix<Real, 2, 1> dmean_dbb, dsdev_dbb;
  tie(mean, dmean_dbb) = interpolate_rotameric_table(
      rotameric_mean_tables_view,
      rotameric_bb_start,
      rotameric_bb_step,
      rotameric_bb_periodicity,
      residue_ind,
      residue_nchi,
      chi_dihe_for_residue,
      dihedrals,
      dihedral_offset_for_res,
      rottable_set_for_res,
      rotmean_table_set_offset,
      rottable_assignment);

  tie(sdev, dsdev_dbb) = interpolate_rotameric_table(
      rotameric_sdev_tables_view,
      rotameric_bb_start,
      rotameric_bb_step,
      rotameric_bb_periodicity,
      residue_ind,
      residue_nchi,
      chi_dihe_for_residue,
      dihedrals,
      dihedral_offset_for_res,
      rottable_set_for_res,
      rotmean_table_set_offset,
      rottable_assignment);

  Int chi_index =
      dihedral_offset_for_res[residue_ind] + chi_dihe_for_residue + 2;
  Real const chi = dihedrals[chi_index];
  Real const chi_dev =
      (chi < -120.0_2rad ? chi + Real(2 * M_PI) - mean : chi - mean);

  // Now calculate d penalty / dbb

  // From Rosetta3:
  // Backbone derivatives for chi-dev penalty.
  // Xmean and sd both depend on phi and psi.
  // Let: f = (X_i-Xmean_i)**2
  // Let: g = 2 sd_i**2
  // Then, chidevpen = f/g
  // and, dchidevpen = (f'g - fg')/(gg)

  Real const f = chi_dev * chi_dev;
  Real const fprime = -2 * chi_dev;
  Real const g = 2 * sdev * sdev;
  Real const invg = 1 / g;
  Real const gprime = 4 * sdev;
  Real const invgg = invg * invg;

  Real const deviation_penalty = f * invg;
  Real const dpen_dchi = 2 * (chi_dev)*invg;

  Eigen::Matrix<Real, 2, 1> ddev_dbb;
  for (Int ii = 0; ii < 2; ++ii) {
    ddev_dbb[ii] =
        (g * fprime * dmean_dbb[ii] - f * gprime * dsdev_dbb[ii]) * invgg;
  }

  return {deviation_penalty, dpen_dchi, ddev_dbb};
}

template <typename Real, typename Int, tmol::Device D>
def rotameric_chi_probability(
    TView<TView<Real, 2, D>, 1, D> rotameric_neglnprob_tables_view,
    TView<Vec<Real, 2>, 1, D> rotameric_bb_start,
    TView<Vec<Real, 2>, 1, D> rotameric_bb_step,
    TView<Vec<Real, 2>, 1, D> rotameric_bb_periodicity,
    TView<Int, 1, D> prob_table_offset_for_rotresidue,
    int residue_ind,
    int rotresidue_ind,
    TView<Real, 1, D> dihedrals,
    TView<Int, 1, D> dihedral_offset_for_res,
    TView<Int, 1, D> rottable_set_for_res,
    TView<Int, 1, D> rotameric_rottable_assignment)
    ->tuple<Real, Eigen::Matrix<Real, 2, 1> > {
  Eigen::Matrix<Real, 2, 1> bbdihe, bbstep;

  Int const table_set = rottable_set_for_res[residue_ind];
  Int const res_rottable = prob_table_offset_for_rotresidue[rotresidue_ind]
                           + rotameric_rottable_assignment[residue_ind];
  Int const res_dihedral_offset = dihedral_offset_for_res[residue_ind];

  for (Int ii = 0; ii < 2; ++ii) {
    Real wrap_iidihe =
        dihedrals[res_dihedral_offset + ii] - rotameric_bb_start[table_set][ii];
    while (wrap_iidihe < 0) {
      wrap_iidihe += 2 * M_PI;
    }
    Real ii_period = rotameric_bb_periodicity[table_set][ii];
    while (wrap_iidihe > ii_period) {
      wrap_iidihe -= ii_period;
    }

    bbstep[ii] = rotameric_bb_step[table_set][ii];
    bbdihe[ii] = wrap_iidihe / bbstep[ii];
  }
  Real V;
  Eigen::Matrix<Real, 2, 1> dVdbb;
  tie(V, dVdbb) =
      tmol::numeric::bspline::ndspline<2, 3, D, Real, Int>::interpolate(
          rotameric_neglnprob_tables_view[res_rottable], bbdihe);
  for (int ii = 0; ii < 2; ++ii) {
    dVdbb[ii] /= bbstep[ii];
  }
  return {V, dVdbb};
}

template <typename Real, typename Int, tmol::Device D>
def semirotameric_energy(
    TView<TView<Real, 3, D>, 1, D> semirotameric_tables_view,
    TView<Vec<Real, 3>, 1, D> semirot_start,
    TView<Vec<Real, 3>, 1, D> semirot_step,
    TView<Vec<Real, 3>, 1, D> semirot_periodicity,
    TView<Int, 1, D> dihedral_offset_for_res,
    TView<Real, 1, D> dihedrals,
    TView<Int, 1, D> semirotameric_rottable_assignment,
    Int resid,
    Int semirot_dihedral_index,
    Int semirot_table_offset,
    Int semirot_table_set)
    ->tuple<Real, Eigen::Matrix<Real, 3, 1> > {
  Eigen::Matrix<Real, 3, 1> dihe;
  Eigen::Matrix<Real, 3, 1> temp_dihe_deg;
  Eigen::Matrix<Real, 3, 1> temp_orig_dihe_deg;
  Eigen::Matrix<Real, 3, 1> dihe_step;
  Eigen::Matrix<Real, 3, 1> temp_dihe_period;
  Eigen::Matrix<Real, 3, 1> temp_dihe_start;

  Int res_dihedral_offset = dihedral_offset_for_res[resid];
  Int table_ind =
      semirotameric_rottable_assignment[resid] + semirot_table_offset;
  for (int ii = 0; ii < 3; ++ii) {
    int ii_dihe_ind =
        ii == 2 ? semirot_dihedral_index : (res_dihedral_offset + ii);
    Real wrap_iidihe =
        dihedrals[ii_dihe_ind] - semirot_start[semirot_table_set][ii];
    temp_dihe_start(ii) = semirot_start[semirot_table_set][ii] * 180 / M_PI;
    temp_orig_dihe_deg(ii) = dihedrals[ii_dihe_ind] * 180 / M_PI;
    while (wrap_iidihe < 0) {
      wrap_iidihe += 2 * M_PI;
    }
    Real ii_period = semirot_periodicity[semirot_table_set][ii];
    while (wrap_iidihe > ii_period) {
      wrap_iidihe -= ii_period;
    }
    temp_dihe_deg(ii) = wrap_iidihe * 180 / M_PI;
    dihe_step[ii] = semirot_step[semirot_table_set][ii];
    dihe[ii] = wrap_iidihe / dihe_step[ii];
    temp_dihe_period(ii) = ii_period;
  }

  Real V;
  Eigen::Matrix<Real, 3, 1> dV_ddihe;
  tie(V, dV_ddihe) =
      tmol::numeric::bspline::ndspline<3, 3, D, Real, Int>::interpolate(
          semirotameric_tables_view[table_ind], dihe);
  for (int ii = 0; ii < 3; ++ii) {
    dV_ddihe[ii] /= dihe_step[ii];
  }

  return {V, dV_ddihe};
}

#undef def
}  // namespace potentials
}  // namespace dunbrack
}  // namespace score
}  // namespace tmol