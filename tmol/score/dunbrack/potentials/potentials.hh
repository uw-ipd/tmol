#pragma once

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
namespace rama {
namespace potentials {

using namespace tmol::score::common;

#define def auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

#define CoordQuad Eigen::Matrix<Real, 4, 3>
#define Real3 Vec<Real, 3>
#define Real2 Vec<Real, 2>
#define Int4 Vec<Int, 4>

template <typename Real, typename Int, typename D>
def measure_dihedrals_V_dV(
    TView<Eigen::Matrix<Real, 2, 1>, 1, D> coordinates,
    Int i,
    TView<Vec<Int, 4>, 1, D> dihedral_atom_inds,
    TView<Real, 1, D> dihedrals,
    TView<Eigen::Matrix<Real, 4, 3>, 1, D> ddihe_dxyz)
    ->tuple<Real, CoordQuad> {
  auto dihe = dihedral_angle<Real>::V_dV(
      coordinates[dihedral_atom_inds[i][0]],
      coordinates[dihedral_atom_inds[i][1]],
      coordinates[dihedral_atom_inds[i][2]],
      coordinates[dihedral_atom_inds[i][3]]);
  dihedrals[i] = dihe.V;
  ddihe_dxyz[i].row(0) = dihe.dV_dI;
  ddihe_dxyz[i].row(1) = dihe.dV_dJ;
  ddihe_dxyz[i].row(2) = dihe.dV_dK;
  ddihe_dxyz[i].row(3) = dihe.dV_dL;
}

template <typename Real, typename Int, typename D>
def classify_rotamer(
    TView<Real, 1, D> dihedrals, Int n_rotameric_chi, Int dihe_offset)
    ->Int {
  // Input dihedral value must be in the range [-pi,+pi)
  Int rotamer_ind = 0;
  Real const deg2rad = M_PI / 180;
  for (int ii = 0; ii < n_rotameric_chi; ++ii) {
    Real iidihe = dihedrals[dihe_offset + ii];
    rotamer_ind *= 3;
    if (iidihe < -120 * deg2rad) {
      // dihedral between -180 and -120: trans
      rotamer_ind += 1;
    } else if (iidihe < 0) {
      // dihedral between -120 and 0: g-
      rotamer_ind += 2;
    } else if (iidihe < 120 * deg2rad) {
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
template <typename Real, typename Int, typename D>
def interpolate_rotameric_table(
    TCollection<Real, 2, D>& rotameric_value_tables,
    TView<Vec<Real, 2, D>, 1, D>& rotameric_bb_start,
    TView<Vec<Real, 2, D>, 1, D>& rotameric_bb_step,
    TView<Vec<Real, 2, D>, 1, D>& rotameric_bb_periodicity,
    Int residue_ind,
    Int residue_nchi,
    Int chi_dihe_for_residue,
    TView<Real, 1, D>& dihedrals,
    TView<Real, 1, D>& dihedral_offset_for_res,
    TView<Real, 1, D>& rottable_set_for_res,
    TView<Int, 1, D>& rotmean_table_offset_for_residue,
    TView<Int, 1, D> rottable_assignment)
    ->tuple<Real, Eigen::Matrix<Real, 2, 1> > {
  Int rottableind_for_set = rottable_assignment[residue_ind];

  Int tableind = rotmean_table_offset_for_res[residue_ind]
                 + residue_nchi * rottableind_for_set + chi_dihe_for_residue;
  Int res_dihedral_offset = dihedral_offset_for_res[residue_ind];

  Int table_set = rottable_set_for_res[residue_ind];

  Eigen::Matrix<Real, 2, 1> bbdihe, bbstep;
  for (int ii = 0; ii < 2; ++ii) {
    Real wrap_iidihe =
        dihedrals[res_dihedral_offset + ii] - rotameric_bb_start[table_set][ii];
    while (wrap_iidihe < 0) {
      wrap_iidihe += 2 * M_Pi;
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
  std::tie(V, dVdbb) = ndspline<2, 3, Real, Int>::interpolate(
      rotameric_value_tables[tableind], bbdihe);
  for (Int ii = 0; ii < 2; ++ii) {
    dVdbb[ii] /= bbstep[ii];
  }
  return {V, dVdbb};
}

template <typename Real, typename Int, typename D>
def chi_deviation_penalty(
    TCollection<Real, 2, D> rotameric_mean_tables,
    TCollection<Real, 2, D> rotameric_sdev_tables,
    TView<Vec<Real, 2, D>, 1, D>& rotameric_bb_start,
    TView<Vec<Real, 2, D>, 1, D>& rotameric_bb_step,
    int residue_ind,
    int residue_nchi,
    Int chi_dihe_for_residue,
    TView<Real, 1, D> dihedrals,
    TView<Real, 1, D> dihedral_offset_for_res,
    TView<Real, 1, D> rottable_set_for_res,
    TView<Int, 1, D> rotmean_table_set_offset,
    TView<Int, 1, D> rottable_assignment)
    ->std::tuple<Real, Real, Eigen::Matrix<Real, 2, 1> > {
  Real mean, sdev;
  Eigen::Matrix<Real, 2, 1> dmean_dbb, dsdev_dbb;
  std::tie(mean, dmean_dbb) = interpolate_rotameric_table(
      rotameric_mean_tables,
      residue_ind,
      residue_nchi,
      chi_dihe_for_residue,
      dihedrals,
      dihedral_offset_for_res,
      rotmean_table_set_offset,
      rottable_assignment);

  std::tie(sdev, dsdev_dbb) = interpolate_rotameric_table(
      rotameric_sdev_tables,
      residue_ind,
      residue_nchi,
      chi_dihe_for_residue,
      dihedrals,
      dihedral_offset_for_res,
      rotmean_table_set_offset,
      rotttable_assignment);

  Int chi_index = rotameric_table_set_offset[residue_ind]
                  + residue_nchi * rotameric_assignment[residue_ind]
                  + chi_dihe_for_residue + 2;
  Real const chi = dihedrals[chi_index];
  Real const chi_dev = chi - mean;

  // Now calculate d penalty / dbb

  // From Rosetta3:
  // Backbone derivatives for chi-dev penalty.
  // Xmean and sd both depend on phi and psi.
  // Let: f = (X_i-Xmean_i)**2
  // Let: g = 2 sd_i**2
  // Then, chidevpen = f/g
  // and, dchidevpen = (f'g - fg')/(gg)

  Real const f = chidev * chidev;
  Real const fprime = -2 * chidev;
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

template <typename Real, typename Int, typename D>
def rotameric_chi_probability(
    TCollection<Real, 2, D> rotameric_prob_tables,
    TView<Int, 1, D> prob_table_offset_for_rotresidue,
    int rotresidue_ind,
    int residue_ind,
    TView<Real, 1, D> dihedrals,
    TView<Real, 1, D> dihedral_offset_for_res,
    TView<Int, 1, D> rottable_assignment)
    ->std::tuple<Real, Eigen::Matrix<Real, 2, 1> > {
  Eigen::Matrix<Real, 2, 1> bbdihe;

  Int const res_rottable = rottable_assignment[residue_ind]
                           + prob_table_offset_for_rotresidue[residue_ind];
  Int const res_offset = dihedral_offset_for_res[residue_ind];
  for (Int ii = 0; ii < 2; ++ii) {
    bbdihe[ii] = dihedrals[res_offset + ii];
  }

  return ndspline<2, 3, Real, Int>::interpolate(
      rotameric_prob_tables[res_rottable], bbdihe);
}

template <typename Real, typename Int, class D>
def semirotameric_energy(
    TCollection<Real, 3, D> semirotameric_tables,
    TView<Vec<Real, 3>, 1, D> semirot_start,
    TView<Vec<Real, 3>, 1, D> semirot_stop,
    TView<Vec<Real, 3>, 1, D> semirot_periodicity,
    TView<Int, 1, D> semirot_table_ofset,
    TView<Int, 1, D> dihedral_offset_for_res,
    TView<Real, 1, D> dihedrals,
    TView<Int, 1, D> rottable_assignment,
    Int resid,
    Int semirot_dihedral_index,
    Int semirot_table_offset,
    Int semirot_table_set)
    ->std::tuple<Real, Eigen::Matrix<Real, 3, 1> > {
  Eigen::Matrix<Real, 3, 1> dihe;
  Eigen::Matrix<Real, 3, 1> dihe_step;

  Int dihedral_start = dihedral_offset_for_res[resid];
  Int table_ind = rottable_assignments[resid] + semirot_table_offset;
  for (int ii = 0; ii < 3; ++ii) {
    int ii_dihe_ind =
        res_dihedral_offset + (ii == 2 ? semirot_dihedral_index : ii);
    Real wrap_iidihe = dihedrals[res_dihedral_offset + ii_dihe_ind]
                       - semirot_start[semirot_table_set][ii];
    while (wrap_iidihe < 0) {
      wrap_iidihe += 2 * M_Pi;
    }
    Real ii_period = semirot_periodicity[semirot_table_set][ii];
    while (wrap_iidihe > ii_period) {
      wrap_iidihe -= ii_period;
    }
    dihe_step[ii] = semirot_step[semitor_table_set][ii];
    dihe[ii] = wrap_iidihe / dihe_step[ii];
  }

  Real V;
  Eigen::Matrix<Real, 3, 1> dV_ddihe;
  tie(V, dV_ddihe) = ndspline<3, 3, Real, Int>::interpolate(
      semirotameric_tables[table_ind], dihe);
  for (int ii = 0; ii < 3; ++ii) {
    dV_ddihe[ii] /= semirot_step[semirot_table_set][ii];
  }
  return {V, dV_ddihe};
}

#undef Real2
#undef Real3

#undef def
}  // namespace potentials
}  // namespace rama
}  // namespace score
}  // namespace tmol
