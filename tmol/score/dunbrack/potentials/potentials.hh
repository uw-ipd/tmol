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
    TView<Eigen::Matrix<Real, 2, 1>, 1, D> coordinates, Int4 indices)
    ->tuple<Real, CoordQuad> {
  return = dihedral_angle<Real>::V_dV(
             coordinates[indices[0]],
             coordinates[indices[1]],
             coordinates[indices[2]],
             coordinates[indices[3]])
}

template <typename Real, typename Int, typename D>
def classify_rotamer(
    TView<Real, 1, D> dihedrals, Int n_rotameric_chi, Int offset)
    ->Int {
  Int rotamer_ind = 0;
  for (int ii = 0; ii < n_rotameric_chi; ++ii) {
    Real iidihe = dihedrals[offset + ii];
    rotamer_ind *= 3;
    if (iidihe < -120) {
      // dihedral between -180 and -120: trans
      rotamer_ind += 1;
    } else if (iidihe < 0) {
      // dihedral between -120 and 0: g-
      rotamer_ind += 2;
    } else if (iidihe < 120) {
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
    TCollection<Real, 2, D> rotameric_value_tables,
    Int residue_ind,
    Int residue_nchi,
    Int chi_dihe_for_residue,
    TView<Real, 1, D> dihedrals,
    TView<Real, 1, D> dihedral_offset_for_res,
    TView<Int, 1, D> rotameric_table_set_offset,
    TView<Int, 1, D> rotameric_assignment)
    ->tuple<Real, Eigen::Matrix<Real, 2, 1> > {
  Int tableind = rotameric_table_set_offset[residue_ind]
                 + residue_nchi * rotameric_assignment[residue_ind]
                 + chi_dihe_for_residue;
  Int res_dihedral_offset = dihedral_offset_for_res[residue_ind];

  Eigen::Matrix<Real, 2, 1> bbdihe;
  for (Int ii = 0; ii < 2; ++ii) {
    bbdihe[ii] = dihedrals[res_dihedral_offset + ii];
  }

  return ndspline<2, 3, Real, Int>::interpolate(
      rotameric_value_tables[tableind], bbdihe);
}

template <typename Real, typename Int, typename D>
def chi_deviation_penalty(
    TCollection<Real, 2, D> rotameric_mean_tables,
    TCollection<Real, 2, D> rotameric_sdev_tables,
    int residue_ind,
    int residue_nchi,
    Int chi_dihe_for_residue,
    TView<Real, 1, D> dihedrals,
    TView<Real, 1, D> dihedral_offset_for_res,
    TView<Int, 1, D> rotameric_table_set_offset,
    TView<Int, 1, D> rotameric_assignment, )
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
      rotameric_table_set_offset,
      rotameric_assignment);

  std::tie(sdev, dsdev_dbb) = interpolate_rotameric_table(
      rotameric_sdev_tables,
      residue_ind,
      residue_nchi,
      chi_dihe_for_residue,
      dihedrals,
      dihedral_offset_for_res,
      rotameric_table_set_offset,
      rotameric_assignment);

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
    TView<Int, 1, D> prob_table_offset_for_residue,
    int residue_ind,
    TView<Real, 1, D> dihedrals,
    TView<Real, 1, D> dihedral_offset_for_res,
    TView<Int, 1, D> rotameric_assignment)
    ->std::tuple<Real, Eigen::Matrix<Real, 2, 1> > {
  Eigen::Matrix<Real, 2, 1> bbdihe;

  Int const res_rot = rotameric_assignment[residue_ind];
  Int const res_offset = dihedral_offset_for_res[residue_ind];
  for (Int ii = 0; ii < 2; ++ii) {
    bbdihe[ii] = dihedrals[res_offset + ii];
  }

  return ndspline<2, 3, Real, Int>::interpolate(
      rotameric_prob_tables[res_rot], bbdihe);
}

template <typename Real, typename Int, class D>
def semirotameric_energy(
    TCollection<Real, 3, D> semirotameric_tables,
    TView<Int, 1, D> semirot_table_ofset,
    TView<Int, 1, D> dihedral_offset_for_res,
    TView<Real, 1, D> dihedrals,
    TView<Int, 1, D> ndihe_for_res,
    TView<Int, 1, D> residue_for_semirotres,
    TView<Int, 1, D> rotameric_assignment,
    Int semirotres)
    ->std::tuple<Real, Eigen::Matrix<Real, 3, 1> > {
  Eigen::Matrix<Real, 3, 1> dihe;
  Int resid = residue_for_semirotres[semirotres];
  Int dihedral_start = dihedral_offset_for_res[resid];
  Int table_ind =
      rotameric_assignments[resid] + semirot_table_offset[semirotres];
  for (int ii = 0; ii < 2; ++ii) {
    dihe[ii] = dihedrals[dihedral_start + ii];
  }
  dihe[2] = dihedrals[dihedral_start + ndihe_for_res[resid]];
  return ndspline<3, 3, Real, Int>::interpolate(
      semirotameric_tables[table_ind], dihe);
}

#undef Real2
#undef Real3

#undef def
}  // namespace potentials
}  // namespace rama
}  // namespace score
}  // namespace tmol
