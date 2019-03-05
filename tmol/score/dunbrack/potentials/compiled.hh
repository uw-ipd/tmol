#pragma once

#include <Eigen/Core>
#include <tuple>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorCollection.h>
#include <tmol/utility/tensor/TensorPack.h>

#include <ATen/Tensor.h>

namespace tmol {
namespace score {
namespace rama {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <tmol::Device D, typename Real, typename Int>
struct DunbrackDispatch {
  static auto f(
      TCollection<Real, 2, D> rotameric_prob_tables,
      TCollection<Real, 2, D> rotameric_mean_tables,
      TCollection<Real, 2, D> rotameric_sdev_tables,
      TView<Vec<Real, 2>, 1, D> rotameric_bb_start,  // ntable-set entries
      TView<Vec<Real, 2>, 1, D> rotameric_bb_step,   // ntable-set entries
      TCollection<Real, 3, D> semirotameric_tables,
      TView<Vec<Real, 3>, 1, D> semirot_start,
      TView<Vec<Real, 3>, 1, D> semirot_step,
      TCollection<Int, 1, D> rotind2tableind,

      TView<Vec<Real, 3>, 1, D> coords,

      TView<Int, 1, D> ndihe_for_res,               // nres x 1
      TView<Int, 1, D> dihedral_offset_for_res,     // nres x 1
      TView<Vec<Int, 4>, 1, D> dihedral_atom_inds,  // ndihe x 4

      TView<Int, 1, D> rottable_set_for_res,            // nres x 1
      TView<Int, 1, D> nrotameric_chi_for_res,          // nres x 1
      TView<Int, 1, D> prob_table_offset_for_residue,   // n-rotameric-res x 1
      TView<Int, 1, D> rotind2tableind_offset_for_res,  // n-rotameric-res x 1

      TView<Int, 1, D> rotmean_table_offset_for_residue,  // n-res x 1

      TView<Int, 2, D> rotameric_chi_desc,  // n-rotameric-chi x 2
      // rotchi_desc[:,0] == residue index for this chi
      // rotchi_desc[:,1] == chi_dihedral_index for res

      TView<Int, 2, D> semirotameric_chi_desc,  // n-semirotameric-residues x 3
      // semirotchi_desc[:,0] == residue index
      // semirotchi_desc[:,1] == semirotchi_dihedral_index res
      // semirotchi_desc[:,2] == semirot_table_offset

      // scratch space, perhaps does not belong as an input parameter?
      TView<Real, 1, D> dihedrals,              // ndihe x 1
      TView<Real, 2, D> ddihe_dxyz,             // ndihe x 3
      TView<Real, 1, D> dihedral_dE_ddihe,      // ndihe x 1
      TView<Real, 2, D> dihedral_dmean_ddihe,   // Where d chimean/d dbbdihe is
                                                // stored, nscdihe x 2
      TView<Real, 2, D> dihedral_dsdev_ddihe,   // Where d chisdev/d dbbdihe is
                                                // stored, nscdihe x 2
      S TView<Int, 1, D> rotameric_assignment,  // nres x 1
      ) -> TPack<Real, 1, D>;
};

}  // namespace potentials
}  // namespace rama
}  // namespace score
}  // namespace tmol
