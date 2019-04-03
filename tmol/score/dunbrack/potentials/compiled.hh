#pragma once

#include <Eigen/Core>
#include <tuple>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorCollection.h>
#include <tmol/utility/tensor/TensorPack.h>

#include <ATen/Tensor.h>

namespace tmol {
namespace score {
namespace dunbrack {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

#define CoordQuad Eigen::Matrix<Real, 4, 3>

template <tmol::Device D, typename Real, typename Int>
struct DunbrackDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, D> coords,

      TCollection<Real, 3, D> rotameric_prob_tables,
      TCollection<Real, 3, D> rotameric_neglnprob_tables,
      TCollection<Real, 3, D> rotameric_mean_tables,
      TCollection<Real, 3, D> rotameric_sdev_tables,
      TView<Vec<Real, 2>, 1, D> rotameric_bb_start,        // ntable-set entries
      TView<Vec<Real, 2>, 1, D> rotameric_bb_step,         // ntable-set entries
      TView<Vec<Real, 2>, 1, D> rotameric_bb_periodicity,  // ntable-set entries
      TCollection<Real, 4, D> semirotameric_tables,        // n-semirot-tabset
      TView<Vec<Real, 3>, 1, D> semirot_start,             // n-semirot-tabset
      TView<Vec<Real, 3>, 1, D> semirot_step,              // n-semirot-tabset
      TView<Vec<Real, 3>, 1, D> semirot_periodicity,       // n-semirot-tabset
      TView<Int, 1, D> rotameric_rotind2tableind,
      TView<Int, 1, D> semirotameric_rotind2tableind,

      TView<Int, 1, D> ndihe_for_res,               // nres x 1
      TView<Int, 1, D> dihedral_offset_for_res,     // nres x 1
      TView<Vec<Int, 4>, 1, D> dihedral_atom_inds,  // ndihe x 4

      TView<Int, 1, D> rottable_set_for_res,    // nres x 1
      TView<Int, 1, D> nchi_for_res,            // nres x 1
      TView<Int, 1, D> nrotameric_chi_for_res,  // nres x 1
      TView<Int, 1, D> rotres2resid,            // nres x 1
      // TView<Int, 1, D> prob_table_offset_for_rotresidue,  // n-rotameric-res
      // x 1
      TView<Int, 1, D> rotind2tableind_offset_for_res,  // n-res x 1

      // TView<Int, 1, D> rotmean_table_offset_for_residue,  // n-res x 1

      TView<Int, 2, D> rotameric_chi_desc,  // n-rotameric-chi x 2
      // rotchi_desc[:,0] == residue index for this chi
      // rotchi_desc[:,1] == chi_dihedral_index for res

      TView<Int, 2, D> semirotameric_chi_desc,  // n-semirotameric-residues x 4
      // semirotchi_desc[:,0] == residue index
      // semirotchi_desc[:,1] == semirotchi_dihedral_index res
      // semirotchi_desc[:,2] == semirot_table_offset -- axed!!!
      // semirotchi_desc[:,3] == semirot_table_set (e.g. 0-7)

      // scratch space, perhaps does not belong as an input parameter?
      TView<Real, 1, D> dihedrals,                        // ndihe x 1
      TView<Eigen::Matrix<Real, 4, 3>, 1, D> ddihe_dxyz,  // ndihe x 3
      // TView<Real, 1, D> rotchi_devpen,                    // n-rotameric-chi
      // x 1 TView<Real, 2, D> ddevpen_dbb,  // Where d chimean/d dbbdihe is
      //                                // stored, nscdihe x 2
      TView<Int, 1, D> rotameric_rottable_assignment,     // nres x 1
      TView<Int, 1, D> semirotameric_rottable_assignment  // nres x 1
      )
      -> std::tuple<
          TPack<Real, 1, D>,        // -ln(prob_rotameric)
          TPack<CoordQuad, 2, D>,   // d(-ln(prob_rotameric)) / dbb atoms
          TPack<Real, 1, D>,        // Erotameric_chi_devpen
          TPack<CoordQuad, 2, D>,   // ddevpen_dtor
          TPack<Real, 1, D>,        // -ln(prob_nonrotameric)
          TPack<CoordQuad, 2, D> >  // d(-ln(prob_nonrotameric)) / dtor atoms
      ;

  static auto df(
      TView<Vec<Real, 3>, 1, D> coords,
      TView<Real, 1, D> dE_drotnlp,
      TView<CoordQuad, 2, D> drot_nlp_dbb_xyz,
      TView<Real, 1, D> dE_ddevpen,
      TView<CoordQuad, 2, D> ddevpen_dtor_xyz,
      TView<Real, 1, D> dE_dnonrotnlp,
      TView<CoordQuad, 2, D> dnonrot_nlp_dtor_xyz,
      TView<Int, 1, D> dihedral_offset_for_res,     // nres x 1
      TView<Vec<Int, 4>, 1, D> dihedral_atom_inds,  // ndihe x 4
      TView<Int, 1, D> rotres2resid,                // nres x 1
      TView<Int, 2, D> rotameric_chi_desc,          // n-rotameric-chi x 2
      // rotchi_desc[:,0] == residue index for this chi
      // rotchi_desc[:,1] == chi_dihedral_index for res

      TView<Int, 2, D> semirotameric_chi_desc  // n-semirotameric-residues x 4
      // semirotchi_desc[:,0] == residue index
      // semirotchi_desc[:,1] == semirotchi_dihedral_index res
      // semirotchi_desc[:,2] == semirot_table_offset
      // semirotchi_desc[:,3] == semirot_table_set (e.g. 0-7)

      ) -> TPack<Vec<Real, 3>, 1, D>;
};

#undef CoordQuad

}  // namespace potentials
}  // namespace dunbrack
}  // namespace score
}  // namespace tmol
