#pragma once

#include <Eigen/Core>
#include <tuple>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorCollection.h>
#include <tmol/utility/tensor/TensorPack.h>

#include <ATen/Tensor.h>

namespace tmol {
namespace pack {
namespace rotamer {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

#define CoordQuad Eigen::Matrix<Real, 4, 3>

template <tmol::Device D, typename Real, typename Int>
struct DunbrackChiSampler {
  static auto f(
      TView<Vec<Real, 3>, 1, D> coords,
      TView<int64_t, 1, D> res_coord_start_ind,

      TView<Vec<int64_t, 2>, 1, D> rottable_set_for_res, // pos 0: seqpos, pos 1: table ind

      TView<Int, 1, D> ndihe_for_res,               // nres x 1
      TView<Int, 1, D> dihedral_offset_for_res,     // nres x 1
      TView<Vec<Int, 4>, 1, D> dihedral_atom_inds,  // ndihe x 4
      
      
      TView<Real, 3, D> rotameric_prob_tables,
      TView<Real, 3, D> rotameric_neglnprob_tables,
      TView<Vec<int64_t, 2>, 1, D> rotprob_table_sizes,
      TView<Vec<int64_t, 2>, 1, D> rotprob_table_strides,
      TView<Real, 3, D> rotameric_mean_tables,
      TView<Real, 3, D> rotameric_sdev_tables,
      TView<Vec<int64_t, 2>, 1, D> rotmean_table_sizes,
      TView<Vec<int64_t, 2>, 1, D> rotmean_table_strides,
      TView<Vec<Real, 2>, 1, D> rotameric_bb_start,        // ntable-set entries
      TView<Vec<Real, 2>, 1, D> rotameric_bb_step,         // ntable-set entries
      TView<Vec<Real, 2>, 1, D> rotameric_bb_periodicity,  // ntable-set entries
      TView<Real, 4, D> semirotameric_tables,              // n-semirot-tabset
      TView<Vec<int64_t, 3>, 1, D> semirot_table_sizes,    // n-semirot-tabset
      TView<Vec<int64_t, 3>, 1, D> semirot_table_strides,  // n-semirot-tabset
      TView<Vec<Real, 3>, 1, D> semirot_start,             // n-semirot-tabset
      TView<Vec<Real, 3>, 1, D> semirot_step,              // n-semirot-tabset
      TView<Vec<Real, 3>, 1, D> semirot_periodicity,       // n-semirot-tabset
      TView<Int, 1, D> rotameric_rotind2tableind,
      TView<Int, 1, D> semirotameric_rotind2tableind,


      TView<Int, 1, D> rottable_set_for_res,              // nres x 1
      TView<Int, 1, D> nchi_for_res,                      // nres x 1
      TView<Int, 1, D> nrotameric_chi_for_res,            // nres x 1
      TView<Int, 1, D> rotres2resid,                      // nres x 1
      TView<Int, 1, D> prob_table_offset_for_rotresidue,  // n-rotameric-res x 1
      TView<Int, 1, D> rotind2tableind_offset_for_res,    // n-res x 1

      TView<Int, 1, D> rotmean_table_offset_for_residue,  // n-res x 1

      TView<Int, 2, D> rotameric_chi_desc,  // n-rotameric-chi x 2
      // rotchi_desc[:,0] == residue index for this chi
      // rotchi_desc[:,1] == chi_dihedral_index for res

      TView<Int, 2, D> semirotameric_chi_desc,  // n-semirotameric-residues x 4
      // semirotchi_desc[:,0] == residue index
      // semirotchi_desc[:,1] == semirotchi_dihedral_index res
      // semirotchi_desc[:,2] == semirot_table_offset
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
  {
    // construct the list of chi for the rotamers that should be built
    // in 7 stages.
    // 1. State which AAs at which positions
    // 2. Determine max # pre-expansion rotamers per AA per position.
    // 3. Exclusive cumsum for offsets on # pre-expansion rotamers
    // 4. Interpolate probabilities for each rotamer, using phi/psi
    // 5. Exclusive segmented scan of post-expansion rot counts
    // 6. allocate space for rotamer chi
    // 7. Interpolate chi values and record them


    Int const nres(res_coord_start_ind.size(0));
    Int const n_rtypes_total(rottable_set_for_res.size(0));

    auto n_possible_rotamers_per_restype_tp = TPack<Int, 1, D>::zeros(n_rtypes_total);
    auto n_possible_rotamers_per_restype = n_possible_rotamers_per_restype_tp.view;

    auto determine_n_possible_rots = [](int i) {
      Int rottable_set = rottable_set_for_res[i][1];
      n_possible_rotamers_per_restype[i] = nrots_for_tableset[rottable_set];
    };

    for (int i = 0; i < n_rtypes_total; ++i) {
      determine_n_possible_rots(i);
    }

    auto rotamers_for_rt_offset_tp = TPack<Int, 1, D>::zeros(n_rtypes_total);
    auto rotamers_for_rt_offset = rotamers_for_rt_offset_tp.view;

    for (int i = 1; i < n_rtypes_total; ++i) {
      rotamers_for_rt_offset[i] = rotamers_for_rt_offset[i-1] + 
	n_possible_rotamers_per_restype[i-1];
    }

    Int const n_possible_rotamers = rotamers_for_rt_offset[n_rtypes_total-1] +
      n_possible_rotamers_per_restype[n_rtypes_total-1];

    auto rotamer_probability_tp = TPack<Real, 1, D>::empty(n_possible_rotamers);
    auto rotamer_probability = rotamer_probability_tp.view;


    auto backbone_dihedrals_tp = TPack<Real, 1, D>::empty(nres*2);
    auto backbone_dihedrals = backbone_dihedrals_tp.view;

    auto compute_backbone_dihedrals = [] (int i) {
      Int at0 = dihedral_atom_inds[i][0];
      Int at1 = dihedral_atom_inds[i][1];
      Int at2 = dihedral_atom_inds[i][2];
      Int at3 = dihedral_atom_inds[i][3];
      auto dihe = dihedral_angle<Real>::V(
	coords[at0], coords[at1], coords[at2], coords[at3]);
      backbone_dihedrals[i] = dihe;
    }

    for (int i = 0; i < dihedrals.size(0); ++i) {
      compute_backbone_dihedrals(i);
    }

    auto residue_for_possible_rotamer_tp = TPack<Int, 1, D>::zeros(n_possible_rotamers);
    auto residue_for_possible_rotamer = residue_for_possible_rotamer_tp.view;
    auto residue_for_possible_rotamer_boundaries_tp = TPack<Int, 1, D>::zeros(n_possible_rotamers);
    auto residue_for_possible_rotamer_boundaries = residue_for_possible_rotamer_boundaries.view;

    auto mark_boundary_beginnings = [](int i){
      residue_for_possible_rotamer_boundaries[
	rotamers_for_rt_offset[i]] = res_for_restype[i];
    }
    for (int i = 0; i < n_rtypes_total; ++i){
      mark_boundary_beginnings(i);
    }

    // Now segmented scan on "max" to get the residue index for each possible
    // rotamer
    for (int i = 1; i < n_possible_rotamers; ++i ) {
      if (!residue_for_possible_rotamer_boundaries[i]) {
	residue_for_possible_rotamers[i] = max(
	  residue_for_possible_rotamers[i],
	  residue_for_possible_rotamers[i-1]);
      }
    }


    auto calculate_rotamer_probabilities = []( int i ) {
      // Compute the probability of the ith possible rotamer
      

    }

  }
};

#undef CoordQuad

}  // namespace rotamer
}  // namespace pack
}  // namespace tmol
