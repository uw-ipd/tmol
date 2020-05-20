#pragma once

#include <Eigen/Core>
#include <tuple>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorCollection.h>
#include <tmol/utility/tensor/TensorPack.h>

#include <tmol/numeric/bspline_compiled/bspline.hh>
#include <tmol/score/common/forall_dispatch.cpu.impl.hh>
#include <tmol/score/common/geom.hh>

#include <tmol/extern/moderngpu/operators.hxx>

#include <ATen/Tensor.h>

namespace tmol {
namespace pack {
namespace rotamer {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <template <tmol::Device> class Dispatch, tmol::Device D, typename Real, typename Int>
struct DunbrackChiSampler {
  static auto f(
      TView<Vec<Real, 3>, 1, D> coords,

      TView<Real, 3, D> rotameric_prob_tables,
      TView<Vec<int64_t, 2>, 1, D> rotprob_table_sizes,
      TView<Vec<int64_t, 2>, 1, D> rotprob_table_strides,
      TView<Real, 3, D> rotameric_mean_tables,
      TView<Real, 3, D> rotameric_sdev_tables,
      TView<Vec<int64_t, 2>, 1, D> rotmean_table_sizes,
      TView<Vec<int64_t, 2>, 1, D> rotmean_table_strides,
      TView<Int, 1, D> rotameric_meansdev_tableset_offsets,
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
      TView<Int, 1, D> all_chi_rotind2tableind,
      TView<Int, 1, D> all_chi_rotind2tableind_offsets,

      TView<int64_t, 1, D> n_rotamers_for_tableset,
      TView<Int, 1, D> n_rotamers_for_tableset_offsets,
      TView<int64_t, 3, D> sorted_rotamer_2_rotamer,
      TView<Int, 1, D> nchi_for_tableset,
      TView<Int, 2, D> rotwells,

      TView<Int, 1, D> ndihe_for_res,               // nres x 1
      TView<Int, 1, D> dihedral_offset_for_res,     // nres x 1
      TView<Vec<Int, 4>, 1, D> dihedral_atom_inds,  // ndihe x 4

      TView<Int, 2, D>
          rottable_set_for_buildable_restype,  // n-buildable-restypes x 2
      TView<Int, 2, D> chi_expansion_for_buildable_restype,
      TView<Real, 3, D> non_dunbrack_expansion_for_buildable_restype,
      TView<Int, 2, D> non_dunbrack_expansion_counts_for_buildable_restype,
      TView<Real, 1, D> prob_cumsum_limit_for_buildable_restype,
      TView<Int, 1, D>
          nchi_for_buildable_restype,  // including hydroxyl chi, e.g.

      // scratch space, perhaps does not belong as an input parameter?
      TView<Real, 1, D> dihedrals  // ndihe x 1
      // ?? TView<Eigen::Matrix<Real, 4, 3>, 1, D> ddihe_dxyz,  // ndihe x 3
      // TView<Real, 1, D> rotchi_devpen,                    // n-rotameric-chi
      // x 1 TView<Real, 2, D> ddevpen_dbb,  // Where d chimean/d dbbdihe is
      //                                // stored, nscdihe x 2
      // ?? TView<Int, 1, D> rotameric_rottable_assignment,     // nres x 1
      // ?? TView<Int, 1, D> semirotameric_rottable_assignment  // nres x 1

      ) -> std::
      tuple<TPack<Int, 1, D>, TPack<Int, 1, D>, TPack<Int, 1, D>, TPack<Real, 2, D> > {
    // std::cout << "Hit compiled.cpu.cpp!" << std::endl;
    auto rval1 = TPack<Real, 1, D>::zeros({5});
    auto rval2 = TPack<Real, 1, D>::zeros({5});
    auto rval1_view = rval1.view;

    auto f = ([=] EIGEN_DEVICE_FUNC(int index) { rval1_view[index] = 4; });
    Dispatch<D>::forall(5, f);

    // construct the list of chi for the rotamers that should be built
    // in 7 stages.
    // 1. State which AAs at which positions
    // 2. Determine max # pre-expansion rotamers per AA per position.
    // 3. Exclusive cumsum for offsets on # pre-expansion rotamers
    // 4. Interpolate probabilities for each rotamer, using phi/psi
    // 5. Exclusive segmented scan of post-expansion rot counts
    // 6. allocate space for rotamer chi
    // 7. Interpolate chi values and record them

    // Named things
    // buildable-residue type:  a combination of a table set to read from and a
    // residue position to build at
    //                          so that if each of N residues read from 1 table,
    //                          there would be N buildable
    //                          residue types and if each residue built all 18
    //                          amino acids represented in
    //                          the Dunbrack library (and separately built both
    //                          HIS and HIS_D) then there
    //                          would be 19N buildable-residue types (the HIS
    //                          and HIS-D residue types would
    //                          be represented separately, meaning the HIS table
    //                          would be read from twice).
    // possible rotamer: a rotamer that might be built for a residue type.
    // Possible rotamers are built
    //                   in decreasing order by probability, and then only the
    //                   top X% of the possible
    //                   rotamers are built. If there are 21N residue types,
    //                   then there are
    //                   sum(i=1, 21N,
    //                   num_possible_rots_for_restable[restable_for_rt[i]] )
    //

    // Input parameter: rottable_set_for_buildable_restype
    //                     array of the dunbrack table set to use for each block
    //                     of residue types; no need to be unique for a single
    //                     residue
    //                     as multiple residue types might share the same table
    //                     set
    //                     (e.g. HIS and HIS_D)
    //                     and the residue index for that residue type
    //                  chi_expansion_for_buildable_restype
    //                     two-D tensor with 0s where a residue type should
    //                     expand
    //                     only use the base rotamer
    //                     and positive integers for different levels of
    //                     expansion
    //                  nchi_for_restype
    //                     some chi not treated by the dunbrack library need
    //                     sampling
    //                     in addition; when writing down the final set of chi
    //                     that
    //                     will be sampled, we have to write down these chi
    //                     values as
    //                     well
    //                  non_dunbrack_expansion for restype
    //                     how should we sample non-dunbrack chi?
    //                     three-D tensor: n-restypes x max-n-chi x max-samples
    //                  non_dunbrack_expansion_counts for restype
    //                     how many elements out of the non_dunbrack_expansion
    //                     array
    //                     are used for each restype for each chi
    //                     2D tensor; 0 for no-expansion
    //                  prob_cumsum_limit_for_restypes
    //                     array of the probability limits for the given residue
    //                     type
    //                     [0..1) which should have been previously calculated
    //                     based on residue burial and residue type

    Int const nres(ndihe_for_res.size(0));
    // The number of buildable residue types across all residues
    Int const n_brt(rottable_set_for_buildable_restype.size(0));

    // I don't remember what I intended for this
    // auto nchi_for_brt_tp = TPack<Int, 1, D>::zeros({n_brt});
    // auto nchi_for_brt = nchi_for_brt_tp.view;

    auto n_possible_rotamers_per_brt_tp = TPack<Int, 1, D>::zeros(n_brt);
    auto n_possible_rotamers_per_brt = n_possible_rotamers_per_brt_tp.view;

    auto determine_n_possible_rots = [=](int brt) {
      Int rottable_set = rottable_set_for_buildable_restype[brt][1];
      n_possible_rotamers_per_brt[brt] = n_rotamers_for_tableset[rottable_set];
    };

    Dispatch<D>::forall(n_brt, determine_n_possible_rots);
    // for (int i = 0; i < n_brt; ++i) {
    //   determine_n_possible_rots(i);
    //   // std::cout << "n possible rots: " << n_possible_rotamers_per_brt[i]
    //   <<
    //   // std::endl;
    // }

    auto possible_rotamer_offset_for_brt_tp = TPack<Int, 1, D>::zeros(n_brt);
    auto possible_rotamer_offset_for_brt =
        possible_rotamer_offset_for_brt_tp.view;

    // Exclusive cumulative sum of n_possible_rotamers_per_restype
    // for (int i = 1; i < n_brt; ++i) {
    //   possible_rotamer_offset_for_brt[i] =
    //       possible_rotamer_offset_for_brt[i - 1]
    //       + n_possible_rotamers_per_brt[i - 1];
    // }
    //
    // // Total number of possible rotamers over all residue types
    // Int const n_possible_rotamers = possible_rotamer_offset_for_brt[n_brt -
    // 1]
    //                                 + n_possible_rotamers_per_brt[n_brt - 1];
    Int const n_possible_rotamers = Dispatch<D>::exclusive_scan_w_final_val(
        n_possible_rotamers_per_brt,
        possible_rotamer_offset_for_brt,
        mgpu::plus_t<Real>());

    // There are some things we need to know about the ith possible rotamer:
    //   1. What buildable_residue type does it come from?
    //   2. What table set does it come from?
    //   3. What residue does it come from?

    auto backbone_dihedrals_tp = TPack<Real, 1, D>::empty(nres * 2);
    auto backbone_dihedrals = backbone_dihedrals_tp.view;

    auto compute_backbone_dihedrals = [=](int i) {
      Int at0 = dihedral_atom_inds[i][0];
      Int at1 = dihedral_atom_inds[i][1];
      Int at2 = dihedral_atom_inds[i][2];
      Int at3 = dihedral_atom_inds[i][3];
      auto dihe = score::common::dihedral_angle<Real>::V(
          coords[at0], coords[at1], coords[at2], coords[at3]);
      backbone_dihedrals[i] = dihe;
    };

    Dispatch<D>::forall(dihedrals.size(0), compute_backbone_dihedrals);

    // for (int i = 0; i < dihedrals.size(0); ++i) {
    //   compute_backbone_dihedrals(i);
    //   // std::cout << "dihedral " << i << " " << 180 / M_PI *
    //   // backbone_dihedrals[i] << std::endl;
    // }

    auto brt_for_possible_rotamer_start_tp =
        TPack<Int, 1, D>::zeros(n_possible_rotamers);
    auto brt_for_possible_rotamer_start =
        brt_for_possible_rotamer_start_tp.view;
    auto brt_for_possible_rotamer_tp =
        TPack<Int, 1, D>::zeros(n_possible_rotamers);
    auto brt_for_possible_rotamer = brt_for_possible_rotamer_tp.view;
    auto brt_for_possible_rotamer_boundaries_tp =
        TPack<Int, 1, D>::zeros(n_possible_rotamers);
    auto brt_for_possible_rotamer_boundaries =
        brt_for_possible_rotamer_boundaries_tp.view;

    auto mark_possrot_boundary_beginnings = [=](int buildable_restype) {
      Int const offset = possible_rotamer_offset_for_brt[buildable_restype];
      brt_for_possible_rotamer_boundaries[offset] = 1;
      brt_for_possible_rotamer_start[offset] = buildable_restype;
    };

    // for (int i = 0; i < n_brt; ++i) {
    //  mark_possrot_boundary_beginnings(i);
    //}
    Dispatch<D>::forall(n_brt, mark_possrot_boundary_beginnings);

    // for (int i = 0; i < n_possible_rotamers; ++i) {
    //   std::cout << "passrot boundary " << i << " " <<
    //   brt_for_possible_rotamer[i] << std::endl;
    // }

    // Non-segmented scan on "max" to get the brt index for each possible
    // rotamer
    // for (int i = 1; i < n_possible_rotamers; ++i) {
    //   brt_for_possible_rotamer[i] = std::max(
    //       brt_for_possible_rotamer[i], brt_for_possible_rotamer[i - 1]);
    // }
    Dispatch<D>::inclusive_scan(
        brt_for_possible_rotamer_start,
        brt_for_possible_rotamer,
        mgpu::maximum_t<Int>());

    // for (int i = 0; i < n_possible_rotamers; ++i) {
    //   std::cout << "passrot boundary " << i << " " <<
    //   brt_for_possible_rotamer[i] << std::endl;
    // }

    // We're eventually going to write down the probabilities for each
    // base rotamer in this tensor
    auto rotamer_probability_tp = TPack<Real, 1, D>::empty(n_possible_rotamers);
    auto rotamer_probability_cumsum_tp =
        TPack<Real, 1, D>::empty(n_possible_rotamers);

    auto rotamer_probability = rotamer_probability_tp.view;
    auto rotamer_probability_cumsum = rotamer_probability_cumsum_tp.view;

    auto calculate_possible_rotamer_probability = [=](int possible_rotamer) {
      // Compute the probability of the ith possible rotamer
      int const brt = brt_for_possible_rotamer[possible_rotamer];
      int const res = rottable_set_for_buildable_restype[brt][0];
      int const table_set = rottable_set_for_buildable_restype[brt][1];
      int const sorted_rotno =
          possible_rotamer - possible_rotamer_offset_for_brt[brt];

      // Caclulate the phi/psi bin indices
      // This needs to be turned into a function...
      Vec<Real, 2> bbdihe, bbstep;
      Vec<Int, 2> bin_index;
      for (int ii = 0; ii < 2; ++ii) {
        Real wrap_iidihe = backbone_dihedrals[2 * res + ii]
                           - rotameric_bb_start[table_set][ii];
        while (wrap_iidihe < 0) {
          wrap_iidihe += 2 * M_PI;
        }
        Real ii_period = rotameric_bb_periodicity[table_set][ii];
        while (wrap_iidihe > ii_period) {
          wrap_iidihe -= ii_period;
        }

        bbstep[ii] = rotameric_bb_step[table_set][ii];
        bbdihe[ii] = wrap_iidihe / bbstep[ii];
        bin_index[ii] = int(bbdihe[ii]);
      }

      // Look up the index of the rotamer: we know where the rotamer is in
      // sorted order
      // but not which chi values this represents. Also look up the "table
      // index" of
      // this rotamer, which for some AAs (e.g. LYS) is not the same as the
      // rotamer
      // index because some rotamers are too strained to be considered (e.g.
      // g+,g+,g+,g+)
      Int const tableset_offset = n_rotamers_for_tableset_offsets[table_set];
      Int const rot_table_ind =
          sorted_rotamer_2_rotamer[bin_index[0]][bin_index[1]]
                                  [tableset_offset + sorted_rotno]
          + tableset_offset;
      // int const local_table_ind = all_chi_rotind2tableind[
      //   all_chi_rotind2tableind_offsets[table_set] + rotno ];
      // int const rot_table_ind = local_table_ind + tableset_offset;

      // Now we know which rotamer we'll be building: time to look up
      // (interpolate)
      // the rotamer's probability from the rotameric_prob_tables
      // Int const rotable_ind = tableset_offset + rotno;

      // std::cout << "tableset " << table_set << " tableset_offset " <<
      // tableset_offset;
      // std::cout << " possible_rotamer " << possible_rotamer << " sorted_rotno
      // " << sorted_rotno;
      // std::cout << " bb inds " << bin_index[0] << " " << bin_index[1];
      // std::cout << " rot_table_ind " << rot_table_ind << std::endl;
      //
      // int nchi = nchi_for_tableset[table_set];
      // std::cout << "rotamer inds:";
      // for (int ii = 0; ii < nchi; ++ii) {
      // 	std::cout << " " << rotwells[rot_table_ind][ii];
      // }
      // std::cout << std::endl;

      TensorAccessor<Real, 2, D> rotprob_slice(
          rotameric_prob_tables.data()
              + rot_table_ind * rotameric_prob_tables.stride(0),
          rotprob_table_sizes.data()->data()
              + rot_table_ind * rotprob_table_sizes.stride(0),
          rotprob_table_strides.data()->data()
              + rot_table_ind * rotprob_table_strides.stride(0));
      auto prob_and_derivs =
          tmol::numeric::bspline::ndspline<2, 3, D, Real, Int>::interpolate(
              rotprob_slice, bbdihe);
      rotamer_probability[possible_rotamer] = std::get<0>(prob_and_derivs);
    };

    for (int ii = 0; ii < n_possible_rotamers; ++ii) {
      calculate_possible_rotamer_probability(ii);
    }

    // OK
    // Now we perform (exclusive) segmented scans on the possible-rotamer
    // probabilities
    // to get the cumulative sums of the probabilities so that we can decide
    // where to put the cutoff

    for (int ii = 0; ii < n_possible_rotamers; ++ii) {
      if (ii > 0 && !brt_for_possible_rotamer_boundaries[ii]) {
        rotamer_probability_cumsum[ii] =
            rotamer_probability_cumsum[ii - 1] + rotamer_probability[ii - 1];
      } else {
        rotamer_probability_cumsum[ii] = 0;
      }
    }

    // And with the cumulative sum, we can now decide which rotamers we will
    // build
    auto build_possible_rotamer_tp =
        TPack<Int, 1, D>::empty(n_possible_rotamers);
    auto build_possible_rotamer = build_possible_rotamer_tp.view;
    auto decide_on_possible_rotamer = [=](int possible_rotamer) {
      int const rt = brt_for_possible_rotamer[possible_rotamer];
      int keep = rotamer_probability_cumsum[possible_rotamer]
                 <= prob_cumsum_limit_for_buildable_restype[rt];
      build_possible_rotamer[possible_rotamer] = keep;
    };
    for (int ii = 0; ii < n_possible_rotamers; ++ii) {
      decide_on_possible_rotamer(ii);
    }

    // Let's count the number of possible rotamers we're keeping per restype
    auto count_rotamers_to_build_tp =
        TPack<Int, 1, D>::zeros(n_possible_rotamers);
    auto count_rotamers_to_build = count_rotamers_to_build_tp.view;

    // exclusive segmented scan on the build_possible_rotamer array
    for (int ii = 1; ii < n_possible_rotamers; ++ii) {
      if (!brt_for_possible_rotamer_boundaries[ii]) {
        count_rotamers_to_build[ii] =
            count_rotamers_to_build[ii - 1] + build_possible_rotamer[ii - 1];
      }
    }

    // And now the count of rotamers to build per restype:
    auto n_rotamers_to_build_per_brt_tp = TPack<Int, 1, D>::zeros(n_brt);
    auto n_rotamers_to_build_per_brt = n_rotamers_to_build_per_brt_tp.view;
    auto count_rots_to_build_per_brt = [=](int brt) {
      Int const offset = possible_rotamer_offset_for_brt[brt];
      Int const npossible = n_possible_rotamers_per_brt[brt];
      Int const brt_count = count_rotamers_to_build[offset + npossible - 1] + 1;
      n_rotamers_to_build_per_brt[brt] = brt_count;
    };
    for (int ii = 0; ii < n_brt; ++ii) {
      count_rots_to_build_per_brt(ii);
      // std::cout << "n_rotamers_to_build_per_brt[" << ii << "] ";
      // std::cout << n_rotamers_to_build_per_brt[ii] << std::endl;
    }

    // max_n_chi: I donno. reduction on max
    Int max_n_chi = 0;
    for (int ii = 0; ii < n_brt; ++ii) {
      max_n_chi = std::max(max_n_chi, nchi_for_buildable_restype[ii]);
    }

    // OK!
    // So the next step is to expand the base rotamers into extra rotamers
    // and to write down all the chi that we'll sample from.
    auto n_expansions_for_brt_tp = TPack<Int, 1, D>::zeros(n_brt);
    auto n_expansions_for_brt = n_expansions_for_brt_tp.view;

    auto expansion_dim_prods_for_brt_tp =
        TPack<Int, 2, D>::empty({n_brt, max_n_chi});
    auto expansion_dim_prods_for_brt = expansion_dim_prods_for_brt_tp.view;

    auto count_expansions_for_brt = [=](int brt) {
      Int const nchi = nchi_for_buildable_restype[brt];
      Int const table_set = rottable_set_for_buildable_restype[brt][1];
      Int const n_dun_chi = nchi_for_tableset[table_set];
      Int n_expansions = 1;

      Int const n_chi = nchi_for_buildable_restype[brt];
      for (int ii = n_chi - 1; ii >= n_dun_chi; --ii) {
        expansion_dim_prods_for_brt[brt][ii] = n_expansions;
        Int ii_expansion =
            non_dunbrack_expansion_counts_for_buildable_restype[brt][ii];
        if (ii_expansion != 0) {
          n_expansions *= ii_expansion;
        }
      }

      for (int ii = n_dun_chi - 1; ii >= 0; --ii) {
        expansion_dim_prods_for_brt[brt][ii] = n_expansions;
        // for now, only consider +/- 1 standard deviation sampling
        if (chi_expansion_for_buildable_restype[brt][ii]) {
          n_expansions *= 3;
        }
      }

      n_expansions_for_brt[brt] = n_expansions;
      n_rotamers_to_build_per_brt[brt] *= n_expansions;
    };

    for (int ii = 0; ii < n_brt; ++ii) {
      count_expansions_for_brt(ii);
      // std::cout << "n_expansions_for_brt[" << ii << "] ";
      // std::cout << n_expansions_for_brt[ii] << " ";
      // std::cout << "n_rotamers_to_build_per_brt[" << ii << "] ";
      // std::cout << n_rotamers_to_build_per_brt[ii] << std::endl;
    }

    auto n_rotamers_to_build_per_brt_offsets_tp =
        TPack<Int, 1, D>::zeros(n_brt);
    auto n_rotamers_to_build_per_brt_offsets =
        n_rotamers_to_build_per_brt_offsets_tp.view;

    // Exclusive cumumaltive sum
    for (int ii = 1; ii < n_brt; ++ii) {
      n_rotamers_to_build_per_brt_offsets[ii] =
          n_rotamers_to_build_per_brt_offsets[ii - 1]
          + n_rotamers_to_build_per_brt[ii - 1];
    }
    Int const n_rotamers = n_rotamers_to_build_per_brt_offsets[n_brt - 1]
                           + n_rotamers_to_build_per_brt[n_brt - 1];

    // Get a mapping from rotamer index to buildable restype
    auto brt_for_rotamer_tp = TPack<Int, 1, D>::zeros(n_rotamers);
    auto brt_for_rotamer = brt_for_rotamer_tp.view;

    // unclear if we need this one...
    auto brt_for_rotamer_boundaries_tp = TPack<Int, 1, D>::zeros(n_rotamers);
    auto brt_for_rotamer_boundaries = brt_for_rotamer_boundaries_tp.view;

    auto mark_rot_brt_boundary_beginnings = [=](int brt) {
      Int const offset = n_rotamers_to_build_per_brt_offsets[brt];
      brt_for_rotamer_boundaries[offset] = 1;
      brt_for_rotamer[offset] = brt;
    };
    for (int ii = 0; ii < n_brt; ++ii) {
      mark_rot_brt_boundary_beginnings(ii);
    }

    // Now scan on max and record the restype for each rotamer
    for (int ii = 1; ii < n_rotamers; ++ii) {
      brt_for_rotamer[ii] =
          std::max(brt_for_rotamer[ii], brt_for_rotamer[ii - 1]);
      // std::cout << "brt for rotamer " << ii << " " << brt_for_rotamer[ii] <<
      // std::endl;
    }

    // OK Now allocate space for the chi that we're going to write to
    // auto chi_for_rotamers_tp = TPack<Real, 2, D>::empty({n_rotamers,
    // max_n_chi});
    auto chi_for_rotamers_tp =
        TPack<Real, 2, D>::zeros({n_rotamers, max_n_chi});
    auto chi_for_rotamers = chi_for_rotamers_tp.view;

    // Now we can construct the chi samples.
    // One thread per rotamer
    // each rotamer figures out from its index
    //  -- which restype it's building for
    //  -- its base rotamer index
    //  -- which expanded rotamer for the base rotamer it is

    auto sample_chi_for_rotamer = [=](int rotamer) {
      int const brt = brt_for_rotamer[rotamer];
      int const res = rottable_set_for_buildable_restype[brt][0];
      int const table_set = rottable_set_for_buildable_restype[brt][1];
      int const expanded_rotamer_for_brt =
          rotamer - n_rotamers_to_build_per_brt_offsets[brt];
      int const n_expansions = n_expansions_for_brt[brt];
      int const base_rotno = expanded_rotamer_for_brt / n_expansions;
      int const expanded_rotno = expanded_rotamer_for_brt % n_expansions;

      Vec<Real, 2> bbdihe, bbstep;
      Vec<Int, 2> bin_index;
      for (int ii = 0; ii < 2; ++ii) {
        Real wrap_iidihe = backbone_dihedrals[2 * res + ii]
                           - rotameric_bb_start[table_set][ii];
        while (wrap_iidihe < 0) {
          wrap_iidihe += 2 * M_PI;
        }
        Real ii_period = rotameric_bb_periodicity[table_set][ii];
        while (wrap_iidihe > ii_period) {
          wrap_iidihe -= ii_period;
        }

        bbstep[ii] = rotameric_bb_step[table_set][ii];
        bbdihe[ii] = wrap_iidihe / bbstep[ii];
        bin_index[ii] = int(bbdihe[ii]);
      }

      // Look up the index of the rotamer: we know where the rotamer is in
      // sorted order
      // but not which chi values this represents.
      Int const n_dun_chi = nchi_for_tableset[table_set];
      Int const n_chi = nchi_for_buildable_restype[brt];
      Int const tableset_offset = n_rotamers_for_tableset_offsets[table_set];
      Int const rotmean_offset = rotameric_meansdev_tableset_offsets[table_set];
      Int const rot_table_start =
          n_dun_chi * sorted_rotamer_2_rotamer[bin_index[0]][bin_index[1]]
                                              [tableset_offset + base_rotno]
          + rotmean_offset;

      // {
      // int nchi = nchi_for_tableset[table_set];
      // std::cout << "rotamer inds:";
      // for (int ii = 0; ii < nchi; ++ii) {
      // 	std::cout << " " << rotwells[rot_table_ind][ii];
      // }
      // std::cout << std::endl;
      // }

      // OK: we have the phi & psi values, and we have the rotamer index
      // Now we can start calculating the chi.

      // How do we do that?
      // For each chi,
      //  if it is a dunbrack chi:
      //    we will look up the mean value, and if
      //    we are expanding this chi, and the expansion
      //    index calls for the standard devation, then
      //    we will also look up the standard devation.
      //    then we will record the perturbed chi value
      //  else
      //    we will look up the non_dunbrack_expansion

      int expanded_rot_remainder = expanded_rotno;
      for (int ii = 0; ii < n_chi; ++ii) {
        int const ii_dim_prods = expansion_dim_prods_for_brt[brt][ii];
        int const ii_expansion = expanded_rot_remainder / ii_dim_prods;
        expanded_rot_remainder = expanded_rot_remainder % ii_dim_prods;

        Real ii_chi;
        if (ii < n_dun_chi) {
          TensorAccessor<Real, 2, D> rotmean_slice(
              rotameric_mean_tables.data()
                  + (rot_table_start + ii) * rotameric_mean_tables.stride(0),
              rotmean_table_sizes.data()->data()
                  + (rot_table_start + ii) * rotmean_table_sizes.stride(0),
              rotmean_table_strides.data()->data()
                  + (rot_table_start + ii) * rotmean_table_strides.stride(0));

          auto mean_and_derivs =
              tmol::numeric::bspline::ndspline<2, 3, D, Real, Int>::interpolate(
                  rotmean_slice, bbdihe);
          ii_chi = std::get<0>(mean_and_derivs);
          if (chi_expansion_for_buildable_restype[brt][ii]
              && ii_expansion > 0) {
            // OK! we expand this chi; so retrieve the standard deviation
            TensorAccessor<Real, 2, D> rotsdev_slice(
                rotameric_sdev_tables.data()
                    + (rot_table_start + ii) * rotameric_sdev_tables.stride(0),
                rotmean_table_sizes.data()->data()
                    + (rot_table_start + ii) * rotmean_table_sizes.stride(0),
                rotmean_table_strides.data()->data()
                    + (rot_table_start + ii) * rotmean_table_strides.stride(0));
            auto sdev_and_derivs =
                tmol::numeric::bspline::ndspline<2, 3, D, Real, Int>::
                    interpolate(rotsdev_slice, bbdihe);
            Real sdev = std::get<0>(sdev_and_derivs);
            if (ii_expansion == 1) {
              ii_chi += sdev;
            } else {
              // i.e ii_expansion == 2
              ii_chi -= sdev;
            }
          }
        } else {
          ii_chi = non_dunbrack_expansion_for_buildable_restype[brt][ii]
                                                               [ii_expansion];
        }
        chi_for_rotamers[rotamer][ii] = ii_chi;
      }
    };

    for (int ii = 0; ii < n_rotamers; ++ii) {
      sample_chi_for_rotamer(ii);
      std::cout << "rotamer " << ii;
      for (int jj = 0; jj < max_n_chi; ++jj) {
        std::cout << " " << 180 / M_PI * chi_for_rotamers[ii][jj];
      }
      std::cout << std::endl;
    }

    return {n_rotamers_to_build_per_brt_tp,
            n_rotamers_to_build_per_brt_offsets_tp,
            brt_for_rotamer_tp,
            chi_for_rotamers_tp};
  }
};

}  // namespace rotamer
}  // namespace pack
}  // namespace tmol
