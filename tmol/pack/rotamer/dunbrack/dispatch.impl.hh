#pragma once

#include <Eigen/Core>
#include <tuple>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorCollection.h>
#include <tmol/utility/tensor/TensorPack.h>

#include <tmol/numeric/bspline_compiled/bspline.hh>
#include <tmol/score/common/forall_dispatch.cpu.impl.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/tuple.hh>

#include <tmol/extern/moderngpu/operators.hxx>

#include <ATen/Tensor.h>

namespace tmol {
namespace pack {
namespace rotamer {
namespace dunbrack {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct DunbrackChiSampler {
  static auto
  f(TView<Vec<Real, 3>, 1, D> coords,

    TView<Real, 3, D> rotameric_prob_tables,
    TView<Vec<int64_t, 2>, 1, D> rotprob_table_sizes,
    TView<Vec<int64_t, 2>, 1, D> rotprob_table_strides,
    TView<Real, 3, D> rotameric_mean_tables,
    TView<Real, 3, D> rotameric_sdev_tables,
    TView<Vec<int64_t, 2>, 1, D> rotmean_table_sizes,
    TView<Vec<int64_t, 2>, 1, D> rotmean_table_strides,
    TView<Int, 1, D> rotameric_meansdev_tableset_offsets,
    TView<Vec<Real, 2>, 1, D> rotameric_bb_start,          // ntable-set entries
    TView<Vec<Real, 2>, 1, D> rotameric_bb_step,           // ntable-set entries
    TView<Vec<Real, 2>, 1, D> rotameric_bb_periodicity,    // ntable-set entries
    TView<Real, 4, D> /*semirotameric_tables*/,            // n-semirot-tabset
    TView<Vec<int64_t, 3>, 1, D> /*semirot_table_sizes*/,  // n-semirot-tabset
    TView<Vec<int64_t, 3>, 1, D> /*semirot_table_strides*/,  // n-semirot-tabset
    TView<Vec<Real, 3>, 1, D> /*semirot_start*/,             // n-semirot-tabset
    TView<Vec<Real, 3>, 1, D> /*semirot_step*/,              // n-semirot-tabset
    TView<Vec<Real, 3>, 1, D> /*semirot_periodicity*/,       // n-semirot-tabset
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
    TView<Int, 1, D> nchi_for_buildable_restype  // inc. hydroxyl chi, e.g.

    )
      -> std::tuple<
          TPack<Int, 1, D>,
          TPack<Int, 1, D>,
          TPack<Int, 1, D>,
          TPack<Real, 2, D> > {
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

    auto n_possible_rotamers_per_brt_tp = TPack<Int, 1, D>::zeros(n_brt);
    auto n_possible_rotamers_per_brt = n_possible_rotamers_per_brt_tp.view;

    // std::cout << "1" << std::endl;
    determine_n_possible_rots(
        rottable_set_for_buildable_restype,
        n_rotamers_for_tableset,
        n_possible_rotamers_per_brt);

    auto possible_rotamer_offset_for_brt_tp = TPack<Int, 1, D>::zeros(n_brt);
    auto possible_rotamer_offset_for_brt =
        possible_rotamer_offset_for_brt_tp.view;

    // std::cout << "2" << std::endl;
    // Exclusive cumulative sum of n_possible_rotamers_per_restype.
    // Get total number of possible rotamers over all residue types
    Int const n_possible_rotamers = Dispatch<D>::exclusive_scan_w_final_val(
        n_possible_rotamers_per_brt,
        possible_rotamer_offset_for_brt,
        mgpu::plus_t<Real>());
    // std::cout << "n_possible_rotamers " << n_possible_rotamers << std::endl;

    // std::cout << "3" << std::endl;
    // There are some things we need to know about the ith possible rotamer:
    //   1. What buildable_residue type does it come from?
    //   2. What table set does it come from?
    //   3. What residue does it come from?

    auto backbone_dihedrals_tp = TPack<Real, 1, D>::empty(nres * 2);
    auto backbone_dihedrals = backbone_dihedrals_tp.view;

    // This should be extracted to a function
    auto compute_backbone_dihedrals = [=] EIGEN_DEVICE_FUNC(int res) {
      for (int dihe_ind = 0; dihe_ind < ndihe_for_res[res]; ++dihe_ind) {
        int i = dihedral_offset_for_res[res] + dihe_ind;
        Int at0 = dihedral_atom_inds[i][0];
        Int at1 = dihedral_atom_inds[i][1];
        Int at2 = dihedral_atom_inds[i][2];
        Int at3 = dihedral_atom_inds[i][3];
        Real dihe = 0;
        if (at0 > 0 && at1 > 0 && at2 > 0 && at3 > 0) {
          dihe = score::common::dihedral_angle<Real>::V(
              coords[at0], coords[at1], coords[at2], coords[at3]);
        } else if (dihe_ind == 0) {
          // neutral phi
          dihe = -60;  // As suggested by Roland Dunbrack
        } else if (dihe_ind == 1) {
          // neutral psi
          dihe = 60;  // As suggested by Roland Dunbrack
        }
        backbone_dihedrals[i] = dihe;
      }
    };

    Dispatch<D>::forall(
        dihedral_offset_for_res.size(0), compute_backbone_dihedrals);

    // std::cout << "4; n_possible_rotamers " << n_possible_rotamers <<
    // std::endl;

    auto brt_for_possible_rotamer_tp =
        TPack<Int, 1, D>::zeros(n_possible_rotamers);
    auto brt_for_possible_rotamer = brt_for_possible_rotamer_tp.view;

    fill_in_brt_for_possrots(
        possible_rotamer_offset_for_brt, brt_for_possible_rotamer);

    // Write down the probabilities for each base rotamer in this tensor
    auto rotamer_probability_tp = TPack<Real, 1, D>::empty(n_possible_rotamers);
    auto rotamer_probability = rotamer_probability_tp.view;

    interpolate_probabilities_for_possible_rotamers(
        rotameric_prob_tables,
        rotprob_table_sizes,
        rotprob_table_strides,
        rotameric_bb_start,
        rotameric_bb_step,
        rotameric_bb_periodicity,
        n_rotamers_for_tableset_offsets,
        sorted_rotamer_2_rotamer,
        rottable_set_for_buildable_restype,
        brt_for_possible_rotamer,
        possible_rotamer_offset_for_brt,
        backbone_dihedrals,
        rotamer_probability);
    // std::cout << "5" << std::endl;

    // And now the count of rotamers to build per restype:
    auto n_rotamers_to_build_per_brt_tp = TPack<Int, 1, D>::zeros(n_brt);
    auto n_rotamers_to_build_per_brt = n_rotamers_to_build_per_brt_tp.view;

    determine_n_base_rotamers_to_build(
        prob_cumsum_limit_for_buildable_restype,
        n_possible_rotamers_per_brt,
        brt_for_possible_rotamer,
        possible_rotamer_offset_for_brt,
        rotamer_probability,
        n_rotamers_to_build_per_brt);
    // std::cout << "6" << std::endl;

    // max_n_chi: reduction on max
    Int max_n_chi =
        Dispatch<D>::reduce(nchi_for_buildable_restype, mgpu::maximum_t<Int>());

    // OK!
    // So the next step is to expand the base rotamers into extra rotamers
    // and to write down all the chi that we'll sample from.
    auto n_expansions_for_brt_tp = TPack<Int, 1, D>::zeros(n_brt);
    auto n_expansions_for_brt = n_expansions_for_brt_tp.view;

    auto expansion_dim_prods_for_brt_tp =
        TPack<Int, 2, D>::empty({n_brt, max_n_chi});
    auto expansion_dim_prods_for_brt = expansion_dim_prods_for_brt_tp.view;

    auto n_rotamers_to_build_per_brt_offsets_tp =
        TPack<Int, 1, D>::zeros(n_brt);
    auto n_rotamers_to_build_per_brt_offsets =
        n_rotamers_to_build_per_brt_offsets_tp.view;

    Int n_rotamers = count_expanded_rotamers(
        nchi_for_buildable_restype,
        rottable_set_for_buildable_restype,
        nchi_for_tableset,
        chi_expansion_for_buildable_restype,
        non_dunbrack_expansion_counts_for_buildable_restype,
        n_expansions_for_brt,
        expansion_dim_prods_for_brt,
        n_rotamers_to_build_per_brt,
        n_rotamers_to_build_per_brt_offsets);
    // std::cout << "n rotamers" << n_rotamers << std::endl;

    // std::cout << "7" << std::endl;

    // Get a mapping from rotamer index to buildable restype
    auto brt_for_rotamer_tp = TPack<Int, 1, D>::zeros(n_rotamers);
    auto brt_for_rotamer = brt_for_rotamer_tp.view;
    map_from_rotamer_index_to_brt(
        n_rotamers_to_build_per_brt_offsets, brt_for_rotamer);
    // std::cout << "8" << std::endl;

    // OK Now allocate space for the chi that we're going to write to
    // auto chi_for_rotamers_tp = TPack<Real, 2, D>::empty({n_rotamers,
    // max_n_chi});
    auto chi_for_rotamers_tp =
        TPack<Real, 2, D>::zeros({n_rotamers, max_n_chi});
    auto chi_for_rotamers = chi_for_rotamers_tp.view;

    sample_chi_for_rotamers(
        rotameric_mean_tables,
        rotameric_sdev_tables,
        rotmean_table_sizes,
        rotmean_table_strides,
        rotameric_meansdev_tableset_offsets,
        rotameric_bb_start,
        rotameric_bb_step,
        rotameric_bb_periodicity,

        sorted_rotamer_2_rotamer,
        nchi_for_tableset,

        rottable_set_for_buildable_restype,
        chi_expansion_for_buildable_restype,
        non_dunbrack_expansion_for_buildable_restype,
        nchi_for_buildable_restype,

        backbone_dihedrals,

        n_rotamers_to_build_per_brt_offsets,
        brt_for_rotamer,
        n_expansions_for_brt,
        n_rotamers_for_tableset_offsets,

        expansion_dim_prods_for_brt,
        chi_for_rotamers);
    // std::cout << "9" << std::endl;

    return {
        n_rotamers_to_build_per_brt_tp,
        n_rotamers_to_build_per_brt_offsets_tp,
        brt_for_rotamer_tp,
        chi_for_rotamers_tp};
  }

  static void determine_n_possible_rots(
      TView<Int, 2, D> rottable_set_for_buildable_restype,
      TView<int64_t, 1, D> n_rotamers_for_tableset,
      TView<Int, 1, D> n_possible_rotamers_per_brt) {
    Int const n_brt = rottable_set_for_buildable_restype.size(0);
    assert(n_possible_rotamers_per_brt.size(0) == n_brt);
    auto lambda_determine_n_possible_rots = [=] EIGEN_DEVICE_FUNC(int brt) {
      Int rottable_set = rottable_set_for_buildable_restype[brt][1];
      n_possible_rotamers_per_brt[brt] = n_rotamers_for_tableset[rottable_set];
    };

    Dispatch<D>::forall(n_brt, lambda_determine_n_possible_rots);
  }

  static void fill_in_brt_for_possrots(
      TView<Int, 1, D> possible_rotamer_offset_for_brt,
      TView<Int, 1, D> brt_for_possible_rotamer) {
    int const n_brt = possible_rotamer_offset_for_brt.size(0);
    int const n_possible_rotamers = brt_for_possible_rotamer.size(0);

    // std::cout << "n possible rotamers: " << n_possible_rotamers << std::endl;
    auto brt_for_possible_rotamer_start_tp =
        TPack<Int, 1, D>::zeros(n_possible_rotamers);
    auto brt_for_possible_rotamer_start =
        brt_for_possible_rotamer_start_tp.view;
    // std::cout << "brt_for_possible_rotamer_start.size(0): " <<
    // brt_for_possible_rotamer_start.size(0) << std::endl;

    auto mark_possrot_boundary_beginnings =
        [=] EIGEN_DEVICE_FUNC(int buildable_restype) {
          Int const offset = possible_rotamer_offset_for_brt[buildable_restype];
          brt_for_possible_rotamer_start[offset] = buildable_restype;
        };

    Dispatch<D>::forall(n_brt, mark_possrot_boundary_beginnings);

    // Non-segmented scan on "max" to get the brt index for each possible
    // rotamer
    Dispatch<D>::inclusive_scan(
        brt_for_possible_rotamer_start,
        brt_for_possible_rotamer,
        mgpu::maximum_t<Int>());
  }

  static void interpolate_probabilities_for_possible_rotamers(
      TView<Real, 3, D> rotameric_prob_tables,
      TView<Vec<int64_t, 2>, 1, D> rotprob_table_sizes,
      TView<Vec<int64_t, 2>, 1, D> rotprob_table_strides,
      TView<Vec<Real, 2>, 1, D> rotameric_bb_start,
      TView<Vec<Real, 2>, 1, D> rotameric_bb_step,
      TView<Vec<Real, 2>, 1, D> rotameric_bb_periodicity,
      TView<Int, 1, D> n_rotamers_for_tableset_offsets,
      TView<int64_t, 3, D> sorted_rotamer_2_rotamer,
      TView<Int, 2, D> rottable_set_for_buildable_restype,
      TView<Int, 1, D> brt_for_possible_rotamer,
      TView<Int, 1, D> possible_rotamer_offset_for_brt,
      TView<Real, 1, D> backbone_dihedrals,
      TView<Real, 1, D> rotamer_probability) {
    int const n_possible_rotamers = brt_for_possible_rotamer.size(0);

    auto calculate_possible_rotamer_probability = [=] EIGEN_DEVICE_FUNC(
                                                      int possible_rotamer) {
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
      // sorted order but not which chi values this represents. Also look
      // up the "table index" of this rotamer, which for some AAs
      // (e.g. LYS) is not the same as the rotamer index because some
      // rotamers are too strained to be considered (e.g. g+,g+,g+,g+)
      Int const tableset_offset = n_rotamers_for_tableset_offsets[table_set];
      Int const rot_table_ind =
          sorted_rotamer_2_rotamer[bin_index[0]][bin_index[1]]
                                  [tableset_offset + sorted_rotno]
          + tableset_offset;

      // Now we know which rotamer we'll be building: time to look up
      // (interpolate) the rotamer's probability from the
      // rotameric_prob_tables

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
      rotamer_probability[possible_rotamer] =
          score::common::get<0>(prob_and_derivs);
    };
    Dispatch<D>::forall(
        n_possible_rotamers, calculate_possible_rotamer_probability);
  }

  static void determine_n_base_rotamers_to_build(
      TView<Real, 1, D> prob_cumsum_limit_for_buildable_restype,
      TView<Int, 1, D> n_possible_rotamers_per_brt,
      TView<Int, 1, D> brt_for_possible_rotamer,
      TView<Int, 1, D> possible_rotamer_offset_for_brt,
      TView<Real, 1, D> rotamer_probability,
      TView<Int, 1, D> n_rotamers_to_build_per_brt) {
    int const n_brt = n_rotamers_to_build_per_brt.size(0);
    int const n_possible_rotamers = rotamer_probability.size(0);
    assert(prob_cumsum_limit_for_buildable_restype.size(0) == n_brt);
    assert(brt_for_possible_rotamer.size(0) == n_possible_rotamers);
    assert(possible_rotamer_offset_for_brt.size(0) == n_brt);

    // OK
    // Now we perform (exclusive) segmented scans on the possible-rotamer
    // probabilities to get the cumulative sums of the probabilities so
    // that we can decide where to put the cutoff
    auto rotamer_probability_cumsum_tp =
        TPack<Real, 1, D>::empty(n_possible_rotamers);
    auto rotamer_probability_cumsum = rotamer_probability_cumsum_tp.view;

    Dispatch<D>::exclusive_segmented_scan(
        rotamer_probability,
        possible_rotamer_offset_for_brt,
        rotamer_probability_cumsum,
        mgpu::plus_t<Real>());

    // And with the cumulative sum, we can now decide which rotamers we will
    // build
    auto build_possible_rotamer_tp =
        TPack<Int, 1, D>::empty(n_possible_rotamers);
    auto build_possible_rotamer = build_possible_rotamer_tp.view;
    auto decide_on_possible_rotamer =
        [=] EIGEN_DEVICE_FUNC(int possible_rotamer) {
          int const rt = brt_for_possible_rotamer[possible_rotamer];
          int keep = rotamer_probability_cumsum[possible_rotamer]
                     <= prob_cumsum_limit_for_buildable_restype[rt];
          build_possible_rotamer[possible_rotamer] = keep;
        };

    Dispatch<D>::forall(n_possible_rotamers, decide_on_possible_rotamer);

    // Let's count the number of possible rotamers we're keeping per restype
    auto count_rotamers_to_build_tp =
        TPack<Int, 1, D>::zeros(n_possible_rotamers);
    auto count_rotamers_to_build = count_rotamers_to_build_tp.view;

    // exclusive segmented scan on the build_possible_rotamer array
    Dispatch<D>::exclusive_segmented_scan(
        build_possible_rotamer,
        possible_rotamer_offset_for_brt,
        count_rotamers_to_build,
        mgpu::plus_t<Int>());

    auto count_rots_to_build_per_brt = [=] EIGEN_DEVICE_FUNC(int brt) {
      Int const offset = possible_rotamer_offset_for_brt[brt];
      Int const npossible = n_possible_rotamers_per_brt[brt];
      Int const brt_count = count_rotamers_to_build[offset + npossible - 1];
      n_rotamers_to_build_per_brt[brt] = brt_count;
    };

    Dispatch<D>::forall(n_brt, count_rots_to_build_per_brt);
  }

  static Int count_expanded_rotamers(
      TView<Int, 1, D> nchi_for_buildable_restype,
      TView<Int, 2, D> rottable_set_for_buildable_restype,
      TView<Int, 1, D> nchi_for_tableset,
      TView<Int, 2, D> chi_expansion_for_buildable_restype,
      TView<Int, 2, D> non_dunbrack_expansion_counts_for_buildable_restype,
      TView<Int, 1, D> n_expansions_for_brt,
      TView<Int, 2, D> expansion_dim_prods_for_brt,
      TView<Int, 1, D> n_rotamers_to_build_per_brt,
      TView<Int, 1, D> n_rotamers_to_build_per_brt_offsets) {
    int const n_brt = nchi_for_buildable_restype.size(0);
    int const max_nchi = expansion_dim_prods_for_brt.size(1);

    assert(rottable_set_for_buildable_restype.size(0) == n_brt);
    assert(chi_expansion_for_buildable_restype.size(0) == n_brt);
    assert(n_expansions_for_brt.size(0) == n_brt);
    assert(expansion_dim_prods_for_brt.size(0) == n_brt);
    assert(n_rotamers_to_build_per_brt.size(0) == n_brt);
    assert(n_rotamers_to_build_per_brt_offsets.size(0) == n_brt);

    auto count_expansions_for_brt = [=] EIGEN_DEVICE_FUNC(int brt) {
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

    Dispatch<D>::forall(n_brt, count_expansions_for_brt);

    // Exclusive cumumaltive sum
    Int const n_rotamers = Dispatch<D>::exclusive_scan_w_final_val(
        n_rotamers_to_build_per_brt,
        n_rotamers_to_build_per_brt_offsets,
        mgpu::plus_t<Int>());

    return n_rotamers;
  }

  static void map_from_rotamer_index_to_brt(
      TView<Int, 1, D> n_rotamers_to_build_per_brt_offsets,
      TView<Int, 1, D> brt_for_rotamer) {
    int const n_rotamers = brt_for_rotamer.size(0);
    int const n_brt = n_rotamers_to_build_per_brt_offsets.size(0);

    auto brt_for_rotamer_start_tp = TPack<Int, 1, D>::zeros(n_rotamers);
    auto brt_for_rotamer_start = brt_for_rotamer_start_tp.view;

    auto mark_rot_brt_boundary_beginnings = [=] EIGEN_DEVICE_FUNC(int brt) {
      Int const offset = n_rotamers_to_build_per_brt_offsets[brt];
      // brt_for_rotamer_boundaries[offset] = 1;
      brt_for_rotamer_start[offset] = brt;
    };

    Dispatch<D>::forall(n_brt, mark_rot_brt_boundary_beginnings);

    // Now scan on max and record the restype for each rotamer
    Dispatch<D>::inclusive_scan(
        brt_for_rotamer_start, brt_for_rotamer, mgpu::maximum_t<Int>());
  }

  static void sample_chi_for_rotamers(
      TView<Real, 3, D> rotameric_mean_tables,
      TView<Real, 3, D> rotameric_sdev_tables,
      TView<Vec<int64_t, 2>, 1, D> rotmean_table_sizes,
      TView<Vec<int64_t, 2>, 1, D> rotmean_table_strides,
      TView<Int, 1, D> rotameric_meansdev_tableset_offsets,
      TView<Vec<Real, 2>, 1, D> rotameric_bb_start,
      TView<Vec<Real, 2>, 1, D> rotameric_bb_step,
      TView<Vec<Real, 2>, 1, D> rotameric_bb_periodicity,

      TView<int64_t, 3, D> sorted_rotamer_2_rotamer,
      TView<Int, 1, D> nchi_for_tableset,

      TView<Int, 2, D> rottable_set_for_buildable_restype,
      TView<Int, 2, D> chi_expansion_for_buildable_restype,
      TView<Real, 3, D> non_dunbrack_expansion_for_buildable_restype,
      TView<Int, 1, D> nchi_for_buildable_restype,

      TView<Real, 1, D> backbone_dihedrals,

      TView<Int, 1, D> n_rotamers_to_build_per_brt_offsets,
      TView<Int, 1, D> brt_for_rotamer,
      TView<Int, 1, D> n_expansions_for_brt,
      TView<Int, 1, D> n_rotamers_for_tableset_offsets,

      TView<Int, 2, D> expansion_dim_prods_for_brt,
      TView<Real, 2, D> chi_for_rotamers) {
    int const n_rotamers = chi_for_rotamers.size(0);
    int const max_n_chi = chi_for_rotamers.size(1);

    // Now we can construct the chi samples.
    // One thread per rotamer
    // each rotamer figures out from its index
    //  -- which restype it's building for
    //  -- its base rotamer index
    //  -- which expanded rotamer for the base rotamer it is

    auto sample_chi_for_rotamer = [=] EIGEN_DEVICE_FUNC(int rotamer) {
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
          n_dun_chi
              * sorted_rotamer_2_rotamer[bin_index[0]][bin_index[1]]
                                        [tableset_offset + base_rotno]
          + rotmean_offset;

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
          ii_chi = score::common::get<0>(mean_and_derivs);
          if (chi_expansion_for_buildable_restype[brt][ii]) {
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
            Real sdev = score::common::get<0>(sdev_and_derivs);
            if (ii_expansion == 0) {
              ii_chi -= sdev;
            } else if (ii_expansion == 2) {
              ii_chi += sdev;
            }
          }
        } else {
          ii_chi = non_dunbrack_expansion_for_buildable_restype[brt][ii]
                                                               [ii_expansion];
        }
        chi_for_rotamers[rotamer][ii] = ii_chi;
      }
    };

    Dispatch<D>::forall(n_rotamers, sample_chi_for_rotamer);
  }
};

}  // namespace dunbrack
}  // namespace rotamer
}  // namespace pack
}  // namespace tmol
