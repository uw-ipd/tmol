#pragma once
#include <tmol/utility/tensor/TensorPack.h>
#include "test.hh"

namespace tmol {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <template <Device> class Dispatch, Device D, typename Real, typename Int>
void DunbrackChiSamplerTester<Dispatch, D, Real, Int>::
    determine_n_possible_rots(
        TView<Int, 2, D> rottable_set_for_buildable_restype,
        TView<int64_t, 1, D> n_rotamers_for_tableset,
        TView<Int, 1, D> n_possible_rotamers_per_brt) {
  pack::rotamer::DunbrackChiSampler<Dispatch, D, Real, Int>::
      determine_n_possible_rots(
          rottable_set_for_buildable_restype,
          n_rotamers_for_tableset,
          n_possible_rotamers_per_brt);
}

template <template <Device> class Dispatch, Device D, typename Real, typename Int>
void DunbrackChiSamplerTester<Dispatch, D, Real, Int>::fill_in_brt_for_possrots(
    TView<Int, 1, D> possible_rotamer_offset_for_brt,
    TView<Int, 1, D> brt_for_possible_rotamer,
    TView<Int, 1, D> brt_for_possible_rotamer_boundaries) {
  pack::rotamer::DunbrackChiSampler<Dispatch, D, Real, Int>::
      fill_in_brt_for_possrots(
          possible_rotamer_offset_for_brt,
          brt_for_possible_rotamer,
          brt_for_possible_rotamer_boundaries);
}

template <template <Device> class Dispatch, Device D, typename Real, typename Int>
void DunbrackChiSamplerTester<Dispatch, D, Real, Int>::
    interpolate_probabilities_for_possible_rotamers(
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
  pack::rotamer::DunbrackChiSampler<Dispatch, D, Real, Int>::
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
}

template <template <Device> class Dispatch, Device D, typename Real, typename Int>
void DunbrackChiSamplerTester<Dispatch, D, Real, Int>::
    determine_n_base_rotamers_to_build(
        TView<Real, 1, D> prob_cumsum_limit_for_buildable_restype,
        TView<Int, 1, D> n_possible_rotamers_per_brt,
        TView<Int, 1, D> brt_for_possible_rotamer,
        TView<Int, 1, D> brt_for_possible_rotamer_boundaries,
        TView<Int, 1, D> possible_rotamer_offset_for_brt,
        TView<Real, 1, D> rotamer_probability,
        TView<Int, 1, D> n_rotamers_to_build_per_brt) {
  pack::rotamer::DunbrackChiSampler<Dispatch, D, Real, Int>::
      determine_n_base_rotamers_to_build(
          prob_cumsum_limit_for_buildable_restype,
          n_possible_rotamers_per_brt,
          brt_for_possible_rotamer,
          brt_for_possible_rotamer_boundaries,
          possible_rotamer_offset_for_brt,
          rotamer_probability,
          n_rotamers_to_build_per_brt);
}

template <template <Device> class Dispatch, Device D, typename Real, typename Int>
Int DunbrackChiSamplerTester<Dispatch, D, Real, Int>::count_expanded_rotamers(
    TView<Int, 1, D> nchi_for_buildable_restype,
    TView<Int, 2, D> rottable_set_for_buildable_restype,
    TView<Int, 1, D> nchi_for_tableset,
    TView<Int, 2, D> chi_expansion_for_buildable_restype,
    TView<Int, 2, D> non_dunbrack_expansion_counts_for_buildable_restype,
    TView<Int, 1, D> n_expansions_for_brt,
    TView<Int, 2, D> expansion_dim_prods_for_brt,
    TView<Int, 1, D> n_rotamers_to_build_per_brt,
    TView<Int, 1, D> n_rotamers_to_build_per_brt_offsets) {
  return pack::rotamer::DunbrackChiSampler<Dispatch, D, Real, Int>::
      count_expanded_rotamers(
          nchi_for_buildable_restype,
          rottable_set_for_buildable_restype,
          nchi_for_tableset,
          chi_expansion_for_buildable_restype,
          non_dunbrack_expansion_counts_for_buildable_restype,
          n_expansions_for_brt,
          expansion_dim_prods_for_brt,
          n_rotamers_to_build_per_brt,
          n_rotamers_to_build_per_brt_offsets);
}

template <template <Device> class Dispatch, Device D, typename Real, typename Int>
void DunbrackChiSamplerTester<Dispatch, D, Real, Int>::
    map_from_rotamer_index_to_brt(
        TView<Int, 1, D> n_rotamers_to_build_per_brt_offsets,
        TView<Int, 1, D> brt_for_rotamer) {
  pack::rotamer::DunbrackChiSampler<Dispatch, D, Real, Int>::
      map_from_rotamer_index_to_brt(
          n_rotamers_to_build_per_brt_offsets, brt_for_rotamer);
}

template <template <Device> class Dispatch, Device D, typename Real, typename Int>
void DunbrackChiSamplerTester<Dispatch, D, Real, Int>::sample_chi_for_rotamers(
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
  pack::rotamer::DunbrackChiSampler<Dispatch, D, Real, Int>::
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
}
}
