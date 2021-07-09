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
      TView<Int, 1, D> nchi_for_buildable_restype

      )
      -> std::tuple<
          TPack<Int, 1, D>,
          TPack<Int, 1, D>,
          TPack<Int, 1, D>,
          TPack<Real, 2, D> >;

  static void determine_n_possible_rots(
      TView<Int, 2, D> rottable_set_for_buildable_restype,
      TView<int64_t, 1, D> n_rotamers_for_tableset,
      TView<Int, 1, D> n_possible_rotamers_per_brt);

  static void fill_in_brt_for_possrots(
      TView<Int, 1, D> possible_rotamer_offset_for_brt,
      TView<Int, 1, D> brt_for_possible_rotamer,
      TView<Int, 1, D> brt_for_possible_rotamer_boundaries);

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
      TView<Real, 1, D> rotamer_probability);

  static void determine_n_base_rotamers_to_build(
      TView<Real, 1, D> prob_cumsum_limit_for_buildable_restype,
      TView<Int, 1, D> n_possible_rotamers_per_brt,
      TView<Int, 1, D> brt_for_possible_rotamer,
      TView<Int, 1, D> brt_for_possible_rotamer_boundaries,
      TView<Int, 1, D> possible_rotamer_offset_for_brt,
      TView<Real, 1, D> rotamer_probability,
      TView<Int, 1, D> n_rotamers_to_build_per_brt);

  static Int count_expanded_rotamers(
      TView<Int, 1, D> nchi_for_buildable_restype,
      TView<Int, 2, D> rottable_set_for_buildable_restype,
      TView<Int, 1, D> nchi_for_tableset,
      TView<Int, 2, D> chi_expansion_for_buildable_restype,
      TView<Int, 2, D> non_dunbrack_expansion_counts_for_buildable_restype,
      TView<Int, 1, D> n_expansions_for_brt,
      TView<Int, 2, D> expansion_dim_prods_for_brt,
      TView<Int, 1, D> n_rotamers_to_build_per_brt,
      TView<Int, 1, D> n_rotamers_to_build_per_brt_offsets);

  static void map_from_rotamer_index_to_brt(
      TView<Int, 1, D> n_rotamers_to_build_per_brt_offsets,
      TView<Int, 1, D> brt_for_rotamer);

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
      TView<Real, 2, D> chi_for_rotamers);
};

}  // namespace dunbrack
}  // namespace rotamer
}  // namespace pack
}  // namespace tmol
