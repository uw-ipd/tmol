#include <pybind11/eigen.h>
#include <tmol/utility/tensor/pybind.h>
#include <torch/extension.h>

#include <tmol/score/common/complex_dispatch.hh>
#include <tmol/pack/rotamer/dunbrack/compiled.hh>

#include <tmol/tests/pack/rotamer/dunbrack/test.hh>

namespace tmol {

template <tmol::Device D, typename Real, typename Int>
void
bind_dispatch(pybind11::module & m)
{
  using namespace pybind11::literals;
  m.def(
      "determine_n_possible_rots",
      &DunbrackChiSamplerTester<
          tmol::score::common::ComplexDispatch,
          D,
          Real,
          Int>::determine_n_possible_rots,
      "rottable_set_for_buildable_restype"_a,
      "n_rotamers_for_tableset"_a,
      "n_possible_rotamers_per_brt"_a);

  m.def(
      "fill_in_brt_for_possrots",
      &DunbrackChiSamplerTester<
          tmol::score::common::ComplexDispatch,
          D,
          Real,
          Int>::fill_in_brt_for_possrots,
    "possible_rotamer_offset_for_brt"_a,
    "brt_for_possible_rotamer"_a);

  m.def(
      "interpolate_probabilities_for_possible_rotamers",
      &DunbrackChiSamplerTester<
          tmol::score::common::ComplexDispatch,
          D,
          Real,
          Int>::interpolate_probabilities_for_possible_rotamers,
    "rotameric_prob_tables"_a,
    "rotprob_table_sizes"_a,
    "rotprob_table_strides"_a,
    "rotameric_bb_start"_a,
    "rotameric_bb_step"_a,
    "rotameric_bb_periodicity"_a,
    "n_rotamers_for_tableset_offsets"_a,
    "sorted_rotamer_2_rotamer"_a,
    "rottable_set_for_buildable_restype"_a,
    "brt_for_possible_rotamer"_a,
    "possible_rotamer_offset_for_brt"_a,
    "backbone_dihedrals"_a,
    "rotamer_probability"_a);

  m.def(
      "determine_n_base_rotamers_to_build",
      &DunbrackChiSamplerTester<
          tmol::score::common::ComplexDispatch,
          D,
          Real,
          Int>::determine_n_base_rotamers_to_build,
    "prob_cumsum_limit_for_buildable_restype"_a,
    "n_possible_rotamers_per_brt"_a,
    "brt_for_possible_rotamer"_a,
    "possible_rotamer_offset_for_brt"_a,
    "rotamer_probability"_a,
    "n_rotamers_to_build_per_brt"_a);
  
  m.def(
      "count_expanded_rotamers",
      &DunbrackChiSamplerTester<
          tmol::score::common::ComplexDispatch,
          D,
          Real,
          Int>::count_expanded_rotamers,
    "nchi_for_buildable_restype"_a,
    "rottable_set_for_buildable_restype"_a,
    "nchi_for_tableset"_a,
    "chi_expansion_for_buildable_restype"_a,
    "non_dunbrack_expansion_counts_for_buildable_restype"_a,
    "n_expansions_for_brt"_a,
    "expansion_dim_prods_for_brt"_a,
    "n_rotamers_to_build_per_brt"_a,
    "n_rotamers_to_build_per_brt_offsets"_a);

  m.def(
      "map_from_rotamer_index_to_brt",
      &DunbrackChiSamplerTester<
          tmol::score::common::ComplexDispatch,
          D,
          Real,
          Int>::map_from_rotamer_index_to_brt,
      "n_rotamers_to_build_per_brt_offsets"_a,
      "brt_for_rotamer"_a);

  m.def(
      "sample_chi_for_rotamers",
      &DunbrackChiSamplerTester<
          tmol::score::common::ComplexDispatch,
          D,
          Real,
          Int>::sample_chi_for_rotamers,
    "rotameric_mean_tables"_a,
    "rotameric_sdev_tables"_a,
    "rotmean_table_sizes"_a,
    "rotmean_table_strides"_a,
    "rotameric_meansdev_tableset_offsets"_a,
    "rotameric_bb_start"_a,
    "rotameric_bb_step"_a,
    "rotameric_bb_periodicity"_a,

    "sorted_rotamer_2_rotamer"_a,
    "nchi_for_tableset"_a,

    "rottable_set_for_buildable_restype"_a,
    "chi_expansion_for_buildable_restype"_a,
    "non_dunbrack_expansion_for_buildable_restype"_a,
    "nchi_for_buildable_restype"_a,

    "backbone_dihedrals"_a,

    "n_rotamers_to_build_per_brt_offsets"_a,
    "brt_for_rotamer"_a,
    "n_expansions_for_brt"_a,
    "n_rotamers_for_tableset_offsets"_a,

    "expansion_dim_prods_for_brt"_a,
    "chi_for_rotamers"_a);

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;
  bind_dispatch<Device::CPU, float, int32_t>(m);
  
#ifdef WITH_CUDA
  bind_dispatch<Device::CUDA, float, int32_t>(m);
#endif
}

}  // namespace tmol
