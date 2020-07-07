#include <torch/script.h>
#include <tmol/utility/autograd.hh>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>
// #include <tmol/score/common/forall_dispatch.hh>
#include <tmol/score/common/complex_dispatch.hh>

#include <vector>

#include "compiled.hh"

namespace tmol {
namespace pack {
namespace rotamer {

using torch::Tensor;


template < template <tmol::Device> class DispatchMethod >
std::vector< Tensor >
dun_sample_chi(
    Tensor coords,

    Tensor rotameric_prob_tables,
    Tensor rotprob_table_sizes,
    Tensor rotprob_table_strides,
    Tensor rotameric_mean_tables,
    Tensor rotameric_sdev_tables,
    Tensor rotmean_table_sizes,
    Tensor rotmean_table_strides,
    Tensor rotameric_meansdev_tableset_offsets,
    Tensor rotameric_bb_start,        // ntable-set entries
    Tensor rotameric_bb_step,         // ntable-set entries
    Tensor rotameric_bb_periodicity,  // ntable-set entries
    Tensor semirotameric_tables,              // n-semirot-tabset
    Tensor semirot_table_sizes,    // n-semirot-tabset
    Tensor semirot_table_strides,  // n-semirot-tabset
    Tensor semirot_start,             // n-semirot-tabset
    Tensor semirot_step,              // n-semirot-tabset
    Tensor semirot_periodicity,       // n-semirot-tabset
    Tensor rotameric_rotind2tableind,
    Tensor semirotameric_rotind2tableind,
    Tensor all_chi_rotind2tableind,
    Tensor all_chi_rotind2tableind_offsets,

    Tensor n_rotamers_for_tableset,
    Tensor n_rotamers_for_tableset_offsets,
    Tensor sorted_rotamer_2_rotamer,
    Tensor nchi_for_tableset,
    Tensor rotwells,

    Tensor ndihe_for_res,               // nres x 1
    Tensor dihedral_offset_for_res,     // nres x 1
    Tensor dihedral_atom_inds,  // ndihe x 4

    Tensor rottable_set_for_buildable_restype,  // n-buildable-restypes x 2
    Tensor chi_expansion_for_buildable_restype,
    Tensor non_dunbrack_expansion_for_buildable_restype,
    Tensor non_dunbrack_expansion_counts_for_buildable_restype,
    Tensor prob_cumsum_limit_for_buildable_restype,
    Tensor nchi_for_buildable_restype
) {
  nvtx_range_push("dunbrack_sample_chi");
  std::cout << "Hit compiled.ops.cpp" << std::endl;

  at::Tensor n_rots_for_brt;
  at::Tensor n_rots_for_brt_offsets;
  at::Tensor brt_for_rotamer;
  at::Tensor chi_for_rotamers;

  using Int = int32_t;

  try {

    TMOL_DISPATCH_FLOATING_DEVICE(
        coords.type(), "dunbrack_sample_chi", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;
	  std::cout << "About to invoke DunbrackChiSampler " << (Dev == tmol::Device::CPU ? "cpu" : "cuda" ) << std::endl;
          auto result = DunbrackChiSampler<DispatchMethod, Dev, Real, Int>::f(
            TCAST(coords),
            //TCAST(res_coord_start_ind),
            TCAST(rotameric_prob_tables),
            TCAST(rotprob_table_sizes),
            TCAST(rotprob_table_strides),
            TCAST(rotameric_mean_tables),
            TCAST(rotameric_sdev_tables),
            TCAST(rotmean_table_sizes),
            TCAST(rotmean_table_strides),
	    TCAST(rotameric_meansdev_tableset_offsets),
            TCAST(rotameric_bb_start),
            TCAST(rotameric_bb_step),
            TCAST(rotameric_bb_periodicity),
            TCAST(semirotameric_tables),
            TCAST(semirot_table_sizes),
            TCAST(semirot_table_strides),
            TCAST(semirot_start),
            TCAST(semirot_step),
            TCAST(semirot_periodicity),
            TCAST(rotameric_rotind2tableind),
            TCAST(semirotameric_rotind2tableind),
            TCAST(all_chi_rotind2tableind),
	    TCAST(all_chi_rotind2tableind_offsets),

	    TCAST(n_rotamers_for_tableset),
	    TCAST(n_rotamers_for_tableset_offsets),
	    TCAST(sorted_rotamer_2_rotamer),
	    TCAST(nchi_for_tableset),
	    TCAST(rotwells),

            TCAST(ndihe_for_res),
            TCAST(dihedral_offset_for_res),
            TCAST(dihedral_atom_inds),
  
            TCAST(rottable_set_for_buildable_restype),
            TCAST(chi_expansion_for_buildable_restype),
            TCAST(non_dunbrack_expansion_for_buildable_restype),
            TCAST(non_dunbrack_expansion_counts_for_buildable_restype),
            TCAST(prob_cumsum_limit_for_buildable_restype),
	    TCAST(nchi_for_buildable_restype)
  
          );
  
          n_rots_for_brt = std::get<0>(result).tensor;
          n_rots_for_brt_offsets = std::get<1>(result).tensor;
	  brt_for_rotamer = std::get<2>(result).tensor;
	  chi_for_rotamers = std::get<3>(result).tensor;
  
  
        }));
  } catch (at::Error err) {
    std::cerr << "caught exception:\n" << err.what_without_backtrace() << std::endl;
    throw err;
  } catch (c10::Error err) {
    std::cerr << "caught exception:\n" << err.what_without_backtrace() << std::endl;
    throw err;
  }

  nvtx_range_pop();

  return {n_rots_for_brt, n_rots_for_brt_offsets,
      brt_for_rotamer, chi_for_rotamers};

  //c10::intrusive_ptr< at::ivalue::TensorList > ret_list;
  try {
    // ret_list = c10::make_intrusive< at::ivalue::TensorList >(
    //   at::ivalue::TensorList({n_rots_for_brt, n_rots_for_brt_offsets,
    //   brt_for_rotamer, chi_for_rotamers}));
      } catch (c10::Error err) {
    std::cerr << "caught: " << err.what_without_backtrace() << std::endl;
  }
};


static auto registry =
    torch::jit::RegisterOperators()
  .op("tmol::dun_sample_chi", &dun_sample_chi<score::common::ComplexDispatch>);


}  // namespace rotamer
}  // namespace pack
}  // namespace tmol
