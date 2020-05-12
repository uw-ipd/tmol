#include <torch/script.h>
#include <tmol/utility/autograd.hh>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/forall_dispatch.hh>

#include "compiled.h"

namespace tmol {
namespace pack {
namespace rotamer {

using torch::Tensor;


template < template <tmol::Device> class DispatchMethod >
//torch::autograd::variable_list
//std::tuple< Tensor, Tensor >
//c10::intrusive_ptr< at::ivalue::TensorList >
Tensor
dun_sample_chi(
    Tensor coords,
    //Tensor res_coord_start_ind,

    Tensor rotameric_prob_tables,
    Tensor rotprob_table_sizes,
    Tensor rotprob_table_strides,
    Tensor rotameric_mean_tables,
    Tensor rotameric_sdev_tables,
    Tensor rotmean_table_sizes,
    Tensor rotmean_table_strides,
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

    Tensor n_rotamers_for_tableset,
    Tensor n_rotamers_for_tableset_offsets,
    Tensor sorted_rotamer_2_rotamer,

    Tensor ndihe_for_res,               // nres x 1
    Tensor dihedral_offset_for_res,     // nres x 1
    Tensor dihedral_atom_inds,  // ndihe x 4

    Tensor rottable_set_for_buildable_restype,  // n-buildable-restypes x 2
    Tensor chi_expansion_for_buildable_restype,
    Tensor non_dunbrack_expansion_for_buildable_restype,
    Tensor non_dunbrack_expansion_counts_for_buildable_restype,
    Tensor prob_cumsum_limit_for_buildable_restype,

    // ?? Tensor nrotameric_chi_for_res,            // nres x 1
    // ?? Tensor rotres2resid,                      // nres x 1
    // ?? Tensor prob_table_offset_for_rotresidue,  // n-rotameric-res x 1
    // ?? Tensor rotind2tableind_offset_for_res,    // n-res x 1

    // ?? Tensor rotmean_table_offset_for_residue,  // n-res x 1

    // ?? Tensor rotameric_chi_desc,  // n-rotameric-chi x 2
    // ?? Tensor semirotameric_chi_desc,  // n-semirotameric-residues x OD4

    Tensor dihedrals                        // ndihe x 1
    // ?? Tensor ddihe_dxyz,  // ndihe x 3
    // ?? Tensor rotameric_rottable_assignment,     // nres x 1
    // ?? Tensor semirotameric_rottable_assignment  // nres x 1
) {
  nvtx_range_push("dunbrack_sample_chi");
  std::cout << "Hit compiled.ops.cpp" << std::endl;

  at::Tensor ret1;
  at::Tensor ret2;

  using Int = int32_t;

  try {

    TMOL_DISPATCH_FLOATING_DEVICE(
        coords.type(), "dunbrack_sample_chi", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;
          std::cout << "Calling dunbrack chi sampler " << sizeof(Real) << " " << sizeof(Int) << std::endl;
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

	    TCAST(n_rotamers_for_tableset),
	    TCAST(n_rotamers_for_tableset_offsets),
	    TCAST(sorted_rotamer_2_rotamer),
  
            TCAST(ndihe_for_res),
            TCAST(dihedral_offset_for_res),
            TCAST(dihedral_atom_inds),
  
            TCAST(rottable_set_for_buildable_restype),
            TCAST(chi_expansion_for_buildable_restype),
            TCAST(non_dunbrack_expansion_for_buildable_restype),
            TCAST(non_dunbrack_expansion_counts_for_buildable_restype),
            TCAST(prob_cumsum_limit_for_buildable_restype),
  
            // ?? TCAST(nrotameric_chi_for_res),
            // ?? TCAST(rotres2resid),
            // ?? TCAST(prob_table_offset_for_rotresidue),
            // ?? TCAST(rotind2tableind_offset_for_res),
            // ?? TCAST(rotmean_table_offset_for_residue),
            // ?? TCAST(rotameric_chi_desc),
            // ?? TCAST(semirotameric_chi_desc),
            TCAST(dihedrals)
            // ?? TCAST(ddihe_dxyz),
            // ?? TCAST(rotameric_rottable_assignment),
            // ?? TCAST(semirotameric_rottable_assignment)
          );
  
          ret1 = std::get<0>(result).tensor;
          ret2 = std::get<0>(result).tensor;
  
  
        }));
  } catch (at::Error err) {
    std::cerr << "caught exception:\n" << err.what_without_backtrace() << std::endl;
    throw err;
  }

  nvtx_range_pop();

  auto ret_list = c10::make_intrusive< at::ivalue::TensorList >(at::ivalue::TensorList({ret1, ret2}));
  //return ret_list;
  return dihedrals;
};


static auto registry =
    torch::jit::RegisterOperators()
  .op("tmol::dun_sample_chi", &dun_sample_chi<score::common::ForallDispatch>);


}  // namespace rotamer
}  // namespace pack
}  // namespace tmol
