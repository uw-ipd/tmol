#include <pybind11/eigen.h>
#include <tmol/utility/tensor/pybind.h>
#include <torch/torch.h>

#include "compiled.hh"

#include <tmol/utility/function_dispatch/pybind.hh>

namespace tmol {
namespace score {
namespace dunbrack {
namespace potentials {

template <tmol::Device Dev, typename Real, typename Int>
void bind_dispatch(pybind11::module& m) {
  using namespace pybind11::literals;
  using namespace tmol::utility::function_dispatch;

  add_dispatch_impl<Dev, Real>(
      m,
      "dunbrack",
      &DunbrackDispatch<Dev, Real, Int>::f,
      "coords"_a,
      "rotameric_prob_tables"_a,
      "rotameric_neglnprob_tables"_a,
      "rotprob_table_sizes"_a,
      "rotprob_table_strides"_a,
      "rotameric_mean_tables"_a,
      "rotameric_sdev_tables"_a,
      "rotmean_table_sizes"_a,
      "rotmean_table_strides"_a,
      "rotameric_bb_start"_a,
      "rotameric_bb_step"_a,
      "rotameric_bb_periodicity"_a,
      "semirotameric_tables"_a,
      "semirot_table_sizes"_a,
      "semirot_table_strides"_a,
      "semirot_start"_a,
      "semirot_step"_a,
      "semirot_periodicity"_a,
      "rotameric_rotind2tableind"_a,
      "semirotameric_rotind2tableind"_a,
      "ndihe_for_res"_a,
      "dihedral_offset_for_res"_a,
      "dihedral_atom_inds"_a,

      "rottable_set_for_res"_a,
      "nchi_for_res"_a,
      "nrotameric_chi_for_res"_a,
      "rotres2resid"_a,
      "prob_table_offset_for_rotresidue"_a,
      "rotind2tableind_offset_for_res"_a,

      "rotmean_table_offset_for_residue"_a,

      "rotameric_chi_desc"_a,
      "semirotameric_chi_desc"_a,

      "dihedrals"_a,
      "ddihe_dxyz"_a,
      //"rotchi_devpen"_a,
      //"ddevpen_dbb"_a,
      "rotameric_rottable_assignment"_a,
      "semirotameric_rottable_assignment"_a);

  add_dispatch_impl<Dev, Real>(
      m,
      "dunbrack_deriv",
      &DunbrackDispatch<Dev, Real, Int>::df,
      "coords"_a,
      "dE_drotnlp"_a,
      "drot_nlp_dbb_xyz"_a,
      "dE_ddevpen"_a,
      "ddevpen_dtor_xyz"_a, 
      "dE_dnonrotnlp"_a,
      "dnonrot_nlp_dtor_xyz"_a,
      "dihedral_offset_for_res"_a,
      "dihedral_atom_inds"_a,
      "rotres2resid"_a,
      "rotameric_chi_desc"_a,
      "semirotameric_chi_desc"_a);

};


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;

  bind_dispatch<tmol::Device::CPU, float, int32_t>(m);
  bind_dispatch<tmol::Device::CPU, double, int32_t>(m);

#ifdef WITH_CUDA
  bind_dispatch<tmol::Device::CUDA, float, int32_t>(m);
  //bind_dispatch<tmol::Device::CUDA, double, int32_t>(m);
#endif
}


}  // namespace potentials
}  // namespace rama
}  // namespace score
}  // namespace tmol
