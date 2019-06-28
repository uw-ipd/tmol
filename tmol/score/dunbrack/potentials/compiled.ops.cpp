#include <torch/script.h>
#include <tmol/utility/autograd.hh>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/forall_dispatch.hh>

#include "dispatch.hh"

namespace tmol {
namespace score {
namespace dunbrack {
namespace potentials {

using torch::Tensor;

template <
    template <
        template <tmol::Device>
        class Dispatch,
        tmol::Device D,
        typename Real,
        typename Int>
    class ScoreDispatch,
    template <tmol::Device>
    class DispatchMethod>
struct ScoreOpBackward : public torch::autograd::Function {
  torch::autograd::SavedVariable saved_coords;
  torch::autograd::SavedVariable saved_dneglnprob_rot_dbb_xyz;
  torch::autograd::SavedVariable saved_drotchi_devpen_dtor_xyz;
  torch::autograd::SavedVariable saved_dneglnprob_nonrot_dtor_xyz;
  torch::autograd::SavedVariable saved_dihedral_offset_for_res;
  torch::autograd::SavedVariable saved_dihedral_atom_inds;
  torch::autograd::SavedVariable saved_rotres2resid;
  torch::autograd::SavedVariable saved_rotameric_chi_desc;
  torch::autograd::SavedVariable saved_semirotameric_chi_desc;

  void release_variables() override {
    saved_coords.reset_data();
    saved_coords.reset_grad_function();
    saved_dneglnprob_rot_dbb_xyz.reset_data();
    saved_dneglnprob_rot_dbb_xyz.reset_grad_function();
    saved_drotchi_devpen_dtor_xyz.reset_data();
    saved_drotchi_devpen_dtor_xyz.reset_grad_function();
    saved_dneglnprob_nonrot_dtor_xyz.reset_data();
    saved_dneglnprob_nonrot_dtor_xyz.reset_grad_function();
    saved_dihedral_offset_for_res.reset_data();
    saved_dihedral_offset_for_res.reset_grad_function();
    saved_dihedral_atom_inds.reset_data();
    saved_dihedral_atom_inds.reset_grad_function();
    saved_rotres2resid.reset_data();
    saved_rotres2resid.reset_grad_function();
    saved_rotameric_chi_desc.reset_data();
    saved_rotameric_chi_desc.reset_grad_function();
    saved_semirotameric_chi_desc.reset_data();
    saved_semirotameric_chi_desc.reset_grad_function();
  }

  ScoreOpBackward(
    torch::autograd::Variable coords,
    torch::autograd::Variable dneglnprob_rot_dbb_xyz,
    torch::autograd::Variable drotchi_devpen_dtor_xyz,
    torch::autograd::Variable dneglnprob_nonrot_dtor_xyz,
    torch::autograd::Variable dihedral_offset_for_res,
    torch::autograd::Variable dihedral_atom_inds,
    torch::autograd::Variable rotres2resid,
    torch::autograd::Variable rotameric_chi_desc,
    torch::autograd::Variable semirotameric_chi_desc
  )   : 
    saved_coords(coords, false), 
    saved_dneglnprob_rot_dbb_xyz(dneglnprob_rot_dbb_xyz, false), 
    saved_drotchi_devpen_dtor_xyz(drotchi_devpen_dtor_xyz, false), 
    saved_dneglnprob_nonrot_dtor_xyz(dneglnprob_nonrot_dtor_xyz, false), 
    saved_dihedral_offset_for_res(dihedral_offset_for_res, false), 
    saved_dihedral_atom_inds(dihedral_atom_inds, false), 
    saved_rotres2resid(rotres2resid, false), 
    saved_rotameric_chi_desc(rotameric_chi_desc, false), 
    saved_semirotameric_chi_desc(semirotameric_chi_desc, false) { }

  torch::autograd::variable_list apply(
      torch::autograd::variable_list&& grads) override {
    auto coords = saved_coords.unpack();
    auto dneglnprob_rot_dbb_xyz = saved_dneglnprob_rot_dbb_xyz.unpack();
    auto drotchi_devpen_dtor_xyz = saved_drotchi_devpen_dtor_xyz.unpack();
    auto dneglnprob_nonrot_dtor_xyz = saved_dneglnprob_nonrot_dtor_xyz.unpack();
    auto dihedral_offset_for_res = saved_dihedral_offset_for_res.unpack();
    auto dihedral_atom_inds = saved_dihedral_atom_inds.unpack();
    auto rotres2resid = saved_rotres2resid.unpack();
    auto rotameric_chi_desc = saved_rotameric_chi_desc.unpack();
    auto semirotameric_chi_desc = saved_semirotameric_chi_desc.unpack();

    at::Tensor dV_dI;
    using Int = int32_t;

    auto dTdV = grads[0];

    TMOL_DISPATCH_FLOATING_DEVICE(
      coords.type(), "ScoreOpBackward", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        auto result = ScoreDispatch<DispatchMethod, Dev, Real, Int>::backward(
            TCAST(dTdV),
            TCAST(coords),
            TCAST(dneglnprob_rot_dbb_xyz),
            TCAST(drotchi_devpen_dtor_xyz),
            TCAST(dneglnprob_nonrot_dtor_xyz),
            TCAST(dihedral_offset_for_res),
            TCAST(dihedral_atom_inds),
            TCAST(rotres2resid),
            TCAST(rotameric_chi_desc),
            TCAST(semirotameric_chi_desc));

        dV_dI = result.tensor;
      }));


    return {dV_dI};
  }
};


template < template <tmol::Device> class DispatchMethod >
Tensor dun_op(
    Tensor coords,
    Tensor rotameric_prob_tables,
    Tensor rotameric_neglnprob_tables,
    Tensor rotprob_table_sizes,
    Tensor rotprob_table_strides,
    Tensor rotameric_mean_tables,
    Tensor rotameric_sdev_tables,
    Tensor rotmean_table_sizes,
    Tensor rotmean_table_strides,
    Tensor rotameric_bb_start,
    Tensor rotameric_bb_step,
    Tensor rotameric_bb_periodicity,
    Tensor semirotameric_tables,
    Tensor semirot_table_sizes,
    Tensor semirot_table_strides,
    Tensor semirot_start,
    Tensor semirot_step,
    Tensor semirot_periodicity,
    Tensor rotameric_rotind2tableind,
    Tensor semirotameric_rotind2tableind,
    Tensor ndihe_for_res,
    Tensor dihedral_offset_for_res,
    Tensor dihedral_atom_inds,
    Tensor rottable_set_for_res,
    Tensor nchi_for_res,
    Tensor nrotameric_chi_for_res,
    Tensor rotres2resid,
    Tensor prob_table_offset_for_rotresidue,
    Tensor rotind2tableind_offset_for_res,
    Tensor rotmean_table_offset_for_residue,
    Tensor rotameric_chi_desc,
    Tensor semirotameric_chi_desc,
    Tensor dihedrals,
    Tensor ddihe_dxyz,
    Tensor rotameric_rottable_assignment,
    Tensor semirotameric_rottable_assignment
) {
  using tmol::utility::connect_backward_pass;
  using tmol::utility::SavedGradsBackward;
  nvtx_range_push("dun_op");

  at::Tensor score;
  at::Tensor neglnprob_rot_tpack;
  at::Tensor dneglnprob_rot_dbb_xyz_tpack;
  at::Tensor rotchi_devpen_tpack;
  at::Tensor drotchi_devpen_dtor_xyz_tpack;
  at::Tensor neglnprob_nonrot_tpack;
  at::Tensor dneglnprob_nonrot_dtor_xyz_tpack;

  using Int = int32_t;

  TMOL_DISPATCH_FLOATING_DEVICE(
      coords.type(), "dun_op", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        auto result = DunbrackDispatch<DispatchMethod, Dev, Real, Int>::forward(
            TCAST(coords),
            TCAST(rotameric_prob_tables),
            TCAST(rotameric_neglnprob_tables),
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
            TCAST(ndihe_for_res),
            TCAST(dihedral_offset_for_res),
            TCAST(dihedral_atom_inds),
            TCAST(rottable_set_for_res),
            TCAST(nchi_for_res),
            TCAST(nrotameric_chi_for_res),
            TCAST(rotres2resid),
            TCAST(prob_table_offset_for_rotresidue),
            TCAST(rotind2tableind_offset_for_res),
            TCAST(rotmean_table_offset_for_residue),
            TCAST(rotameric_chi_desc),
            TCAST(semirotameric_chi_desc),
            TCAST(dihedrals),
            TCAST(ddihe_dxyz),
            TCAST(rotameric_rottable_assignment),
            TCAST(semirotameric_rottable_assignment));

        score = std::get<0>(result).tensor;

        // save for backwards
        dneglnprob_rot_dbb_xyz_tpack = std::get<1>(result).tensor;
        drotchi_devpen_dtor_xyz_tpack = std::get<2>(result).tensor;
        dneglnprob_nonrot_dtor_xyz_tpack = std::get<3>(result).tensor;
      }));


  auto backward_op = connect_backward_pass({
        dneglnprob_rot_dbb_xyz_tpack,
        drotchi_devpen_dtor_xyz_tpack,
        dneglnprob_nonrot_dtor_xyz_tpack,
        dihedral_offset_for_res,
        dihedral_atom_inds,
        rotres2resid,
        rotameric_chi_desc,
        semirotameric_chi_desc
    }, score, [&]() {
      return std::shared_ptr<ScoreOpBackward<DunbrackDispatch, common::ForallDispatch>>(
        new ScoreOpBackward<DunbrackDispatch, common::ForallDispatch>( 
            coords,
            dneglnprob_rot_dbb_xyz_tpack,
            drotchi_devpen_dtor_xyz_tpack,
            dneglnprob_nonrot_dtor_xyz_tpack,
            dihedral_offset_for_res,
            dihedral_atom_inds,
            rotres2resid,
            rotameric_chi_desc,
            semirotameric_chi_desc
        ), 
        torch::autograd::deleteFunction);
  });


  nvtx_range_pop();
  return backward_op;
};


static auto registry =
    torch::jit::RegisterOperators()
        .op("tmol::score_dun", &dun_op<common::ForallDispatch>);


}  // namespace potentials
}  // namespace dun
}  // namespace score
}  // namespace tmol
