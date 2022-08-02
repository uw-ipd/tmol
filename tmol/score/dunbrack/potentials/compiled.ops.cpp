#include <torch/torch.h>
#include <torch/script.h>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/forall_dispatch.hh>

#include "dispatch.hh"

namespace tmol {
namespace score {
namespace dunbrack {
namespace potentials {

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::tensor_list;

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
class DunOp : public Function<DunOp<ScoreDispatch, DispatchMethod>> {
 public:
  static Tensor forward(
      AutogradContext* ctx,
      Tensor coords,
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
      Tensor semirotameric_rottable_assignment) {
    at::Tensor score;
    at::Tensor dneglnprob_rot_dbb_xyz;
    at::Tensor drotchi_devpen_dtor_xyz;
    at::Tensor dneglnprob_nonrot_dtor_xyz;

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        coords.type(), "dun_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result =
              DunbrackDispatch<DispatchMethod, Dev, Real, Int>::forward(
                  TCAST(coords),
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
          dneglnprob_rot_dbb_xyz = std::get<1>(result).tensor;
          drotchi_devpen_dtor_xyz = std::get<2>(result).tensor;
          dneglnprob_nonrot_dtor_xyz = std::get<3>(result).tensor;
        }));

    ctx->save_for_backward({coords,
                            dneglnprob_rot_dbb_xyz,
                            drotchi_devpen_dtor_xyz,
                            dneglnprob_nonrot_dtor_xyz,
                            dihedral_offset_for_res,
                            dihedral_atom_inds,
                            rotres2resid,
                            rotameric_chi_desc,
                            semirotameric_chi_desc});

    return score;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    int i = 0;

    auto coords = saved[i++];
    auto dneglnprob_rot_dbb_xyz = saved[i++];
    auto drotchi_devpen_dtor_xyz = saved[i++];
    auto dneglnprob_nonrot_dtor_xyz = saved[i++];
    auto dihedral_offset_for_res = saved[i++];
    auto dihedral_atom_inds = saved[i++];
    auto rotres2resid = saved[i++];
    auto rotameric_chi_desc = saved[i++];
    auto semirotameric_chi_desc = saved[i++];

    using Int = int32_t;

    at::Tensor dT_dI;
    auto dTdV = grad_outputs[0];

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

          dT_dI = result.tensor;
        }));

    return {
        dT_dI,           torch::Tensor(), torch::Tensor(), torch::Tensor(),
        torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
        torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
        torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
        torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
        torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
        torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
        torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
        torch::Tensor(), torch::Tensor(), torch::Tensor(),
    };
  }
};

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
Tensor dun_op(
    Tensor coords,
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
    Tensor semirotameric_rottable_assignment) {
  return DunOp<ScoreDispatch, DispatchMethod>::apply(
      coords,
      rotameric_neglnprob_tables,
      rotprob_table_sizes,
      rotprob_table_strides,
      rotameric_mean_tables,
      rotameric_sdev_tables,
      rotmean_table_sizes,
      rotmean_table_strides,
      rotameric_bb_start,
      rotameric_bb_step,
      rotameric_bb_periodicity,
      semirotameric_tables,
      semirot_table_sizes,
      semirot_table_strides,
      semirot_start,
      semirot_step,
      semirot_periodicity,
      rotameric_rotind2tableind,
      semirotameric_rotind2tableind,
      ndihe_for_res,
      dihedral_offset_for_res,
      dihedral_atom_inds,
      rottable_set_for_res,
      nchi_for_res,
      nrotameric_chi_for_res,
      rotres2resid,
      prob_table_offset_for_rotresidue,
      rotind2tableind_offset_for_res,
      rotmean_table_offset_for_residue,
      rotameric_chi_desc,
      semirotameric_chi_desc,
      dihedrals,
      ddihe_dxyz,
      rotameric_rottable_assignment,
      semirotameric_rottable_assignment);
}

// Macro indirection to force TORCH_EXTENSION_NAME macro expansion
// See https://stackoverflow.com/a/3221914
#define TORCH_LIBRARY_(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_(TORCH_EXTENSION_NAME, m) {
  m.def("score_dun", &dun_op<DunbrackDispatch, common::ForallDispatch>);
}

}  // namespace potentials
}  // namespace dunbrack
}  // namespace score
}  // namespace tmol
