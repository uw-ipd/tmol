#include <torch/torch.h>
#include <torch/script.h>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/simple_dispatch.hh>
#include <tmol/score/common/forall_dispatch.hh>
#include <tmol/score/common/device_operations.hh>

// #include "dispatch.hh"
#include <tmol/io/details/compiled/gen_pose_hydrogens.hh>
// #include "rotamer_pair_energy_lkball.hh"
// #include "lk_ball_pose_score.hh"

namespace tmol {
namespace io {
namespace details {
namespace compiled {

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::tensor_list;

class PoseHydrogenGen : public torch::autograd::Function<PoseHydrogenGen> {
 public:
  static Tensor forward(
      AutogradContext* ctx,
      Tensor pose_coords,
      Tensor h_coords_missing,
      Tensor pose_stack_block_coord_offset,
      Tensor pose_stack_block_type,
      Tensor pose_stack_inter_residue_connections,
      Tensor block_type_n_atoms,
      Tensor block_type_atom_downstream_of_conn,
      Tensor block_type_atom_ancestors,
      Tensor block_type_atom_icoors) {
    at::Tensor new_coords;

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        pose_coords.type(), "hydrogen_gen_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result =
              GeneratePoseHydrogens<common::DeviceOperations, Dev, Real, Int>::
                  forward(
                      TCAST(pose_coords),
                      TCAST(h_coords_missing),
                      TCAST(pose_stack_block_coord_offset),
                      TCAST(pose_stack_block_type),
                      TCAST(pose_stack_inter_residue_connections),
                      TCAST(block_type_n_atoms),
                      TCAST(block_type_atom_downstream_of_conn),
                      TCAST(block_type_atom_ancestors),
                      TCAST(block_type_atom_icoors));

          new_coords = result.tensor;
        }));

    ctx->save_for_backward({pose_coords,
                            h_coords_missing,
                            pose_stack_block_coord_offset,
                            pose_stack_block_type,
                            pose_stack_inter_residue_connections,
                            block_type_n_atoms,
                            block_type_atom_downstream_of_conn,
                            block_type_atom_ancestors,
                            block_type_atom_icoors});

    return new_coords;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();

    int i = 0;

    auto pose_coords = saved[i++];
    auto h_coords_missing = saved[i++];
    auto pose_stack_block_coord_offset = saved[i++];
    auto pose_stack_block_type = saved[i++];
    auto pose_stack_inter_residue_connections = saved[i++];
    auto block_type_n_atoms = saved[i++];
    auto block_type_atom_downstream_of_conn = saved[i++];
    auto block_type_atom_ancestors = saved[i++];
    auto block_type_atom_icoors = saved[i++];

    at::Tensor dE_d_orig_coords;

    using Int = int32_t;

    auto dE_d_new_coords = grad_outputs[0];

    TMOL_DISPATCH_FLOATING_DEVICE(
        pose_coords.type(), "hydrogen_gen_backward", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result =
              GeneratePoseHydrogens<common::DeviceOperations, Dev, Real, Int>::
                  backward(
                      TCAST(dE_d_new_coords),
                      TCAST(pose_coords),
                      TCAST(h_coords_missing),
                      TCAST(pose_stack_block_coord_offset),
                      TCAST(pose_stack_block_type),
                      TCAST(pose_stack_inter_residue_connections),
                      TCAST(block_type_n_atoms),
                      TCAST(block_type_atom_downstream_of_conn),
                      TCAST(block_type_atom_ancestors),
                      TCAST(block_type_atom_icoors));
          dE_d_orig_coords = result.tensor;
        }));

    return {dE_d_orig_coords,
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),

            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor()};
  };
};

Tensor pose_hgen_op(
    Tensor pose_coords,
    Tensor h_coords_missing,
    Tensor pose_stack_block_coord_offset,
    Tensor pose_stack_block_type,
    Tensor pose_stack_inter_residue_connections,
    Tensor block_type_n_atoms,
    Tensor block_type_atom_downstream_of_conn,
    Tensor block_type_atom_ancestors,
    Tensor block_type_atom_icoors) {
  return PoseHydrogenGen::apply(
      Tensor pose_coords,
      Tensor h_coords_missing,
      Tensor pose_stack_block_coord_offset,
      Tensor pose_stack_block_type,
      Tensor pose_stack_inter_residue_connections,
      Tensor block_type_n_atoms,
      Tensor block_type_atom_downstream_of_conn,
      Tensor block_type_atom_ancestors,
      Tensor block_type_atom_icoors);
};

// Macro indirection to force TORCH_EXTENSION_NAME macro expansion
// See https://stackoverflow.com/a/3221914
#define TORCH_LIBRARY_(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_(TORCH_EXTENSION_NAME, m) {
  m.def("gen_pose_hydrogens", &pose_hgen_op);
}

}  // namespace compiled
}  // namespace details
}  // namespace io
}  // namespace tmol
