#include <torch/torch.h>
#include <torch/script.h>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/simple_dispatch.hh>
#include <tmol/score/common/forall_dispatch.hh>
#include <tmol/score/common/device_operations.hh>

#include <tmol/io/details/compiled/gen_pose_leaf_atoms.hh>

namespace tmol {
namespace io {
namespace details {
namespace compiled {

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::tensor_list;

class PoseLeafAtomGen : public torch::autograd::Function<PoseLeafAtomGen> {
 public:
  static Tensor forward(
      AutogradContext* ctx,
      Tensor orig_coords,
      Tensor orig_coords_atom_missing,
      Tensor pose_stack_atom_missing,
      Tensor pose_stack_block_coord_offset,
      Tensor pose_stack_block_type,
      Tensor pose_stack_inter_block_connections,
      Tensor block_type_n_atoms,
      Tensor block_type_atom_downstream_of_conn,
      Tensor block_type_atom_ancestors,
      Tensor block_type_atom_icoors,
      Tensor block_type_atom_ancestors_backup,
      Tensor block_type_atom_icoors_backup) {
    at::Tensor new_coords;

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        orig_coords.type(), "leaf_atom_gen_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result = GeneratePoseLeafAtoms<
              score::common::DeviceOperations,
              Dev,
              Real,
              Int>::
              forward(
                  TCAST(orig_coords),
                  TCAST(orig_coords_atom_missing),
                  TCAST(pose_stack_atom_missing),
                  TCAST(pose_stack_block_coord_offset),
                  TCAST(pose_stack_block_type),
                  TCAST(pose_stack_inter_block_connections),
                  TCAST(block_type_n_atoms),
                  TCAST(block_type_atom_downstream_of_conn),
                  TCAST(block_type_atom_ancestors),
                  TCAST(block_type_atom_icoors),
                  TCAST(block_type_atom_ancestors_backup),
                  TCAST(block_type_atom_icoors_backup));

          new_coords = result.tensor;
        }));

    ctx->save_for_backward(
        {orig_coords,
         new_coords,
         orig_coords_atom_missing,
         pose_stack_atom_missing,
         pose_stack_block_coord_offset,
         pose_stack_block_type,
         pose_stack_inter_block_connections,
         block_type_n_atoms,
         block_type_atom_downstream_of_conn,
         block_type_atom_ancestors,
         block_type_atom_icoors,
         block_type_atom_ancestors_backup,
         block_type_atom_icoors_backup});

    return new_coords;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();

    int i = 0;

    auto orig_coords = saved[i++];
    auto new_coords = saved[i++];
    auto orig_coords_atom_missing = saved[i++];
    auto pose_stack_atom_missing = saved[i++];
    auto pose_stack_block_coord_offset = saved[i++];
    auto pose_stack_block_type = saved[i++];
    auto pose_stack_inter_block_connections = saved[i++];
    auto block_type_n_atoms = saved[i++];
    auto block_type_atom_downstream_of_conn = saved[i++];
    auto block_type_atom_ancestors = saved[i++];
    auto block_type_atom_icoors = saved[i++];
    auto block_type_atom_ancestors_backup = saved[i++];
    auto block_type_atom_icoors_backup = saved[i++];

    at::Tensor dE_d_orig_coords;

    using Int = int32_t;

    auto dE_d_new_coords = grad_outputs[0];

    TMOL_DISPATCH_FLOATING_DEVICE(
        orig_coords.type(), "leaf_atom_gen_backward", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result = GeneratePoseLeafAtoms<
              score::common::DeviceOperations,
              Dev,
              Real,
              Int>::
              backward(
                  TCAST(dE_d_new_coords),
                  TCAST(new_coords),
                  TCAST(orig_coords),
                  TCAST(orig_coords_atom_missing),
                  TCAST(pose_stack_atom_missing),
                  TCAST(pose_stack_block_coord_offset),
                  TCAST(pose_stack_block_type),
                  TCAST(pose_stack_inter_block_connections),
                  TCAST(block_type_n_atoms),
                  TCAST(block_type_atom_downstream_of_conn),
                  TCAST(block_type_atom_ancestors),
                  TCAST(block_type_atom_icoors),
                  TCAST(block_type_atom_ancestors_backup),
                  TCAST(block_type_atom_icoors_backup));
          dE_d_orig_coords = result.tensor;
        }));

    return {
        dE_d_orig_coords,
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
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

Tensor pose_leaf_atom_gen_op(
    Tensor orig_coords,
    Tensor orig_coords_atom_missing,
    Tensor pose_stack_atom_missing,
    Tensor pose_stack_block_coord_offset,
    Tensor pose_stack_block_type,
    Tensor pose_stack_inter_block_connections,
    Tensor block_type_n_atoms,
    Tensor block_type_atom_downstream_of_conn,
    Tensor block_type_atom_ancestors,
    Tensor block_type_atom_icoors,
    Tensor block_type_atom_ancestors_backup,
    Tensor block_type_atom_icoors_backup) {
  return PoseLeafAtomGen::apply(
      orig_coords,
      orig_coords_atom_missing,
      pose_stack_atom_missing,
      pose_stack_block_coord_offset,
      pose_stack_block_type,
      pose_stack_inter_block_connections,
      block_type_n_atoms,
      block_type_atom_downstream_of_conn,
      block_type_atom_ancestors,
      block_type_atom_icoors,
      block_type_atom_ancestors_backup,
      block_type_atom_icoors_backup);
};

// Macro indirection to force TORCH_EXTENSION_NAME macro expansion
// See https://stackoverflow.com/a/3221914
#define TORCH_LIBRARY_(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_(TORCH_EXTENSION_NAME, m) {
  m.def("gen_pose_leaf_atoms", &pose_leaf_atom_gen_op);
}

}  // namespace compiled
}  // namespace details
}  // namespace io
}  // namespace tmol
