#include <torch/torch.h>
#include <torch/script.h>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/device_operations.hh>
#include <tmol/score/common/forall_dispatch.hh>

#include <pybind11/pybind11.h>

#include "params.hh"
#include "genbonded_pose_score.hh"

namespace tmol {
namespace score {
namespace genbonded {
namespace potentials {

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::tensor_list;

using namespace tmol::score::common;

// ---------------------------------------------------------------------------
// Pose-level scoring autograd op
// ---------------------------------------------------------------------------
template <template <tmol::Device> class DispatchMethod>
class GenBondedPoseScoreOp
    : public torch::autograd::Function<GenBondedPoseScoreOp<DispatchMethod>> {
 public:
  static std::vector<Tensor> forward(
      AutogradContext* ctx,
      // Standard rotamer-layout params (common to all 2-body terms)
      Tensor rot_coords,
      Tensor rot_coord_offset,
      Tensor pose_ind_for_atom,
      Tensor first_rot_for_block,
      Tensor first_rot_block_type,
      Tensor block_ind_for_rot,
      Tensor pose_ind_for_rot,
      Tensor block_type_ind_for_rot,
      Tensor n_rots_for_pose,
      Tensor rot_offset_for_pose,
      Tensor n_rots_for_block,
      Tensor rot_offset_for_block,
      int64_t max_n_rots_per_pose,
      // Term-specific params (from get_score_term_attributes)
      Tensor pose_stack_inter_block_connections,
      Tensor atom_paths_from_conn,
      // Combined intra-block subgraphs (proper + improper, tagged)
      Tensor gen_intra_subgraphs,
      Tensor gen_intra_subgraph_offsets,
      Tensor gen_intra_params,
      // Inter-block torsions (hash table + bond types)
      Tensor gen_atom_type_hierarchy,
      Tensor gen_connection_bond_types,
      Tensor gen_inter_torsion_hash_keys,
      Tensor gen_inter_torsion_hash_values,
      bool output_block_pair_energies) {
    at::Tensor score;
    at::Tensor dscore_dcoords;

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        rot_coords.options(), "genbonded_pose_score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result =
              GenBondedPoseScoreDispatch<DispatchMethod, Dev, Real, Int>::
                  forward(
                      TCAST(rot_coords),
                      TCAST(rot_coord_offset),
                      TCAST(pose_ind_for_atom),
                      TCAST(first_rot_for_block),
                      TCAST(first_rot_block_type),
                      TCAST(block_ind_for_rot),
                      TCAST(pose_ind_for_rot),
                      TCAST(block_type_ind_for_rot),
                      TCAST(n_rots_for_pose),
                      TCAST(rot_offset_for_pose),
                      TCAST(n_rots_for_block),
                      TCAST(rot_offset_for_block),
                      max_n_rots_per_pose,

                      TCAST(pose_stack_inter_block_connections),
                      TCAST(atom_paths_from_conn),
                      TCAST(gen_intra_subgraphs),
                      TCAST(gen_intra_subgraph_offsets),
                      TCAST(gen_intra_params),
                      TCAST(gen_atom_type_hierarchy),
                      TCAST(gen_connection_bond_types),
                      TCAST(gen_inter_torsion_hash_keys),
                      TCAST(gen_inter_torsion_hash_values),

                      output_block_pair_energies,
                      rot_coords.requires_grad());

          score = std::get<0>(result).tensor;
          dscore_dcoords = std::get<1>(result).tensor;
        }));

    if (output_block_pair_energies) {
      auto max_n_rots_per_pose_tp =
          TPack<Int, 1, tmol::Device::CPU>::full(1, max_n_rots_per_pose);
      ctx->save_for_backward(
          {rot_coords,
           rot_coord_offset,
           pose_ind_for_atom,
           first_rot_for_block,
           first_rot_block_type,
           block_ind_for_rot,
           pose_ind_for_rot,
           block_type_ind_for_rot,
           n_rots_for_pose,
           rot_offset_for_pose,
           n_rots_for_block,
           rot_offset_for_block,
           max_n_rots_per_pose_tp.tensor,

           pose_stack_inter_block_connections,
           atom_paths_from_conn,
           gen_intra_subgraphs,
           gen_intra_subgraph_offsets,
           gen_intra_params,
           gen_atom_type_hierarchy,
           gen_connection_bond_types,
           gen_inter_torsion_hash_keys,
           gen_inter_torsion_hash_values});
    } else {
      score = score.squeeze(-1).squeeze(-1);
      ctx->save_for_backward({dscore_dcoords, pose_ind_for_atom});
    }
    return {score, dscore_dcoords};
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    at::Tensor dV_d_pose_coords;

    if (saved.size() == 2) {
      // Single-score mode: reuse the pre-computed gradient
      auto saved_grad = saved[0];
      auto pose_ind_for_atom = saved[1];
      auto atom_ingrads = grad_outputs[0].index_select(1, pose_ind_for_atom);
      while (atom_ingrads.dim() < saved_grad.dim()) {
        atom_ingrads = atom_ingrads.unsqueeze(-1);
      }
      dV_d_pose_coords = saved_grad * atom_ingrads;
    } else {
      // Block-pair mode: re-run backward dispatch
      using Int = int32_t;
      int i = 0;
      auto rot_coords = saved[i++];
      auto rot_coord_offset = saved[i++];
      auto pose_ind_for_atom = saved[i++];
      auto first_rot_for_block = saved[i++];
      auto first_rot_block_type = saved[i++];
      auto block_ind_for_rot = saved[i++];
      auto pose_ind_for_rot = saved[i++];
      auto block_type_ind_for_rot = saved[i++];
      auto n_rots_for_pose = saved[i++];
      auto rot_offset_for_pose = saved[i++];
      auto n_rots_for_block = saved[i++];
      auto rot_offset_for_block = saved[i++];
      auto max_n_rots_per_pose =
          TPack<int32_t, 1, tmol::Device::CPU>(saved[i++]).view[0];

      auto pose_stack_inter_block_connections = saved[i++];
      auto atom_paths_from_conn = saved[i++];
      auto gen_intra_subgraphs = saved[i++];
      auto gen_intra_subgraph_offsets = saved[i++];
      auto gen_intra_params = saved[i++];
      auto gen_atom_type_hierarchy = saved[i++];
      auto gen_connection_bond_types = saved[i++];
      auto gen_inter_torsion_hash_keys = saved[i++];
      auto gen_inter_torsion_hash_values = saved[i++];

      TMOL_DISPATCH_FLOATING_DEVICE(
          rot_coords.options(), "genbonded_pose_score_backward_op", ([&] {
            using Real = scalar_t;
            constexpr tmol::Device Dev = device_t;

            auto result =
                GenBondedPoseScoreDispatch<DispatchMethod, Dev, Real, Int>::
                    backward(
                        TCAST(rot_coords),
                        TCAST(rot_coord_offset),
                        TCAST(pose_ind_for_atom),
                        TCAST(first_rot_for_block),
                        TCAST(first_rot_block_type),
                        TCAST(block_ind_for_rot),
                        TCAST(pose_ind_for_rot),
                        TCAST(block_type_ind_for_rot),
                        TCAST(n_rots_for_pose),
                        TCAST(rot_offset_for_pose),
                        TCAST(n_rots_for_block),
                        TCAST(rot_offset_for_block),
                        max_n_rots_per_pose,

                        TCAST(pose_stack_inter_block_connections),
                        TCAST(atom_paths_from_conn),
                        TCAST(gen_intra_subgraphs),
                        TCAST(gen_intra_subgraph_offsets),
                        TCAST(gen_intra_params),
                        TCAST(gen_atom_type_hierarchy),
                        TCAST(gen_connection_bond_types),
                        TCAST(gen_inter_torsion_hash_keys),
                        TCAST(gen_inter_torsion_hash_values),

                        TCAST(grad_outputs[0]));
            dV_d_pose_coords = result.tensor;
          }));
    }

    // Return one gradient per forward() parameter (None for non-tensor args).
    // Parameters: rot_coords, rot_coord_offset, pose_ind_for_atom,
    //   first_rot_for_block, first_rot_block_type, block_ind_for_rot,
    //   pose_ind_for_rot, block_type_ind_for_rot, n_rots_for_pose,
    //   rot_offset_for_pose, n_rots_for_block, rot_offset_for_block,
    //   max_n_rots_per_pose (int64),
    //   pose_stack_inter_block_connections, atom_paths_from_conn,
    //   gen_intra_subgraphs, gen_intra_subgraph_offsets, gen_intra_params,
    //   gen_atom_type_hierarchy, gen_connection_bond_types,
    //   gen_inter_torsion_hash_keys, gen_inter_torsion_hash_values,
    //   output_block_pair_energies (bool)
    return {
        dV_d_pose_coords,
        torch::Tensor(),  // rot_coord_offset
        torch::Tensor(),  // pose_ind_for_atom
        torch::Tensor(),  // first_rot_for_block
        torch::Tensor(),  // first_rot_block_type
        torch::Tensor(),  // block_ind_for_rot
        torch::Tensor(),  // pose_ind_for_rot
        torch::Tensor(),  // block_type_ind_for_rot
        torch::Tensor(),  // n_rots_for_pose
        torch::Tensor(),  // rot_offset_for_pose
        torch::Tensor(),  // n_rots_for_block
        torch::Tensor(),  // rot_offset_for_block
        torch::Tensor(),  // max_n_rots_per_pose (int)
        torch::Tensor(),  // pose_stack_inter_block_connections
        torch::Tensor(),  // atom_paths_from_conn
        torch::Tensor(),  // gen_intra_subgraphs
        torch::Tensor(),  // gen_intra_subgraph_offsets
        torch::Tensor(),  // gen_intra_params
        torch::Tensor(),  // gen_atom_type_hierarchy
        torch::Tensor(),  // gen_connection_bond_types
        torch::Tensor(),  // gen_inter_torsion_hash_keys
        torch::Tensor(),  // gen_inter_torsion_hash_values
        torch::Tensor(),  // output_block_pair_energies (bool)
    };
  }
};

// ---------------------------------------------------------------------------
// Rotamer-level scoring autograd op
// ---------------------------------------------------------------------------
template <template <tmol::Device> class DispatchMethod>
class GenBondedRotamerScoreOp : public torch::autograd::Function<
                                    GenBondedRotamerScoreOp<DispatchMethod>> {
 public:
  static std::vector<Tensor> forward(
      AutogradContext* ctx,
      Tensor rot_coords,
      Tensor rot_coord_offset,
      Tensor pose_ind_for_atom,
      Tensor first_rot_for_block,
      Tensor first_rot_block_type,
      Tensor block_ind_for_rot,
      Tensor pose_ind_for_rot,
      Tensor block_type_ind_for_rot,
      Tensor n_rots_for_pose,
      Tensor rot_offset_for_pose,
      Tensor n_rots_for_block,
      Tensor rot_offset_for_block,
      int64_t max_n_rots_per_pose,

      Tensor pose_stack_inter_block_connections,
      Tensor atom_paths_from_conn,
      Tensor gen_intra_subgraphs,
      Tensor gen_intra_subgraph_offsets,
      Tensor gen_intra_params,
      Tensor gen_atom_type_hierarchy,
      Tensor gen_connection_bond_types,
      Tensor gen_inter_torsion_hash_keys,
      Tensor gen_inter_torsion_hash_values,

      bool output_block_pair_energies) {
    at::Tensor score;
    at::Tensor dscore_dcoords;
    at::Tensor dispatch_indices;
    at::Tensor n_output_intxns;
    at::Tensor rotconn_for_output;

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        rot_coords.options(), "genbonded_rotamer_score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result =
              GenBondedRotamerScoreDispatch<DispatchMethod, Dev, Real, Int>::
                  forward(
                      TCAST(rot_coords),
                      TCAST(rot_coord_offset),
                      TCAST(pose_ind_for_atom),
                      TCAST(first_rot_for_block),
                      TCAST(first_rot_block_type),
                      TCAST(block_ind_for_rot),
                      TCAST(pose_ind_for_rot),
                      TCAST(block_type_ind_for_rot),
                      TCAST(n_rots_for_pose),
                      TCAST(rot_offset_for_pose),
                      TCAST(n_rots_for_block),
                      TCAST(rot_offset_for_block),
                      max_n_rots_per_pose,

                      TCAST(pose_stack_inter_block_connections),
                      TCAST(atom_paths_from_conn),
                      TCAST(gen_intra_subgraphs),
                      TCAST(gen_intra_subgraph_offsets),
                      TCAST(gen_intra_params),
                      TCAST(gen_atom_type_hierarchy),
                      TCAST(gen_connection_bond_types),
                      TCAST(gen_inter_torsion_hash_keys),
                      TCAST(gen_inter_torsion_hash_values),

                      output_block_pair_energies,
                      rot_coords.requires_grad());

          score = std::get<0>(result).tensor;
          dscore_dcoords = std::get<1>(result).tensor;
          dispatch_indices = std::get<2>(result).tensor;
          n_output_intxns = std::get<3>(result).tensor;
          rotconn_for_output = std::get<4>(result).tensor;
        }));

    auto max_n_rots_per_pose_tp =
        TPack<Int, 1, tmol::Device::CPU>::full(1, max_n_rots_per_pose);
    ctx->save_for_backward(
        {rot_coords,
         rot_coord_offset,
         pose_ind_for_atom,
         first_rot_for_block,
         first_rot_block_type,
         block_ind_for_rot,
         pose_ind_for_rot,
         block_type_ind_for_rot,
         n_rots_for_pose,
         rot_offset_for_pose,
         n_rots_for_block,
         rot_offset_for_block,
         max_n_rots_per_pose_tp.tensor,

         pose_stack_inter_block_connections,
         atom_paths_from_conn,
         gen_intra_subgraphs,
         gen_intra_subgraph_offsets,
         gen_intra_params,
         gen_atom_type_hierarchy,
         gen_connection_bond_types,
         gen_inter_torsion_hash_keys,
         gen_inter_torsion_hash_values,

         dispatch_indices,
         n_output_intxns,
         rotconn_for_output});

    return {score, dispatch_indices};
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    at::Tensor dV_d_pose_coords;
    using Int = int32_t;

    int i = 0;
    auto rot_coords = saved[i++];
    auto rot_coord_offset = saved[i++];
    auto pose_ind_for_atom = saved[i++];
    auto first_rot_for_block = saved[i++];
    auto first_rot_block_type = saved[i++];
    auto block_ind_for_rot = saved[i++];
    auto pose_ind_for_rot = saved[i++];
    auto block_type_ind_for_rot = saved[i++];
    auto n_rots_for_pose = saved[i++];
    auto rot_offset_for_pose = saved[i++];
    auto n_rots_for_block = saved[i++];
    auto rot_offset_for_block = saved[i++];
    auto max_n_rots_per_pose =
        TPack<int32_t, 1, tmol::Device::CPU>(saved[i++]).view[0];

    auto pose_stack_inter_block_connections = saved[i++];
    auto atom_paths_from_conn = saved[i++];
    auto gen_intra_subgraphs = saved[i++];
    auto gen_intra_subgraph_offsets = saved[i++];
    auto gen_intra_params = saved[i++];
    auto gen_atom_type_hierarchy = saved[i++];
    auto gen_connection_bond_types = saved[i++];
    auto gen_inter_torsion_hash_keys = saved[i++];
    auto gen_inter_torsion_hash_values = saved[i++];
    auto dispatch_indices = saved[i++];
    auto n_output_intxns = saved[i++];
    auto rotconn_for_output = saved[i++];

    TMOL_DISPATCH_FLOATING_DEVICE(
        rot_coords.options(), "genbonded_rotamer_score_backward_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result =
              GenBondedRotamerScoreDispatch<DispatchMethod, Dev, Real, Int>::
                  backward(
                      TCAST(rot_coords),
                      TCAST(rot_coord_offset),
                      TCAST(pose_ind_for_atom),
                      TCAST(first_rot_for_block),
                      TCAST(first_rot_block_type),
                      TCAST(block_ind_for_rot),
                      TCAST(pose_ind_for_rot),
                      TCAST(block_type_ind_for_rot),
                      TCAST(n_rots_for_pose),
                      TCAST(rot_offset_for_pose),
                      TCAST(n_rots_for_block),
                      TCAST(rot_offset_for_block),
                      max_n_rots_per_pose,

                      TCAST(pose_stack_inter_block_connections),
                      TCAST(atom_paths_from_conn),
                      TCAST(gen_intra_subgraphs),
                      TCAST(gen_intra_subgraph_offsets),
                      TCAST(gen_intra_params),
                      TCAST(gen_atom_type_hierarchy),
                      TCAST(gen_connection_bond_types),
                      TCAST(gen_inter_torsion_hash_keys),
                      TCAST(gen_inter_torsion_hash_values),

                      TCAST(dispatch_indices),
                      TCAST(n_output_intxns),
                      TCAST(rotconn_for_output),
                      TCAST(grad_outputs[0]));
          dV_d_pose_coords = result.tensor;
        }));

    // One None per forward() parameter.
    return {
        dV_d_pose_coords,
        torch::Tensor(),  // rot_coord_offset
        torch::Tensor(),  // pose_ind_for_atom
        torch::Tensor(),  // first_rot_for_block
        torch::Tensor(),  // first_rot_block_type
        torch::Tensor(),  // block_ind_for_rot
        torch::Tensor(),  // pose_ind_for_rot
        torch::Tensor(),  // block_type_ind_for_rot
        torch::Tensor(),  // n_rots_for_pose
        torch::Tensor(),  // rot_offset_for_pose
        torch::Tensor(),  // n_rots_for_block
        torch::Tensor(),  // rot_offset_for_block
        torch::Tensor(),  // max_n_rots_per_pose (int)
        torch::Tensor(),  // pose_stack_inter_block_connections
        torch::Tensor(),  // atom_paths_from_conn
        torch::Tensor(),  // gen_intra_subgraphs
        torch::Tensor(),  // gen_intra_subgraph_offsets
        torch::Tensor(),  // gen_intra_params
        torch::Tensor(),  // gen_atom_type_hierarchy
        torch::Tensor(),  // gen_connection_bond_types
        torch::Tensor(),  // gen_inter_torsion_hash_keys
        torch::Tensor(),  // gen_inter_torsion_hash_values
        torch::Tensor(),  // output_block_pair_energies (bool)
    };
  }
};

// ---------------------------------------------------------------------------
// Free-function wrappers (called by TORCH_LIBRARY registration)
// ---------------------------------------------------------------------------
template <template <tmol::Device> class DispatchMethod>
std::vector<Tensor> genbonded_pose_scores_op(
    Tensor rot_coords,
    Tensor rot_coord_offset,
    Tensor pose_ind_for_atom,
    Tensor first_rot_for_block,
    Tensor first_rot_block_type,
    Tensor block_ind_for_rot,
    Tensor pose_ind_for_rot,
    Tensor block_type_ind_for_rot,
    Tensor n_rots_for_pose,
    Tensor rot_offset_for_pose,
    Tensor n_rots_for_block,
    Tensor rot_offset_for_block,
    int64_t max_n_rots_per_pose,

    Tensor pose_stack_inter_block_connections,
    Tensor atom_paths_from_conn,
    Tensor gen_intra_subgraphs,
    Tensor gen_intra_subgraph_offsets,
    Tensor gen_intra_params,
    Tensor gen_atom_type_hierarchy,
    Tensor gen_connection_bond_types,
    Tensor gen_inter_torsion_hash_keys,
    Tensor gen_inter_torsion_hash_values,

    bool output_block_pair_energies) {
  return GenBondedPoseScoreOp<DispatchMethod>::apply(
      rot_coords,
      rot_coord_offset,
      pose_ind_for_atom,
      first_rot_for_block,
      first_rot_block_type,
      block_ind_for_rot,
      pose_ind_for_rot,
      block_type_ind_for_rot,
      n_rots_for_pose,
      rot_offset_for_pose,
      n_rots_for_block,
      rot_offset_for_block,
      max_n_rots_per_pose,

      pose_stack_inter_block_connections,
      atom_paths_from_conn,
      gen_intra_subgraphs,
      gen_intra_subgraph_offsets,
      gen_intra_params,
      gen_atom_type_hierarchy,
      gen_connection_bond_types,
      gen_inter_torsion_hash_keys,
      gen_inter_torsion_hash_values,

      output_block_pair_energies);
}

template <template <tmol::Device> class DispatchMethod>
std::vector<Tensor> genbonded_rotamer_scores_op(
    Tensor rot_coords,
    Tensor rot_coord_offset,
    Tensor pose_ind_for_atom,
    Tensor first_rot_for_block,
    Tensor first_rot_block_type,
    Tensor block_ind_for_rot,
    Tensor pose_ind_for_rot,
    Tensor block_type_ind_for_rot,
    Tensor n_rots_for_pose,
    Tensor rot_offset_for_pose,
    Tensor n_rots_for_block,
    Tensor rot_offset_for_block,
    int64_t max_n_rots_per_pose,

    Tensor pose_stack_inter_block_connections,
    Tensor atom_paths_from_conn,
    Tensor gen_intra_subgraphs,
    Tensor gen_intra_subgraph_offsets,
    Tensor gen_intra_params,
    Tensor gen_atom_type_hierarchy,
    Tensor gen_connection_bond_types,
    Tensor gen_inter_torsion_hash_keys,
    Tensor gen_inter_torsion_hash_values,

    bool output_block_pair_energies) {
  return GenBondedRotamerScoreOp<DispatchMethod>::apply(
      rot_coords,
      rot_coord_offset,
      pose_ind_for_atom,
      first_rot_for_block,
      first_rot_block_type,
      block_ind_for_rot,
      pose_ind_for_rot,
      block_type_ind_for_rot,
      n_rots_for_pose,
      rot_offset_for_pose,
      n_rots_for_block,
      rot_offset_for_block,
      max_n_rots_per_pose,

      pose_stack_inter_block_connections,
      atom_paths_from_conn,
      gen_intra_subgraphs,
      gen_intra_subgraph_offsets,
      gen_intra_params,
      gen_atom_type_hierarchy,
      gen_connection_bond_types,
      gen_inter_torsion_hash_keys,
      gen_inter_torsion_hash_values,

      output_block_pair_energies);
}

TORCH_LIBRARY(tmol_genbonded, m) {
  m.def("genbonded_pose_scores", &genbonded_pose_scores_op<DeviceOperations>);
  m.def(
      "genbonded_rotamer_scores",
      &genbonded_rotamer_scores_op<DeviceOperations>);
}

}  // namespace potentials
}  // namespace genbonded
}  // namespace score
}  // namespace tmol
