#include <torch/torch.h>
#include <torch/script.h>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/simple_dispatch.hh>
#include <tmol/score/common/device_operations.hh>

#include "common.hh"
#include "common_dispatch.hh"
#include "params.hh"

namespace tmol {
namespace kinematics {

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::tensor_list;

class KinematicOp : public torch::autograd::Function<KinematicOp> {
 public:
  static Tensor forward(
      AutogradContext* ctx,
      Tensor dofs,
      Tensor nodes_f,
      Tensor scans_f,
      Tensor gens_f,
      Tensor nodes_b,
      Tensor scans_b,
      Tensor gens_b,
      Tensor kintree) {
    // printf("KinematicOp::forward\n");
    at::Tensor coords;
    at::Tensor HTs;

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(dofs.type(), "forward_kin_op", ([&] {
                                    using Real = scalar_t;
                                    constexpr tmol::Device Dev = device_t;

                                    auto result =
                                        ForwardKinDispatch<Dev, Real, Int>::f(
                                            TCAST(dofs),
                                            TCAST(nodes_f),
                                            TCAST(scans_f),
                                            TCAST(gens_f),
                                            TCAST(kintree));

                                    coords = std::get<0>(result).tensor;
                                    HTs = std::get<1>(result).tensor;
                                  }));

    ctx->save_for_backward({HTs, dofs, nodes_b, scans_b, gens_b, kintree});
    // printf("KinematicOp::forward -- end\n");

    return coords;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    // printf("KinematicOp::backward\n");
    auto saved = ctx->get_saved_variables();
    int i = 0;
    auto HTs = saved[i++];
    auto dofs = saved[i++];
    auto nodes_b = saved[i++];
    auto scans_b = saved[i++];
    auto gens_b = saved[i++];
    auto kintree = saved[i++];

    at::Tensor dV_ddof;
    using Int = int32_t;
    auto dVdx = grad_outputs[0];
    TMOL_DISPATCH_FLOATING_DEVICE(HTs.type(), "kin_deriv_op", ([&] {
                                    using Real = scalar_t;
                                    constexpr tmol::Device Dev = device_t;

                                    auto result =
                                        KinDerivDispatch<Dev, Real, Int>::f(
                                            TCAST(dVdx),
                                            TCAST(HTs),
                                            TCAST(dofs),
                                            TCAST(nodes_b),
                                            TCAST(scans_b),
                                            TCAST(gens_b),
                                            TCAST(kintree));

                                    dV_ddof = result.tensor;
                                  }));

    // printf("KinematicOp::backward -- end\n");
    return {
        dV_ddof,
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
    };
  }
};

Tensor kinematic_op(
    Tensor dofs,
    Tensor nodes_f,
    Tensor scans_f,
    Tensor gens_f,
    Tensor nodes_b,
    Tensor scans_b,
    Tensor gens_b,
    Tensor kintree) {
  // printf("kinematic_op\n");
  Tensor retval = KinematicOp::apply(
      dofs, nodes_f, scans_f, gens_f, nodes_b, scans_b, gens_b, kintree);
  // printf("kinematic_op -- end\n");
  return retval;
}

Tensor forward_only_op(
    Tensor dofs,
    Tensor nodes_f,
    Tensor scans_f,
    Tensor gens_f,
    Tensor kintree) {
  at::Tensor coords;

  using Int = int32_t;

  TMOL_DISPATCH_FLOATING_DEVICE(dofs.type(), "forward_kin_only_op", ([&] {
                                  using Real = scalar_t;
                                  constexpr tmol::Device Dev = device_t;

                                  auto result =
                                      ForwardKinDispatch<Dev, Real, Int>::f(
                                          TCAST(dofs),
                                          TCAST(nodes_f),
                                          TCAST(scans_f),
                                          TCAST(gens_f),
                                          TCAST(kintree));

                                  coords = std::get<0>(result).tensor;
                                }));

  return coords;
};

auto get_kfo_indices_for_atoms(
    Tensor pose_stack_block_coord_offset,
    Tensor pose_stack_block_type,
    Tensor block_type_n_atoms,
    Tensor block_type_atom_is_real) -> tensor_list {
  // printf("GET KFO INDICES FOR ATOMS\n");
  at::Tensor block_kfo_offset_tp;
  at::Tensor kfo_2_orig_mapping_tp;
  at::Tensor atom_kfo_index;
  TMOL_DISPATCH_INDEX_DEVICE(
      pose_stack_block_coord_offset.type(), "get_kfo_indices_for_atoms", ([&] {
        // using Int = index_t;
        using Int = int32_t;  // ONLY 32-bit integers supported! No atomicAdd
                              // for signed 64-bit integers in CUDA
        // using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        auto result =
            KinForestFromStencil<score::common::DeviceOperations, Dev, Int>::
                get_kfo_indices_for_atoms(
                    TCAST(pose_stack_block_coord_offset),
                    TCAST(pose_stack_block_type),
                    TCAST(block_type_n_atoms),
                    TCAST(block_type_atom_is_real));
        block_kfo_offset_tp = std::get<0>(result).tensor;
        kfo_2_orig_mapping_tp = std::get<1>(result).tensor;
        atom_kfo_index = std::get<2>(result).tensor;
      }));
  return {block_kfo_offset_tp, kfo_2_orig_mapping_tp, atom_kfo_index};
}

auto get_kfo_atom_parents(
    Tensor pose_stack_block_type,                 // P x L
    Tensor pose_stack_inter_residue_connections,  // P x L x C x 2
    Tensor pose_stack_ff_parent,                  // P x L
    // Tensor pose_stack_ff_conn_to_parent,          // P x L
    Tensor pose_stack_block_in_and_first_out,  // P x L x 2
    Tensor block_type_parents,                 // T x O x A
    Tensor kfo_2_orig_mapping,                 // K x 3
    Tensor atom_kfo_index,                     // P x L x A
    Tensor block_type_jump_atom,               // T
    Tensor block_type_n_conn,                  // T
    Tensor block_type_conn_atom) -> tensor_list {
  // printf("GET KFO ATOM PARENTS\n");
  at::Tensor kfo_parent_atoms;
  at::Tensor kfo_grandparent_atoms;
  TMOL_DISPATCH_INDEX_DEVICE(
      pose_stack_block_type.type(), "get_kfo_atom_parents", ([&] {
        using Int = int32_t;  // ONLY 32-bit integers supported! No atomicAdd
                              // for signed 64-bit integers in CUDA
        // using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        auto result =
            KinForestFromStencil<score::common::DeviceOperations, Dev, Int>::
                get_kfo_atom_parents(
                    TCAST(pose_stack_block_type),
                    TCAST(pose_stack_inter_residue_connections),
                    TCAST(pose_stack_ff_parent),
                    // TCAST(pose_stack_ff_conn_to_parent),
                    TCAST(pose_stack_block_in_and_first_out),
                    TCAST(block_type_parents),
                    TCAST(kfo_2_orig_mapping),
                    TCAST(atom_kfo_index),
                    TCAST(block_type_jump_atom),
                    TCAST(block_type_n_conn),
                    TCAST(block_type_conn_atom));

        kfo_parent_atoms = std::get<0>(result).tensor;
        kfo_grandparent_atoms = std::get<1>(result).tensor;
      }));
  return {kfo_parent_atoms, kfo_grandparent_atoms};
}

auto get_children(
    Tensor pose_stack_block_type,              // P x L
    Tensor pose_stack_block_in_and_first_out,  // P x L
    Tensor kfo_2_orig_mapping,                 // K x 3
    Tensor kfo_parent_atoms,                   // K
    Tensor block_type_n_conn                   // T
    ) -> tensor_list {
  // printf("GET CHILDREN\n");
  at::Tensor n_children;
  at::Tensor child_list_span;
  at::Tensor child_list;
  at::Tensor is_atom_jump;

  TMOL_DISPATCH_INDEX_DEVICE(
      pose_stack_block_type.type(), "get_children", ([&] {
        // using Int = index_t;
        using Int = int32_t;  // ONLY 32-bit integers supported! No atomicAdd
                              // for signed 64-bit integers in CUDA
        // using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        auto result =
            KinForestFromStencil<score::common::DeviceOperations, Dev, Int>::
                get_children(
                    TCAST(pose_stack_block_type),
                    TCAST(pose_stack_block_in_and_first_out),
                    TCAST(kfo_2_orig_mapping),
                    TCAST(kfo_parent_atoms),
                    TCAST(block_type_n_conn));

        n_children = std::get<0>(result).tensor;
        child_list_span = std::get<1>(result).tensor;
        child_list = std::get<2>(result).tensor;
        is_atom_jump = std::get<3>(result).tensor;
      }));
  return {n_children, child_list_span, child_list, is_atom_jump};
}

auto get_id_and_frame_xyz(
    int64_t max_n_pose_atoms,
    Tensor pose_stack_block_coord_offset,
    Tensor kfo_2_orig_mapping,  // K x 3
    Tensor parents,             // P x L
    Tensor child_list_span,     // P x L
    Tensor child_list,          // K x 3
    Tensor is_atom_jump         // K
    ) -> tensor_list {
  // printf("GET FRAME X Y Z\n");
  at::Tensor id;
  at::Tensor frame_x;
  at::Tensor frame_y;
  at::Tensor frame_z;
  at::Tensor keep_dof_fixed;

  TMOL_DISPATCH_INDEX_DEVICE(
      parents.type(), "get_id_and_frame_xyz", ([&] {
        // using Int = index_t;
        using Int = int32_t;  // ONLY 32-bit integers supported! No atomicAdd
                              // for signed 64-bit integers in CUDA
        // using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        auto result =
            KinForestFromStencil<score::common::DeviceOperations, Dev, Int>::
                get_id_and_frame_xyz(
                    max_n_pose_atoms,
                    TCAST(pose_stack_block_coord_offset),
                    TCAST(kfo_2_orig_mapping),
                    TCAST(parents),
                    TCAST(child_list_span),
                    TCAST(child_list),
                    TCAST(is_atom_jump));

        id = std::get<0>(result).tensor;
        frame_x = std::get<1>(result).tensor;
        frame_y = std::get<2>(result).tensor;
        frame_z = std::get<3>(result).tensor;
        keep_dof_fixed = std::get<4>(result).tensor;
      }));
  return {id, frame_x, frame_y, frame_z, keep_dof_fixed};
}

auto calculate_ff_edge_delays(
    Tensor pose_stack_block_coord_offset,  // P x L
    Tensor pose_stack_block_type,          // x - P x L
    Tensor ff_edges_cpu,  // y - P x E x 4 -- 0: type, 1: start, 2: stop, 3:
                          // jump ind
    Tensor block_type_kts_conn_info,    // y - T x I x O x C x 2 -- 2 is for gen
                                        // (0) and scan (1)
    Tensor block_type_nodes_for_gens,   // y - T x I x O x G x N
    Tensor block_type_scan_path_starts  // y - T x I x O x G x S
    ) -> tensor_list {
  // printf("CALCULATE FF EDGE DELAYS\n");
  Tensor dfs_order_of_ff_edges;
  Tensor n_ff_edges;
  Tensor ff_edge_parent;
  Tensor first_ff_edge_for_block_cpu;
  Tensor pose_stack_ff_parent;
  Tensor max_gen_depth_of_ff_edge;
  Tensor first_child_of_ff_edge;
  Tensor delay_for_edge;
  Tensor toposort_index_for_edge;
  TMOL_DISPATCH_INDEX_DEVICE(
      pose_stack_block_type.type(), "calculate_ff_edge_delays", ([&] {
        // using Int = index_t;
        using Int = int32_t;  // ONLY 32-bit integers supported! No atomicAdd
                              // for signed 64-bit integers in CUDA
        // using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        auto result =
            KinForestFromStencil<score::common::DeviceOperations, Dev, Int>::
                calculate_ff_edge_delays(
                    TCAST(pose_stack_block_coord_offset),
                    TCAST(pose_stack_block_type),
                    TCAST(ff_edges_cpu),
                    TCAST(block_type_kts_conn_info),
                    TCAST(block_type_nodes_for_gens),
                    TCAST(block_type_scan_path_starts));
        dfs_order_of_ff_edges = std::get<0>(result).tensor;
        n_ff_edges = std::get<1>(result).tensor;
        ff_edge_parent = std::get<2>(result).tensor;
        first_ff_edge_for_block_cpu = std::get<3>(result).tensor;
        pose_stack_ff_parent = std::get<4>(result).tensor;
        max_gen_depth_of_ff_edge = std::get<5>(result).tensor;
        first_child_of_ff_edge = std::get<6>(result).tensor;
        delay_for_edge = std::get<7>(result).tensor;
        toposort_index_for_edge = std::get<8>(result).tensor;
      }));
  return {
      dfs_order_of_ff_edges,
      n_ff_edges,
      ff_edge_parent,
      first_ff_edge_for_block_cpu,
      pose_stack_ff_parent,
      max_gen_depth_of_ff_edge,
      first_child_of_ff_edge,
      delay_for_edge,
      toposort_index_for_edge};
}

auto get_jump_atom_indices(
    Tensor ff_edges,  // P x E x 4 -- 0: type, 1: start, 2: stop, 3: jump ind
    Tensor pose_stack_block_type,  // P x L
    Tensor block_type_jump_atom    // T
    ) -> Tensor {
  Tensor pose_stack_atom_for_jump;
  TMOL_DISPATCH_INDEX_DEVICE(
      pose_stack_block_type.type(), "calculate_ff_edge_delays", ([&] {
        // using Int = index_t;
        using Int = int32_t;  // ONLY 32-bit integers supported! No atomicAdd
                              // for signed 64-bit integers in CUDA
        // using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        auto result =
            KinForestFromStencil<score::common::DeviceOperations, Dev, Int>::
                get_jump_atom_indices(
                    TCAST(ff_edges),
                    TCAST(pose_stack_block_type),
                    TCAST(block_type_jump_atom));
        pose_stack_atom_for_jump = result.tensor;
      }));
  return pose_stack_atom_for_jump;
}

auto get_block_parent_connectivity_from_toposort(
    Tensor pose_stack_block_type,                 // P x L
    Tensor pose_stack_inter_residue_connections,  // P x L x C x 2
    Tensor pose_stack_ff_parent,
    Tensor dfs_order_of_ff_edges,
    Tensor n_ff_edges,               // P
    Tensor ff_edges,                 // P x E x 4
    Tensor first_ff_edge_for_block,  // P x L
    // Tensor max_n_gens_for_ff_edge, // P x E
    Tensor first_child_of_ff_edge,    // P x E
    Tensor delay_for_edge,            // P x E
    Tensor topo_sort_index_for_edge,  // (P*E)
    Tensor block_type_n_conn,         // T
    Tensor block_type_polymeric_conn_index) -> Tensor {
  // printf("GET BLOCK PARENT CONNECTIVITY FROM TOPOSORT\n");

  Tensor pose_stack_block_in_and_first_out;
  TMOL_DISPATCH_INDEX_DEVICE(
      pose_stack_block_type.type(), "calculate_ff_edge_delays", ([&] {
        // using Int = index_t;
        using Int = int32_t;  // ONLY 32-bit integers supported! No atomicAdd
                              // for signed 64-bit integers in CUDA
        // using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        auto result =
            KinForestFromStencil<score::common::DeviceOperations, Dev, Int>::
                get_block_parent_connectivity_from_toposort(
                    TCAST(pose_stack_block_type),  // P x L
                    TCAST(
                        pose_stack_inter_residue_connections),  // P x L x C x 2
                    TCAST(pose_stack_ff_parent),
                    TCAST(dfs_order_of_ff_edges),
                    TCAST(n_ff_edges),               // P
                    TCAST(ff_edges),                 // P x E x 4
                    TCAST(first_ff_edge_for_block),  // P x L
                    // TCAST(max_n_gens_for_ff_edge), // P x E
                    TCAST(first_child_of_ff_edge),    // P x E
                    TCAST(delay_for_edge),            // P x E
                    TCAST(topo_sort_index_for_edge),  // (P*E)
                    TCAST(block_type_n_conn),         // T
                    TCAST(block_type_polymeric_conn_index));
        pose_stack_block_in_and_first_out = result.tensor;
      }));
  return pose_stack_block_in_and_first_out;
}

auto get_scans2(
    int64_t const max_n_atoms_per_pose,
    Tensor pose_stack_block_coord_offset,         // P x L
    Tensor pose_stack_block_type,                 // P x L
    Tensor pose_stack_inter_residue_connections,  // P x L x C x 2
    Tensor ff_edges,  // P x E x 4 -- 0: type, 1: start, 2: stop, 3: jump ind
    int64_t const max_delay,
    Tensor delay_for_edge,                     // P x E
    Tensor topo_sort_index_for_edge,           // (P*E)
    Tensor first_ff_edge_for_block,            // P x L
    Tensor pose_stack_ff_parent,               // P x L
    Tensor pose_stack_block_in_and_first_out,  // P x L x 2
    Tensor block_type_parents,                 // T x O x A
    Tensor kfo_2_orig_mapping,                 // K x 3
    Tensor atom_kfo_index,                     // P x L x A
    Tensor block_type_jump_atom,               // T
    Tensor block_type_n_conn,                  // T
    Tensor block_type_polymeric_conn_index,  // T x 2 - 2 is for "down" and "up"
                                             // connections.
    Tensor block_type_n_gens,                // T x I x O
    Tensor block_type_kts_conn_info,         // T x I x O x C x 2 - 2 is for
                                             // gen (0) and scan (1)
    Tensor block_type_nodes_for_gens,        // T x I x O x G x N
    Tensor block_type_n_scan_paths,          // T x I x O x G
    Tensor block_type_scan_path_starts,      // T x I x O x G x S
    Tensor block_type_scan_path_is_real,     // T x I x O x G x S
    Tensor block_type_scan_path_is_inter_block,  // T x I x O x G x S
    Tensor block_type_scan_path_length           // T x I x O x G x S
    ) -> tensor_list {
  // printf("GET SCANS2\n");
  Tensor nodes_fw;
  Tensor scans_fw;
  Tensor gens_fw;
  Tensor nodes_bw;
  Tensor scans_bw;
  Tensor gens_bw;
  TMOL_DISPATCH_INDEX_DEVICE(
      pose_stack_block_type.type(), "calculate_ff_edge_delays", ([&] {
        // using Int = index_t;
        using Int = int32_t;  // ONLY 32-bit integers supported! No atomicAdd
                              // for signed 64-bit integers in CUDA
        // using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        auto result =
            KinForestFromStencil<score::common::DeviceOperations, Dev, Int>::
                get_scans2(
                    max_n_atoms_per_pose,
                    TCAST(pose_stack_block_coord_offset),
                    TCAST(pose_stack_block_type),
                    TCAST(pose_stack_inter_residue_connections),
                    TCAST(ff_edges),
                    max_delay,
                    TCAST(delay_for_edge),
                    TCAST(topo_sort_index_for_edge),
                    TCAST(first_ff_edge_for_block),
                    TCAST(pose_stack_ff_parent),
                    TCAST(pose_stack_block_in_and_first_out),
                    TCAST(block_type_parents),
                    TCAST(kfo_2_orig_mapping),
                    TCAST(atom_kfo_index),
                    TCAST(block_type_jump_atom),
                    TCAST(block_type_n_conn),
                    TCAST(block_type_polymeric_conn_index),
                    TCAST(block_type_n_gens),
                    TCAST(block_type_kts_conn_info),
                    TCAST(block_type_nodes_for_gens),
                    TCAST(block_type_n_scan_paths),
                    TCAST(block_type_scan_path_starts),
                    TCAST(block_type_scan_path_is_real),
                    TCAST(block_type_scan_path_is_inter_block),
                    TCAST(block_type_scan_path_length));
        nodes_fw = std::get<0>(result).tensor;
        scans_fw = std::get<1>(result).tensor;
        gens_fw = std::get<2>(result).tensor;
        nodes_bw = std::get<3>(result).tensor;
        scans_bw = std::get<4>(result).tensor;
        gens_bw = std::get<5>(result).tensor;
      }));
  return {nodes_fw, scans_fw, gens_fw, nodes_bw, scans_bw, gens_bw};
}

auto minimizer_map_from_movemap(
    Tensor kinforest_id,
    int64_t max_n_atoms_per_pose,
    Tensor pose_stack_block_coord_offset,
    Tensor pose_stack_block_type,  // P x L
    Tensor pose_stack_inter_block_connections,
    Tensor pose_stack_block_in_and_first_out,  // P x L x 2
    Tensor pose_stack_atom_for_jump,
    Tensor keep_dof_fixed,
    Tensor bt_n_named_torsions,
    Tensor bt_uaid_for_torsion,
    Tensor bt_torsion_direction,
    Tensor bt_named_torsion_is_bb,
    Tensor bt_which_mcsc_torsion_for_named_torsion,
    Tensor bt_atom_downstream_of_conn,
    bool move_all_jumps,
    bool move_all_mc,
    bool move_all_sc,
    bool move_all_named_torsions,
    Tensor move_jumps,
    Tensor move_jumps_mask,
    Tensor move_mcs,
    Tensor move_mcs_mask,
    Tensor move_scs,
    Tensor move_scs_mask,
    Tensor move_named_torsions,
    Tensor move_named_torsions_mask,
    Tensor move_jump_dof,
    Tensor move_jump_dof_mask,
    Tensor move_mc_dof,
    Tensor move_mc_dof_mask,
    Tensor move_sc_dof,
    Tensor move_sc_dof_mask,
    Tensor move_named_torsion_dof,
    Tensor move_named_torsion_dof_mask,
    Tensor move_atom_dof,
    Tensor move_atom_dof_mask) -> Tensor {
  // Minimizer map: a boolean vector of the DOFs that are free
  Tensor minimizer_map;  // maybe more??
  TMOL_DISPATCH_INDEX_DEVICE(
      pose_stack_block_type.type(), "minimizer_map_from_movemap", ([&] {
        using Int = int32_t;
        constexpr tmol::Device Dev = device_t;

        auto result =
            KinForestFromStencil<score::common::DeviceOperations, Dev, Int>::
                create_minimizer_map(
                    TCAST(kinforest_id),
                    max_n_atoms_per_pose,
                    TCAST(pose_stack_block_coord_offset),
                    TCAST(pose_stack_block_type),
                    TCAST(pose_stack_inter_block_connections),
                    TCAST(pose_stack_block_in_and_first_out),
                    TCAST(pose_stack_atom_for_jump),
                    TCAST(keep_dof_fixed),
                    TCAST(bt_n_named_torsions),
                    TCAST(bt_uaid_for_torsion),
                    TCAST(bt_torsion_direction),
                    TCAST(bt_named_torsion_is_bb),
                    TCAST(bt_which_mcsc_torsion_for_named_torsion),
                    TCAST(bt_atom_downstream_of_conn),
                    move_all_jumps,
                    move_all_mc,
                    move_all_sc,
                    move_all_named_torsions,
                    TCAST(move_jumps),
                    TCAST(move_jumps_mask),
                    TCAST(move_mcs),
                    TCAST(move_mcs_mask),
                    TCAST(move_scs),
                    TCAST(move_scs_mask),
                    TCAST(move_named_torsions),
                    TCAST(move_named_torsions_mask),
                    TCAST(move_jump_dof),
                    TCAST(move_jump_dof_mask),
                    TCAST(move_mc_dof),
                    TCAST(move_mc_dof_mask),
                    TCAST(move_sc_dof),
                    TCAST(move_sc_dof_mask),
                    TCAST(move_named_torsion_dof),
                    TCAST(move_named_torsion_dof_mask),
                    TCAST(move_atom_dof),
                    TCAST(move_atom_dof_mask));
        // minimizer_map = std::get<0>(result).tensor;
        minimizer_map = result.tensor;
      }));
  return minimizer_map;
}

// Macro indirection to force TORCH_EXTENSION_NAME macro expansion
// See https://stackoverflow.com/a/3221914
#define TORCH_LIBRARY_(ns, m) TORCH_LIBRARY(ns, m)

TORCH_LIBRARY_(TORCH_EXTENSION_NAME, m) {
  m.def("forward_kin_op", &kinematic_op);
  m.def("forward_only_op", &forward_only_op);
  // m.def("fix_jump_nodes_op", &fix_jump_nodes_op);
  m.def("get_kfo_indices_for_atoms", &get_kfo_indices_for_atoms);
  m.def("get_kfo_atom_parents", &get_kfo_atom_parents);
  m.def("get_children", &get_children);
  m.def("get_id_and_frame_xyz", &get_id_and_frame_xyz);
  m.def("calculate_ff_edge_delays", &calculate_ff_edge_delays);
  m.def("get_jump_atom_indices", &get_jump_atom_indices);
  m.def(
      "get_block_parent_connectivity_from_toposort",
      &get_block_parent_connectivity_from_toposort);
  m.def("get_kinforest_scans_from_stencils", &get_scans2);
  m.def("get_kinforest_scans_from_stencils2", &get_scans2);
  m.def("minimizer_map_from_movemap", &minimizer_map_from_movemap);
}

}  // namespace kinematics
}  // namespace tmol
