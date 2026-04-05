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

// MPS round-trip helper: TPack allocates on CPU for MPS inputs; move back.
static inline at::Tensor mps_to_dev(at::Tensor t, c10::Device dev) {
  return dev.is_mps() ? t.to(dev) : t;
}

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
    at::Tensor coords;
    at::Tensor HTs;

    c10::Device orig_device = dofs.device();
    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(dofs.options(), "forward_kin_op", ([&] {
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

    coords = mps_to_dev(coords, orig_device);
    HTs = mps_to_dev(HTs, orig_device);
    ctx->save_for_backward({HTs, dofs, nodes_b, scans_b, gens_b, kintree});

    return coords;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    int i = 0;
    auto HTs = saved[i++];
    auto dofs = saved[i++];
    auto nodes_b = saved[i++];
    auto scans_b = saved[i++];
    auto gens_b = saved[i++];
    auto kintree = saved[i++];

    at::Tensor dV_ddof;
    c10::Device orig_device = dofs.device();
    using Int = int32_t;
    auto dVdx = grad_outputs[0];
    TMOL_DISPATCH_FLOATING_DEVICE(HTs.options(), "kin_deriv_op", ([&] {
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

    dV_ddof = mps_to_dev(dV_ddof, orig_device);

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
  Tensor retval = KinematicOp::apply(
      dofs, nodes_f, scans_f, gens_f, nodes_b, scans_b, gens_b, kintree);
  return retval;
}

Tensor forward_only_op(
    Tensor dofs,
    Tensor nodes_f,
    Tensor scans_f,
    Tensor gens_f,
    Tensor kintree) {
  at::Tensor coords;

  c10::Device orig_device = dofs.device();
  using Int = int32_t;

  TMOL_DISPATCH_FLOATING_DEVICE(dofs.options(), "forward_kin_only_op", ([&] {
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

  return mps_to_dev(coords, orig_device);
};

auto get_kfo_indices_for_atoms(
    Tensor pose_stack_block_coord_offset,
    Tensor pose_stack_block_type,
    Tensor block_type_n_atoms,
    Tensor block_type_atom_is_real) -> tensor_list {
  at::Tensor block_kfo_offset_tp;
  at::Tensor kfo_2_orig_mapping_tp;
  at::Tensor atom_kfo_index;
  c10::Device dev = pose_stack_block_coord_offset.device();
  TMOL_DISPATCH_INDEX_DEVICE(
      pose_stack_block_coord_offset.options(),
      "get_kfo_indices_for_atoms",
      ([&] {
        using Int = int32_t;
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
  return {
      mps_to_dev(block_kfo_offset_tp, dev),
      mps_to_dev(kfo_2_orig_mapping_tp, dev),
      mps_to_dev(atom_kfo_index, dev)};
}

auto get_kfo_atom_parents(
    Tensor pose_stack_block_type,                 // P x L
    Tensor pose_stack_inter_residue_connections,  // P x L x C x 2
    Tensor pose_stack_ff_parent,                  // P x L
    Tensor pose_stack_block_in_and_first_out,     // P x L x 2
    Tensor block_type_parents,                    // T x O x A
    Tensor kfo_2_orig_mapping,                    // K x 3
    Tensor atom_kfo_index,                        // P x L x A
    Tensor block_type_jump_atom,                  // T
    Tensor block_type_n_conn,                     // T
    Tensor block_type_conn_atom) -> tensor_list {
  at::Tensor kfo_parent_atoms;
  at::Tensor kfo_grandparent_atoms;
  c10::Device dev = pose_stack_block_type.device();
  TMOL_DISPATCH_INDEX_DEVICE(
      pose_stack_block_type.options(), "get_kfo_atom_parents", ([&] {
        using Int = int32_t;
        constexpr tmol::Device Dev = device_t;

        auto result =
            KinForestFromStencil<score::common::DeviceOperations, Dev, Int>::
                get_kfo_atom_parents(
                    TCAST(pose_stack_block_type),
                    TCAST(pose_stack_inter_residue_connections),
                    TCAST(pose_stack_ff_parent),
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
  return {mps_to_dev(kfo_parent_atoms, dev), mps_to_dev(kfo_grandparent_atoms, dev)};
}

auto get_children(
    Tensor pose_stack_block_type,              // P x L
    Tensor pose_stack_block_in_and_first_out,  // P x L
    Tensor kfo_2_orig_mapping,                 // K x 3
    Tensor kfo_parent_atoms,                   // K
    Tensor block_type_n_conn                   // T
    ) -> tensor_list {
  at::Tensor n_children;
  at::Tensor child_list_span;
  at::Tensor child_list;
  at::Tensor is_atom_jump;

  c10::Device dev = pose_stack_block_type.device();
  TMOL_DISPATCH_INDEX_DEVICE(
      pose_stack_block_type.options(), "get_children", ([&] {
        using Int = int32_t;
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
  return {
      mps_to_dev(n_children, dev),
      mps_to_dev(child_list_span, dev),
      mps_to_dev(child_list, dev),
      mps_to_dev(is_atom_jump, dev)};
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
  at::Tensor id;
  at::Tensor frame_x;
  at::Tensor frame_y;
  at::Tensor frame_z;
  at::Tensor keep_dof_fixed;

  c10::Device dev = parents.device();
  TMOL_DISPATCH_INDEX_DEVICE(
      parents.options(), "get_id_and_frame_xyz", ([&] {
        using Int = int32_t;
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
  return {
      mps_to_dev(id, dev),
      mps_to_dev(frame_x, dev),
      mps_to_dev(frame_y, dev),
      mps_to_dev(frame_z, dev),
      mps_to_dev(keep_dof_fixed, dev)};
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
  Tensor dfs_order_of_ff_edges;
  Tensor n_ff_edges;
  Tensor ff_edge_parent;
  Tensor first_ff_edge_for_block_cpu;
  Tensor pose_stack_ff_parent;
  Tensor max_gen_depth_of_ff_edge;
  Tensor first_child_of_ff_edge;
  Tensor delay_for_edge;
  Tensor toposort_index_for_edge;
  c10::Device dev = pose_stack_block_type.device();
  TMOL_DISPATCH_INDEX_DEVICE(
      pose_stack_block_type.options(), "calculate_ff_edge_delays", ([&] {
        using Int = int32_t;
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
      mps_to_dev(dfs_order_of_ff_edges, dev),
      mps_to_dev(n_ff_edges, dev),
      mps_to_dev(ff_edge_parent, dev),
      mps_to_dev(first_ff_edge_for_block_cpu, dev),
      mps_to_dev(pose_stack_ff_parent, dev),
      mps_to_dev(max_gen_depth_of_ff_edge, dev),
      mps_to_dev(first_child_of_ff_edge, dev),
      mps_to_dev(delay_for_edge, dev),
      mps_to_dev(toposort_index_for_edge, dev)};
}

auto get_jump_atom_indices(
    Tensor ff_edges,  // P x E x 4 -- 0: type, 1: start, 2: stop, 3: jump ind
    Tensor pose_stack_block_type,  // P x L
    Tensor block_type_jump_atom    // T
    ) -> tensor_list {
  Tensor pose_stack_atom_for_jump;
  Tensor pose_stack_atom_for_root_jump;
  c10::Device dev = pose_stack_block_type.device();
  TMOL_DISPATCH_INDEX_DEVICE(
      pose_stack_block_type.options(), "calculate_ff_edge_delays", ([&] {
        using Int = int32_t;
        constexpr tmol::Device Dev = device_t;

        auto result =
            KinForestFromStencil<score::common::DeviceOperations, Dev, Int>::
                get_jump_atom_indices(
                    TCAST(ff_edges),
                    TCAST(pose_stack_block_type),
                    TCAST(block_type_jump_atom));
        pose_stack_atom_for_jump = std::get<0>(result).tensor;
        pose_stack_atom_for_root_jump = std::get<1>(result).tensor;
      }));
  return {
      mps_to_dev(pose_stack_atom_for_jump, dev),
      mps_to_dev(pose_stack_atom_for_root_jump, dev)};
}

auto get_block_parent_connectivity_from_toposort(
    Tensor pose_stack_block_type,                 // P x L
    Tensor pose_stack_inter_residue_connections,  // P x L x C x 2
    Tensor pose_stack_ff_parent,
    Tensor dfs_order_of_ff_edges,
    Tensor n_ff_edges,                // P
    Tensor ff_edges,                  // P x E x 4
    Tensor first_ff_edge_for_block,   // P x L
    Tensor first_child_of_ff_edge,    // P x E
    Tensor delay_for_edge,            // P x E
    Tensor topo_sort_index_for_edge,  // (P*E)
    Tensor block_type_n_conn,         // T
    Tensor block_type_polymeric_conn_index) -> Tensor {
  Tensor pose_stack_block_in_and_first_out;
  c10::Device dev = pose_stack_block_type.device();
  TMOL_DISPATCH_INDEX_DEVICE(
      pose_stack_block_type.options(), "calculate_ff_edge_delays", ([&] {
        using Int = int32_t;
        constexpr tmol::Device Dev = device_t;

        auto result =
            KinForestFromStencil<score::common::DeviceOperations, Dev, Int>::
                get_block_parent_connectivity_from_toposort(
                    TCAST(pose_stack_block_type),
                    TCAST(pose_stack_inter_residue_connections),
                    TCAST(pose_stack_ff_parent),
                    TCAST(dfs_order_of_ff_edges),
                    TCAST(n_ff_edges),
                    TCAST(ff_edges),
                    TCAST(first_ff_edge_for_block),
                    TCAST(first_child_of_ff_edge),
                    TCAST(delay_for_edge),
                    TCAST(topo_sort_index_for_edge),
                    TCAST(block_type_n_conn),
                    TCAST(block_type_polymeric_conn_index));
        pose_stack_block_in_and_first_out = result.tensor;
      }));
  return mps_to_dev(pose_stack_block_in_and_first_out, dev);
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
  Tensor nodes_fw;
  Tensor scans_fw;
  Tensor gens_fw;
  Tensor nodes_bw;
  Tensor scans_bw;
  Tensor gens_bw;
  c10::Device dev = pose_stack_block_type.device();
  TMOL_DISPATCH_INDEX_DEVICE(
      pose_stack_block_type.options(), "calculate_ff_edge_delays", ([&] {
        using Int = int32_t;
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
  return {
      mps_to_dev(nodes_fw, dev),
      mps_to_dev(scans_fw, dev),
      mps_to_dev(gens_fw, dev),
      mps_to_dev(nodes_bw, dev),
      mps_to_dev(scans_bw, dev),
      mps_to_dev(gens_bw, dev)};
}

auto minimizer_map_from_movemap(
    Tensor kinforest_id,
    int64_t max_n_atoms_per_pose,
    Tensor pose_stack_block_coord_offset,
    Tensor pose_stack_block_type,  // P x L
    Tensor pose_stack_inter_block_connections,
    Tensor pose_stack_block_in_and_first_out,  // P x L x 2
    Tensor pose_stack_atom_for_jump,           // P x E x 2
    Tensor pose_stack_atom_for_root_jump,      // P x L
    Tensor keep_dof_fixed,
    Tensor bt_n_named_torsions,
    Tensor bt_uaid_for_torsion,
    Tensor bt_named_torsion_is_mc,
    Tensor bt_which_mcsc_torsion_for_named_torsion,
    Tensor bt_atom_downstream_of_conn,
    bool move_all_jumps,
    bool move_all_root_jumps,
    bool move_all_mc,
    bool move_all_sc,
    bool move_all_named_torsions,
    bool non_ideal,
    Tensor move_jumps,
    Tensor move_jumps_mask,
    Tensor move_root_jumps,
    Tensor move_root_jumps_mask,
    Tensor move_mcs,
    Tensor move_mcs_mask,
    Tensor move_scs,
    Tensor move_scs_mask,
    Tensor move_named_torsions,
    Tensor move_named_torsions_mask,
    Tensor move_jump_dof,
    Tensor move_jump_dof_mask,
    Tensor move_root_jump_dof,
    Tensor move_root_jump_dof_mask,
    Tensor move_mc,
    Tensor move_mc_mask,
    Tensor move_sc,
    Tensor move_sc_mask,
    Tensor move_named_torsion,
    Tensor move_named_torsion_mask,
    Tensor move_atom_dof,
    Tensor move_atom_dof_mask) -> Tensor {
  // Minimizer map: a boolean vector of the DOFs that are free
  Tensor minimizer_map;
  c10::Device dev = pose_stack_block_type.device();
  TMOL_DISPATCH_INDEX_DEVICE(
      pose_stack_block_type.options(), "minimizer_map_from_movemap", ([&] {
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
                    TCAST(pose_stack_atom_for_root_jump),
                    TCAST(keep_dof_fixed),
                    TCAST(bt_n_named_torsions),
                    TCAST(bt_uaid_for_torsion),
                    TCAST(bt_named_torsion_is_mc),
                    TCAST(bt_which_mcsc_torsion_for_named_torsion),
                    TCAST(bt_atom_downstream_of_conn),
                    move_all_jumps,
                    move_all_root_jumps,
                    move_all_mc,
                    move_all_sc,
                    move_all_named_torsions,
                    non_ideal,
                    TCAST(move_jumps),
                    TCAST(move_jumps_mask),
                    TCAST(move_root_jumps),
                    TCAST(move_root_jumps_mask),
                    TCAST(move_mcs),
                    TCAST(move_mcs_mask),
                    TCAST(move_scs),
                    TCAST(move_scs_mask),
                    TCAST(move_named_torsions),
                    TCAST(move_named_torsions_mask),
                    TCAST(move_jump_dof),
                    TCAST(move_jump_dof_mask),
                    TCAST(move_root_jump_dof),
                    TCAST(move_root_jump_dof_mask),
                    TCAST(move_mc),
                    TCAST(move_mc_mask),
                    TCAST(move_sc),
                    TCAST(move_sc_mask),
                    TCAST(move_named_torsion),
                    TCAST(move_named_torsion_mask),
                    TCAST(move_atom_dof),
                    TCAST(move_atom_dof_mask));
        minimizer_map = result.tensor;
      }));
  return mps_to_dev(minimizer_map, dev);
}

// See https://stackoverflow.com/a/3221914

TORCH_LIBRARY(tmol_kin, m) {
  m.def("forward_kin_op", &kinematic_op);
  m.def("forward_only_op", &forward_only_op);
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
