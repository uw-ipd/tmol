#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/count_pair.hh>
#include <tmol/score/common/data_loading.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/uaid_util.hh>
#include <tmol/score/common/warp_segreduce.hh>
#include <tmol/score/common/warp_stride_reduce.hh>

#include <tmol/score/disulfide/potentials/disulfide_pose_score.hh>

// Operator definitions; safe for CPU comM_PIlation
#include <moderngpu/operators.hxx>

#include <chrono>
#include <cmath>

#include "params.hh"
#include "potentials.hh"

namespace tmol {
namespace score {
namespace disulfide {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

// The disulfide potential is intended to score cyteine-cysteine chemical
// bonds and the parameters are taken from distributions of bond geometries
// in the PDB. However, the logic is general enough that it can support
// non-cysteine residues that form S-S chemical bonds, and even so general
// that it can support cases where a residue forms more than one disulfide
// bond to its neighbors. Have we ever seen such a beast? No. Should we spend
// a few extra minutes to make sure that we can handle it properly _when_
// (not "if"; it's never "if") that crazy beast shows up.
//
// Just like the other potentials, this function will return a list of
// values that it has computed along with a list of which rotamer pairs
// those values correspond to. The trick is to keep track of which
// connection pair is being evaluated in a second tensor, thus allowing
// us to score the beast's double disulfide bond to another beast
// (that is, two disulfide bonds between residues i and j) without
// double counting.
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto DisulfidePoseScoreDispatch<DeviceDispatch, D, Real, Int>::forward(
    // common params
    TView<Vec<Real, 3>, 1, D> rot_coords,
    TView<Int, 1, D> rot_coord_offset,
    TView<Int, 1, D> pose_ind_for_atom,
    TView<Int, 2, D> first_rot_for_block,
    TView<Int, 2, D> first_rot_block_type,
    TView<Int, 1, D> block_ind_for_rot,
    TView<Int, 1, D> pose_ind_for_rot,
    TView<Int, 1, D> block_type_ind_for_rot,
    TView<Int, 1, D> n_rots_for_pose,
    TView<Int, 1, D> rot_offset_for_pose,
    TView<Int, 2, D> n_rots_for_block,
    TView<Int, 2, D> rot_offset_for_block,
    Int max_n_rots_per_pose,

    TView<Int, 2, D> pose_stack_block_type,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<bool, 2, D> disulfide_conns,
    TView<Int, 3, D> block_type_atom_downstream_of_conn,

    TView<DisulfideGlobalParams<Real>, 1, D> global_params,
    bool output_block_pair_energies,
    bool compute_derivs

    ) -> std::tuple<TPack<Real, 4, D>, TPack<Vec<Real, 3>, 2, D>> {
  int const n_atoms = rot_coords.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);
  int const max_n_conns = disulfide_conns.size(1);

  assert(first_rot_block_type.size(0) == n_poses);
  assert(first_rot_block_type.size(1) == max_n_blocks);

  assert(block_ind_for_rot.size(0) == n_rots);
  assert(pose_ind_for_rot.size(0) == n_rots);
  assert(block_type_ind_for_rot.size(0) == n_rots);

  assert(n_rots_for_pose.size(0) == n_poses);
  assert(rot_offset_for_pose.size(0) == n_poses);

  assert(n_rots_for_block.size(0) == n_poses);
  assert(n_rots_for_block.size(1) == max_n_blocks);

  assert(rot_offset_for_block.size(0) == n_poses);
  assert(rot_offset_for_block.size(1) == max_n_blocks);

  assert(pose_stack_inter_block_connections.size(0) == n_poses);
  assert(pose_stack_inter_block_connections.size(1) == max_n_blocks);
  assert(pose_stack_inter_block_connections.size(2) == max_n_conns);

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;

  int n_V;
  if (output_block_pair_energies) {
    n_V = max_n_blocks;
  } else {
    n_V = 1;
  }
  TPack<Real, 4, D> V_t = TPack<Real, 4, D>::zeros({1, n_poses, n_V, n_V});
  auto dV_dx_t = TPack<Vec<Real, 3>, 2, D>::zeros({1, n_atoms});

  auto V = V_t.view;
  auto dV_dx = dV_dx_t.view;

  auto eval_energies = ([=] TMOL_DEVICE_FUNC(int ind) {
    int const pose_ind = ind / max_n_blocks;
    int const block_ind1 = ind % max_n_blocks;

    int const rot_ind1 = first_rot_for_block[pose_ind][block_ind1];
    if (rot_ind1 < 0) {
      return;
    }
    int const block_type1 = first_rot_block_type[pose_ind][block_ind1];
    if (block_type1 < 0) {
      return;
    }
    for (int conn_ind1 = 0; conn_ind1 < max_n_conns; conn_ind1++) {
      if (disulfide_conns[block_type1][conn_ind1]) {
        int const block_ind2 =
            pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1]
                                              [0];
        int const conn_ind2 =
            pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1]
                                              [1];
        if (block_ind2 < block_ind1) {
          // Only count disulfides to upper residues;
          // coincidentally handles the case when block_ind2 == -1
          continue;
        }
        int const block_type2 = first_rot_block_type[pose_ind][block_ind2];
        int const rot_ind2 = first_rot_for_block[pose_ind][block_ind2];
        if (rot_ind2 < 0) {
          continue;
        }

        const auto& params = global_params[0];

        int atom_offset1 = rot_coord_offset[rot_ind1];
        int atom_offset2 = rot_coord_offset[rot_ind2];

        // Skip if the other end isn't capable of a disulfide
        // bond -- imagine a CYS with a chemical bond to some
        // kind of funky group; it could happen!
        if (!disulfide_conns[block_type2][conn_ind2]) {
          return;
        }

        // Get the 6 atoms that we need for the disulfides
        auto block1_CA_ind =
            atom_offset1
            + block_type_atom_downstream_of_conn[block_type1][conn_ind1][2];
        auto block1_CB_ind =
            atom_offset1
            + block_type_atom_downstream_of_conn[block_type1][conn_ind1][1];
        auto block1_S_ind =
            atom_offset1
            + block_type_atom_downstream_of_conn[block_type1][conn_ind1][0];
        auto block2_S_ind =
            atom_offset2
            + block_type_atom_downstream_of_conn[block_type2][conn_ind2][0];
        auto block2_CB_ind =
            atom_offset2
            + block_type_atom_downstream_of_conn[block_type2][conn_ind2][1];
        auto block2_CA_ind =
            atom_offset2
            + block_type_atom_downstream_of_conn[block_type2][conn_ind2][2];

        int const Vind1 = output_block_pair_energies ? block_ind1 : 0;
        int const Vind2 = output_block_pair_energies ? block_ind2 : 0;
        // Calculate score and derivatives and put them in the out tensors
        accumulate_disulfide_potential<Real, D>(
            rot_coords,
            pose_ind,
            block_ind1,
            block1_CA_ind,
            block1_CB_ind,
            block1_S_ind,
            block_ind2,
            block2_S_ind,
            block2_CB_ind,
            block2_CA_ind,

            params,

            output_block_pair_energies,
            V[0][pose_ind][Vind1][Vind2],
            dV_dx);
      }
    }
  });

  DeviceDispatch<D>::template forall<launch_t>(
      n_poses * max_n_blocks, eval_energies);

  return {V_t, dV_dx_t};
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto DisulfidePoseScoreDispatch<DeviceDispatch, D, Real, Int>::backward(
    // common params
    TView<Vec<Real, 3>, 1, D> rot_coords,
    TView<Int, 1, D> rot_coord_offset,
    TView<Int, 1, D> pose_ind_for_atom,
    TView<Int, 2, D> first_rot_for_block,
    TView<Int, 2, D> first_rot_block_type,
    TView<Int, 1, D> block_ind_for_rot,
    TView<Int, 1, D> pose_ind_for_rot,
    TView<Int, 1, D> block_type_ind_for_rot,
    TView<Int, 1, D> n_rots_for_pose,
    TView<Int, 1, D> rot_offset_for_pose,
    TView<Int, 2, D> n_rots_for_block,
    TView<Int, 2, D> rot_offset_for_block,
    Int max_n_rots_per_pose,

    TView<Int, 2, D> pose_stack_block_type,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<bool, 2, D> disulfide_conns,
    TView<Int, 3, D> block_type_atom_downstream_of_conn,

    TView<DisulfideGlobalParams<Real>, 1, D> global_params,

    TView<Real, 4, D> dTdV  // n_terms x n_dispatch_total
    ) -> TPack<Vec<Real, 3>, 2, D> {
  int const n_atoms = rot_coords.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);
  int const max_n_conns = disulfide_conns.size(1);

  assert(first_rot_block_type.size(0) == n_poses);
  assert(first_rot_block_type.size(1) == max_n_blocks);

  assert(block_ind_for_rot.size(0) == n_rots);
  assert(pose_ind_for_rot.size(0) == n_rots);
  assert(block_type_ind_for_rot.size(0) == n_rots);

  assert(n_rots_for_pose.size(0) == n_poses);
  assert(rot_offset_for_pose.size(0) == n_poses);

  assert(n_rots_for_block.size(0) == n_poses);
  assert(n_rots_for_block.size(1) == max_n_blocks);

  assert(rot_offset_for_block.size(0) == n_poses);
  assert(rot_offset_for_block.size(1) == max_n_blocks);

  assert(pose_stack_inter_block_connections.size(0) == n_poses);
  assert(pose_stack_inter_block_connections.size(1) == max_n_blocks);
  assert(pose_stack_inter_block_connections.size(2) == max_n_conns);

  assert(dTdV.size(0) == 1);
  assert(dTdV.size(1) == n_poses);
  // backward pass only when block_pair_scoring
  assert(dTdV.size(2) == max_n_blocks);
  assert(dTdV.size(3) == max_n_blocks);

  auto dV_dcoords_t = TPack<Vec<Real, 3>, 2, D>::zeros({1, n_atoms});
  auto dV_dx = dV_dcoords_t.view;

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;

  auto eval_derivs = ([=] TMOL_DEVICE_FUNC(int ind) {
    int const pose_ind = ind / max_n_blocks;
    int const block_ind1 = (ind % max_n_blocks);

    int const rot_ind1 = first_rot_for_block[pose_ind][block_ind1];
    if (rot_ind1 < 0) {
      return;
    }
    int const block_type1 = first_rot_block_type[pose_ind][block_ind1];
    if (block_type1 < 0) {
      return;
    }
    for (int conn_ind1 = 0; conn_ind1 < max_n_conns; conn_ind1++) {
      if (disulfide_conns[block_type1][conn_ind1]) {
        int const block_ind2 =
            pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1]
                                              [0];
        int const conn_ind2 =
            pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1]
                                              [1];
        if (block_ind2 < block_ind1) {
          // Only count disulfides to upper residues;
          // coincidentally handles the case when block_ind2 == -1
          continue;
        }
        int const block_type2 = first_rot_block_type[pose_ind][block_ind2];
        int const rot_ind2 = first_rot_for_block[pose_ind][block_ind2];
        if (rot_ind2 < 0) {
          continue;
        }

        const auto& params = global_params[0];

        int atom_offset1 = rot_coord_offset[rot_ind1];
        int atom_offset2 = rot_coord_offset[rot_ind2];

        if (!disulfide_conns[block_type2][conn_ind2]) {
          return;
        }
        // Get the 6 atoms that we need for the disulfides
        auto block1_CA_ind =
            atom_offset1
            + block_type_atom_downstream_of_conn[block_type1][conn_ind1][2];
        auto block1_CB_ind =
            atom_offset1
            + block_type_atom_downstream_of_conn[block_type1][conn_ind1][1];
        auto block1_S_ind =
            atom_offset1
            + block_type_atom_downstream_of_conn[block_type1][conn_ind1][0];
        auto block2_S_ind =
            atom_offset2
            + block_type_atom_downstream_of_conn[block_type2][conn_ind2][0];
        auto block2_CB_ind =
            atom_offset2
            + block_type_atom_downstream_of_conn[block_type2][conn_ind2][1];
        auto block2_CA_ind =
            atom_offset2
            + block_type_atom_downstream_of_conn[block_type2][conn_ind2][2];

        // Calculate score and derivatives and put them in the out tensors
        accumulate_disulfide_derivs<Real, D>(
            rot_coords,
            block_ind1,
            block1_CA_ind,
            block1_CB_ind,
            block1_S_ind,
            block_ind2,
            block2_S_ind,
            block2_CB_ind,
            block2_CA_ind,

            params,

            dV_dx,
            dTdV[0][pose_ind][block_ind1][block_ind2]);
      }
    }
  });

  DeviceDispatch<D>::template forall<launch_t>(
      n_poses * max_n_blocks, eval_derivs);

  return dV_dcoords_t;
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto DisulfideRotamerScoreDispatch<DeviceDispatch, D, Real, Int>::forward(
    // common params
    TView<Vec<Real, 3>, 1, D> rot_coords,
    TView<Int, 1, D> rot_coord_offset,
    TView<Int, 1, D> pose_ind_for_atom,
    TView<Int, 2, D> first_rot_for_block,
    TView<Int, 2, D> first_rot_block_type,
    TView<Int, 1, D> block_ind_for_rot,
    TView<Int, 1, D> pose_ind_for_rot,
    TView<Int, 1, D> block_type_ind_for_rot,
    TView<Int, 1, D> n_rots_for_pose,
    TView<Int, 1, D> rot_offset_for_pose,
    TView<Int, 2, D> n_rots_for_block,
    TView<Int, 2, D> rot_offset_for_block,
    Int max_n_rots_per_pose,

    TView<Int, 2, D> pose_stack_block_type,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<bool, 2, D> disulfide_conns,
    TView<Int, 3, D> block_type_atom_downstream_of_conn,

    TView<DisulfideGlobalParams<Real>, 1, D> global_params,
    bool output_block_pair_energies,
    bool compute_derivs

    )
    -> std::tuple<
        TPack<Real, 2, D>,
        TPack<Vec<Real, 3>, 2, D>,
        TPack<Int, 2, D>,
        TPack<Int, 2, D>> {
  int const n_atoms = rot_coords.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);
  int const max_n_conns = disulfide_conns.size(1);

  assert(first_rot_block_type.size(0) == n_poses);
  assert(first_rot_block_type.size(1) == max_n_blocks);

  assert(block_ind_for_rot.size(0) == n_rots);
  assert(pose_ind_for_rot.size(0) == n_rots);
  assert(block_type_ind_for_rot.size(0) == n_rots);

  assert(n_rots_for_pose.size(0) == n_poses);
  assert(rot_offset_for_pose.size(0) == n_poses);

  assert(n_rots_for_block.size(0) == n_poses);
  assert(n_rots_for_block.size(1) == max_n_blocks);

  assert(rot_offset_for_block.size(0) == n_poses);
  assert(rot_offset_for_block.size(1) == max_n_blocks);

  assert(pose_stack_inter_block_connections.size(0) == n_poses);
  assert(pose_stack_inter_block_connections.size(1) == max_n_blocks);
  assert(pose_stack_inter_block_connections.size(2) == max_n_conns);

  auto n_energies_for_rot_t = TPack<Int, 1, D>::zeros({n_rots});
  auto n_energies_for_rot = n_energies_for_rot_t.view;

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;
  auto count_dispatch_indices = ([=] TMOL_DEVICE_FUNC(int rot_ind) {
    // count how many other rotamers this rotamer has disulfide bonds to
    int const pose_ind = pose_ind_for_rot[rot_ind];
    int const block_ind = block_ind_for_rot[rot_ind];
    if (pose_ind == -1 || block_ind == -1) {
      return;
    }

    int const rot_block_type = block_type_ind_for_rot[rot_ind];
    if (rot_block_type == -1) {
      return;
    }
    int n_energies = 0;
    for (int conn_index = 0; conn_index < max_n_conns; conn_index++) {
      if (disulfide_conns[rot_block_type][conn_index]) {
        int const other_block_index =
            pose_stack_inter_block_connections[pose_ind][block_ind][conn_index]
                                              [0];

        if (other_block_index < block_ind) {
          // Only count disulfides to upper residues
          // as block_ind >= 0, this will ALSO handle the
          // scenario when other_block_index == -1
          continue;
        }
        int const other_block_n_rots =
            n_rots_for_block[pose_ind][other_block_index];
        n_energies += other_block_n_rots;
      }
    }
    if (n_energies != 0) {
      n_energies_for_rot[rot_ind] = n_energies;
    }
  });
  DeviceDispatch<D>::template forall<launch_t>(n_rots, count_dispatch_indices);
  auto n_energies_for_rot_offset_t = TPack<Int, 1, D>::zeros({n_rots});
  auto n_energies_for_rot_offset = n_energies_for_rot_offset_t.view;
  int n_dispatch_total =
      DeviceDispatch<D>::template scan_and_return_total<mgpu::scan_type_exc>(
          n_energies_for_rot.data(),
          n_energies_for_rot_offset.data(),
          n_rots,
          mgpu::plus_t<Int>());

  TPack<Real, 2, D> V_t;
  auto dispatch_indices_t = TPack<Int, 2, D>::zeros({3, n_dispatch_total});
  auto conns_for_dispatch_indices_t =
      TPack<Int, 2, D>::zeros({2, n_dispatch_total});
  if (output_block_pair_energies) {
    V_t = TPack<Real, 2, D>::zeros({1, n_dispatch_total});
  } else {
    V_t = TPack<Real, 2, D>::zeros({1, n_poses});
  }
  auto dV_dx_t = TPack<Vec<Real, 3>, 2, D>::zeros({1, n_atoms});

  auto V = V_t.view;
  auto dV_dx = dV_dx_t.view;
  auto dispatch_indices = dispatch_indices_t.view;
  // keep track of which connection between residues we are
  // scoring as it is possible (though I've never seen it!)
  // to imagine a disulfide-like chemical bond forming twice
  // between two residues
  auto conns_for_dispatch_indices = conns_for_dispatch_indices_t.view;

  int const max_n_energies_for_rot = DeviceDispatch<D>::reduce(
      n_energies_for_rot.data(), n_rots, mgpu::maximum_t<Int>());

  auto mark_dispatch_indices = ([=] TMOL_DEVICE_FUNC(int ind) {
    int const rot_ind1 = ind / max_n_energies_for_rot;
    // which disulfide connection to upper neighbors is this?
    int const upper_rot_ind = ind % max_n_energies_for_rot;
    int local_rot_ind2 = upper_rot_ind;

    int const pose_ind = pose_ind_for_rot[rot_ind1];
    int const block_ind1 = block_ind_for_rot[rot_ind1];
    if (pose_ind == -1 || block_ind1 == -1) {
      return;
    }
    int const n_energies = n_energies_for_rot[rot_ind1];
    if (upper_rot_ind >= n_energies) {
      // either n_energies is 0 because this rotamer has
      // no RPEs to evaluate or we have some upper neighbor
      // energies to calculate, but fewer than the maximum
      // and thus we are "of the end" of the list.
      return;
    }

    int const block_type1 = block_type_ind_for_rot[rot_ind1];

    int const offset_for_conn = 0;
    for (int conn_ind1 = 0; conn_ind1 < max_n_conns; conn_ind1++) {
      if (disulfide_conns[block_type1][conn_ind1]) {
        int const block_ind2 =
            pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1]
                                              [0];
        if (block_ind2 < block_ind1) {
          // Only count disulfides to upper residues
          continue;
        }
        int const conn_ind2 =
            pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1]
                                              [1];

        int const n_rots2 = n_rots_for_block[pose_ind][block_ind2];
        if (local_rot_ind2 < n_rots2) {
          // We have figured out which upper rotamer this
          // index is referring to
          int const rot_ind2 =
              first_rot_for_block[pose_ind][block_ind2] + local_rot_ind2;
          int const sparse_index =
              n_energies_for_rot_offset[rot_ind1] + upper_rot_ind;

          dispatch_indices[0][sparse_index] = pose_ind;
          dispatch_indices[1][sparse_index] = rot_ind1;
          dispatch_indices[2][sparse_index] = rot_ind2;
          conns_for_dispatch_indices[0][sparse_index] = conn_ind1;
          conns_for_dispatch_indices[1][sparse_index] = conn_ind2;
          break;
        } else {
          // this code is only reachable if we have some block
          // type wherein there are two or more disulfides -- imagine
          // something like valine but two sulfurs instead of two methyls
          // -- and residue i has disulfide bonds to both residues
          // j and k; so we evaluate i's rotamer's energies with j's
          // rotamers and also and i's rotamer's energies with k's
          // rotamers. Well, we have walked off the end of the list for
          // residue j, so we must instead go to the next connection
          // for residue k.
          local_rot_ind2 -= n_rots2;
        }
      }
    }
  });
  DeviceDispatch<D>::template forall<launch_t>(
      n_rots * max_n_energies_for_rot, mark_dispatch_indices);

  auto eval_energies = ([=] TMOL_DEVICE_FUNC(int dispatch_ind) {
    int const pose_ind = dispatch_indices[0][dispatch_ind];
    int const rot_ind1 = dispatch_indices[1][dispatch_ind];
    int const rot_ind2 = dispatch_indices[2][dispatch_ind];

    int const block_ind1 = block_ind_for_rot[rot_ind1];
    int const block_ind2 = block_ind_for_rot[rot_ind2];
    int const block_type1 = block_type_ind_for_rot[rot_ind1];
    int const block_type2 = block_type_ind_for_rot[rot_ind2];

    const auto& params = global_params[0];

    int atom_offset1 = rot_coord_offset[rot_ind1];
    int atom_offset2 = rot_coord_offset[rot_ind2];
    int conn_ind1 = conns_for_dispatch_indices[0][dispatch_ind];
    int conn_ind2 = conns_for_dispatch_indices[1][dispatch_ind];

    // We are scoring only one connection!

    // Skip if the other end isn't capable of a disulfide
    // bond -- imagine a CYS with a chemical bond to some
    // kind of funky group; it could happen!
    if (!disulfide_conns[block_type2][conn_ind2]) {
      return;
    }

    // Get the 6 atoms that we need for the disulfides
    auto block1_CA_ind =
        atom_offset1
        + block_type_atom_downstream_of_conn[block_type1][conn_ind1][2];
    auto block1_CB_ind =
        atom_offset1
        + block_type_atom_downstream_of_conn[block_type1][conn_ind1][1];
    auto block1_S_ind =
        atom_offset1
        + block_type_atom_downstream_of_conn[block_type1][conn_ind1][0];
    auto block2_S_ind =
        atom_offset2
        + block_type_atom_downstream_of_conn[block_type2][conn_ind2][0];
    auto block2_CB_ind =
        atom_offset2
        + block_type_atom_downstream_of_conn[block_type2][conn_ind2][1];
    auto block2_CA_ind =
        atom_offset2
        + block_type_atom_downstream_of_conn[block_type2][conn_ind2][2];

    // Calculate score and derivatives and put them in the out tensors
    accumulate_disulfide_potential<Real, D>(
        rot_coords,
        pose_ind,
        block_ind1,
        block1_CA_ind,
        block1_CB_ind,
        block1_S_ind,
        block_ind2,
        block2_S_ind,
        block2_CB_ind,
        block2_CA_ind,

        params,

        output_block_pair_energies,
        V[0][dispatch_ind],
        dV_dx);
  });
  DeviceDispatch<D>::template forall<launch_t>(n_dispatch_total, eval_energies);

  return {V_t, dV_dx_t, dispatch_indices_t, conns_for_dispatch_indices_t};
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto DisulfideRotamerScoreDispatch<DeviceDispatch, D, Real, Int>::backward(
    // common params
    TView<Vec<Real, 3>, 1, D> rot_coords,
    TView<Int, 1, D> rot_coord_offset,
    TView<Int, 1, D> pose_ind_for_atom,
    TView<Int, 2, D> first_rot_for_block,
    TView<Int, 2, D> first_rot_block_type,
    TView<Int, 1, D> block_ind_for_rot,
    TView<Int, 1, D> pose_ind_for_rot,
    TView<Int, 1, D> block_type_ind_for_rot,
    TView<Int, 1, D> n_rots_for_pose,
    TView<Int, 1, D> rot_offset_for_pose,
    TView<Int, 2, D> n_rots_for_block,
    TView<Int, 2, D> rot_offset_for_block,
    Int max_n_rots_per_pose,

    TView<Int, 2, D> pose_stack_block_type,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<bool, 2, D> disulfide_conns,
    TView<Int, 3, D> block_type_atom_downstream_of_conn,

    TView<DisulfideGlobalParams<Real>, 1, D> global_params,
    TView<Int, 2, D> dispatch_indices,            // from forward pass
    TView<Int, 2, D> conns_for_dispatch_indices,  // from forward pass

    TView<Real, 2, D> dTdV  // n_terms x n_dispatch_total
    ) -> TPack<Vec<Real, 3>, 2, D> {
  int const n_atoms = rot_coords.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);
  int const max_n_conns = disulfide_conns.size(1);
  int n_dispatch_total = dispatch_indices.size(1);

  assert(first_rot_block_type.size(0) == n_poses);
  assert(first_rot_block_type.size(1) == max_n_blocks);

  assert(block_ind_for_rot.size(0) == n_rots);
  assert(pose_ind_for_rot.size(0) == n_rots);
  assert(block_type_ind_for_rot.size(0) == n_rots);

  assert(n_rots_for_pose.size(0) == n_poses);
  assert(rot_offset_for_pose.size(0) == n_poses);

  assert(n_rots_for_block.size(0) == n_poses);
  assert(n_rots_for_block.size(1) == max_n_blocks);

  assert(rot_offset_for_block.size(0) == n_poses);
  assert(rot_offset_for_block.size(1) == max_n_blocks);

  assert(pose_stack_inter_block_connections.size(0) == n_poses);
  assert(pose_stack_inter_block_connections.size(1) == max_n_blocks);
  assert(pose_stack_inter_block_connections.size(2) == max_n_conns);

  assert(dispatch_indices.size(0) == 3);
  assert(conns_for_dispatch_indices.size(0) == 2);
  assert(conns_for_dispatch_indices.size(1) == n_dispatch_total);
  assert(dTdV.size(0) == 1);
  assert(
      dTdV.size(1)
      == n_dispatch_total);  // backward pass only when block_pair_scoring

  auto dV_dcoords_t = TPack<Vec<Real, 3>, 2, D>::zeros({1, n_atoms});
  auto dV_dx = dV_dcoords_t.view;

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;

  auto eval_derivs = ([=] TMOL_DEVICE_FUNC(int dispatch_ind) {
    int const pose_ind = dispatch_indices[0][dispatch_ind];
    int const rot_ind1 = dispatch_indices[1][dispatch_ind];
    int const rot_ind2 = dispatch_indices[2][dispatch_ind];

    int const block_ind1 = block_ind_for_rot[rot_ind1];
    int const block_ind2 = block_ind_for_rot[rot_ind2];
    int const block_type1 = block_type_ind_for_rot[rot_ind1];
    int const block_type2 = block_type_ind_for_rot[rot_ind2];

    const auto& params = global_params[0];

    int atom_offset1 = rot_coord_offset[rot_ind1];
    int atom_offset2 = rot_coord_offset[rot_ind2];
    int conn_ind1 = conns_for_dispatch_indices[0][dispatch_ind];
    int conn_ind2 = conns_for_dispatch_indices[1][dispatch_ind];

    if (!disulfide_conns[block_type2][conn_ind2]) {
      return;
    }
    // Get the 6 atoms that we need for the disulfides
    auto block1_CA_ind =
        atom_offset1
        + block_type_atom_downstream_of_conn[block_type1][conn_ind1][2];
    auto block1_CB_ind =
        atom_offset1
        + block_type_atom_downstream_of_conn[block_type1][conn_ind1][1];
    auto block1_S_ind =
        atom_offset1
        + block_type_atom_downstream_of_conn[block_type1][conn_ind1][0];
    auto block2_S_ind =
        atom_offset2
        + block_type_atom_downstream_of_conn[block_type2][conn_ind2][0];
    auto block2_CB_ind =
        atom_offset2
        + block_type_atom_downstream_of_conn[block_type2][conn_ind2][1];
    auto block2_CA_ind =
        atom_offset2
        + block_type_atom_downstream_of_conn[block_type2][conn_ind2][2];

    // Calculate score and derivatives and put them in the out tensors
    accumulate_disulfide_derivs<Real, D>(
        rot_coords,
        block_ind1,
        block1_CA_ind,
        block1_CB_ind,
        block1_S_ind,
        block_ind2,
        block2_S_ind,
        block2_CB_ind,
        block2_CA_ind,

        params,

        dV_dx,
        dTdV[0][dispatch_ind]);
  });

  DeviceDispatch<D>::template forall<launch_t>(n_dispatch_total, eval_derivs);

  return dV_dcoords_t;
}

}  // namespace potentials
}  // namespace disulfide
}  // namespace score
}  // namespace tmol
