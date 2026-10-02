#pragma once

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/counting.hh>
#include <tmol/score/common/launch_box_macros.hh>

#include <tmol/score/metal/potentials/metal_coordination_pose_score.hh>

#include <moderngpu/operators.hxx>

#include "params.hh"
#include "potentials.hh"

namespace tmol {
namespace score {
namespace metal {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

// conn_virt values that are not a virtual atom index
constexpr int NOT_A_SITE = -2;

// Each metal block owns its site connections: the fan is a one-body energy on
// the block, and each filled site is a two-body energy between the site's
// metal atom (a block may hold several) and the block across the connection.
// Donor blocks never initiate, so every site is scored once.

// Bridges: two metals in different blocks bonded to one donor atom. Each is
// visited once, from block1 as the metal whose own atom it charges:
//  - through a donor in another block, from the lower of the two metal
//    blocks (metal site conn1 on block1 -> donor -> another site on "other");
//  - through a donor in block1 itself, an internal satisfier of block1's
//    metals that also fills a site on "other".
// f(code, other_block, other_type, metal1_atom, metal2_atom, d1, d2, key);
// code names the bridge for bridge_from_code
template <typename Int, typename Real, tmol::Device D, typename Func>
TMOL_DEVICE_FUNC void for_each_bridge(
    int pose_ind,
    int block_ind1,
    int block_type1,
    int max_n_conns,
    int max_n_internal,
    TView<Int, 2, D> first_rot_block_type,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Int, 2, D> conn_atom,
    TView<Int, 2, D> conn_metal,
    TView<Int, 2, D> conn_virt,
    TView<Int, 2, D> conn_key,
    TView<MetalSiteParams<Real>, 3, D> site_params,
    TView<Int, 3, D> bridge_internal_metal,
    TView<Real, 3, D> bridge_internal_d0,
    Func f) {
  for (int conn1 = 0; conn1 < max_n_conns; conn1++) {
    int const partner =
        pose_stack_inter_block_connections[pose_ind][block_ind1][conn1][0];
    if (partner < 0) {
      continue;
    }
    int const pconn =
        pose_stack_inter_block_connections[pose_ind][block_ind1][conn1][1];
    int const ptype = first_rot_block_type[pose_ind][partner];
    if (ptype < 0) {
      continue;
    }
    if (conn_virt[block_type1][conn1] != NOT_A_SITE) {
      int const key = conn_key[ptype][pconn];
      if (key < 0) {
        continue;
      }
      int const donor_atom = conn_atom[ptype][pconn];
      for (int c = 0; c < max_n_conns; c++) {
        if (c == pconn || conn_atom[ptype][c] != donor_atom) {
          continue;
        }
        int const other =
            pose_stack_inter_block_connections[pose_ind][partner][c][0];
        if (other <= block_ind1) {
          continue;
        }
        int const oconn =
            pose_stack_inter_block_connections[pose_ind][partner][c][1];
        int const otype = first_rot_block_type[pose_ind][other];
        if (otype < 0 || conn_virt[otype][oconn] == NOT_A_SITE) {
          continue;
        }
        f(conn1 * max_n_conns + c,
          other,
          otype,
          conn_metal[block_type1][conn1],
          conn_metal[otype][oconn],
          site_params[block_type1][conn1][key].d0,
          site_params[otype][oconn][key].d0,
          key);
      }
    } else if (conn_virt[ptype][pconn] != NOT_A_SITE) {
      int const key = conn_key[block_type1][conn1];
      if (key < 0) {
        continue;
      }
      for (int k = 0; k < max_n_internal; k++) {
        int const metal = bridge_internal_metal[block_type1][conn1][k];
        if (metal < 0) {
          break;
        }
        f(max_n_conns * max_n_conns + conn1 * max_n_internal + k,
          partner,
          ptype,
          metal,
          conn_metal[ptype][pconn],
          bridge_internal_d0[block_type1][conn1][k],
          site_params[ptype][pconn][key].d0,
          key);
      }
    }
  }
}

// The bridge for_each_bridge named by code, between rotamers of block_ind1
// (block type type1) and of the other metal's block (block type type2).
template <typename Int, typename Real, tmol::Device D>
struct Bridge {
  bool valid;
  int metal1;
  int metal2;
  Real d1;
  Real d2;
  int key;
};

template <typename Int, typename Real, tmol::Device D>
TMOL_DEVICE_FUNC Bridge<Int, Real, D> bridge_from_code(
    int code,
    int pose_ind,
    int block_ind1,
    int type1,
    int type2,
    int max_n_conns,
    int max_n_internal,
    TView<Int, 2, D> first_rot_block_type,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Int, 2, D> conn_metal,
    TView<Int, 2, D> conn_key,
    TView<MetalSiteParams<Real>, 3, D> site_params,
    TView<Int, 3, D> bridge_internal_metal,
    TView<Real, 3, D> bridge_internal_d0) {
  Bridge<Int, Real, D> out{false, -1, -1, 0, 0, -1};
  int const n_external = max_n_conns * max_n_conns;
  if (code < n_external) {
    int const conn1 = code / max_n_conns;
    int const c = code % max_n_conns;
    int const partner =
        pose_stack_inter_block_connections[pose_ind][block_ind1][conn1][0];
    int const pconn =
        pose_stack_inter_block_connections[pose_ind][block_ind1][conn1][1];
    int const oconn =
        pose_stack_inter_block_connections[pose_ind][partner][c][1];
    int const key = conn_key[first_rot_block_type[pose_ind][partner]][pconn];
    if (key < 0) {
      return out;
    }
    out = {
        true,
        conn_metal[type1][conn1],
        conn_metal[type2][oconn],
        site_params[type1][conn1][key].d0,
        site_params[type2][oconn][key].d0,
        key};
  } else {
    int const conn1 = (code - n_external) / max_n_internal;
    int const k = (code - n_external) % max_n_internal;
    int const pconn =
        pose_stack_inter_block_connections[pose_ind][block_ind1][conn1][1];
    int const key = conn_key[type1][conn1];
    int const metal = bridge_internal_metal[type1][conn1][k];
    if (key < 0 || metal < 0) {
      return out;
    }
    out = {
        true,
        metal,
        conn_metal[type2][pconn],
        bridge_internal_d0[type1][conn1][k],
        site_params[type2][pconn][key].d0,
        key};
  }
  return out;
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto MetalCoordinationPoseScoreDispatch<DeviceDispatch, D, Real, Int>::forward(
    ContextManager& mgr,
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

    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Int, 2, D> conn_atom,
    TView<Int, 2, D> conn_metal,
    TView<Int, 2, D> conn_virt,
    TView<Int, 2, D> conn_key,
    TView<MetalSiteParams<Real>, 3, D> site_params,
    TView<Vec<Int, 2>, 2, D> fan_atoms,
    TView<MetalFanParams<Real>, 2, D> fan_params,
    TView<Int, 3, D> bridge_internal_metal,
    TView<Real, 3, D> bridge_internal_d0,
    TView<MetalBridgeParams<Real>, 1, D> bridge_params,
    bool output_block_pair_energies,
    bool compute_derivs)
    -> std::tuple<TPack<Real, 4, D>, TPack<Vec<Real, 3>, 2, D>> {
  int const n_atoms = rot_coords.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);
  int const max_n_conns = conn_virt.size(1);
  int const max_n_fan = fan_atoms.size(1);
  int const max_n_internal = bridge_internal_metal.size(2);

  assert(pose_stack_inter_block_connections.size(0) == n_poses);
  assert(pose_stack_inter_block_connections.size(1) == max_n_blocks);
  assert(pose_stack_inter_block_connections.size(2) == max_n_conns);

  LAUNCH_BOX_32;

  int const n_V = output_block_pair_energies ? max_n_blocks : 1;
  auto V_t = TPack<Real, 4, D>::zeros({1, n_poses, n_V, n_V});
  auto dV_dx_t = compute_derivs
                     ? TPack<Vec<Real, 3>, 2, D>::zeros({1, n_atoms})
                     : TPack<Vec<Real, 3>, 2, D>::empty({1, n_atoms});
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
    int const offset1 = rot_coord_offset[rot_ind1];
    int const Vind1 = output_block_pair_energies ? block_ind1 : 0;

    for (int i = 0; i < max_n_fan; i++) {
      Vec<Int, 2> const pair = fan_atoms[block_type1][i];
      if (pair[0] < 0) {
        break;
      }
      accumulate_metal_fan_pair<Real, D>(
          rot_coords,
          offset1 + pair[0],
          offset1 + pair[1],
          fan_params[block_type1][i],
          dV_dx,
          compute_derivs,
          Real(1),
          &V[0][pose_ind][Vind1][Vind1]);
    }

    for (int conn1 = 0; conn1 < max_n_conns; conn1++) {
      int const virt = conn_virt[block_type1][conn1];
      if (virt == NOT_A_SITE) {
        continue;
      }
      int const block_ind2 =
          pose_stack_inter_block_connections[pose_ind][block_ind1][conn1][0];
      if (block_ind2 < 0) {
        continue;
      }
      int const conn2 =
          pose_stack_inter_block_connections[pose_ind][block_ind1][conn1][1];
      int const rot_ind2 = first_rot_for_block[pose_ind][block_ind2];
      int const block_type2 = first_rot_block_type[pose_ind][block_ind2];
      if (rot_ind2 < 0 || block_type2 < 0) {
        continue;
      }
      int const key = conn_key[block_type2][conn2];
      if (key < 0) {
        continue;
      }
      // upper triangle, as the other pair terms report it
      int const lo = block_ind1 < block_ind2 ? block_ind1 : block_ind2;
      int const hi = block_ind1 < block_ind2 ? block_ind2 : block_ind1;
      int const Vlo = output_block_pair_energies ? lo : 0;
      int const Vhi = output_block_pair_energies ? hi : 0;
      accumulate_metal_site<Real, D>(
          rot_coords,
          offset1 + conn_metal[block_type1][conn1],
          virt >= 0 ? offset1 + virt : -1,
          rot_coord_offset[rot_ind2] + conn_atom[block_type2][conn2],
          site_params[block_type1][conn1][key],
          dV_dx,
          compute_derivs,
          Real(1),
          &V[0][pose_ind][Vlo][Vhi]);
    }

    for_each_bridge<Int, Real, D>(
        pose_ind,
        block_ind1,
        block_type1,
        max_n_conns,
        max_n_internal,
        first_rot_block_type,
        pose_stack_inter_block_connections,
        conn_atom,
        conn_metal,
        conn_virt,
        conn_key,
        site_params,
        bridge_internal_metal,
        bridge_internal_d0,
        [&](int,
            int other,
            int,
            int metal1,
            int metal2,
            Real d1,
            Real d2,
            int key) {
          int const lo = block_ind1 < other ? block_ind1 : other;
          int const hi = block_ind1 < other ? other : block_ind1;
          int const Vlo = output_block_pair_energies ? lo : 0;
          int const Vhi = output_block_pair_energies ? hi : 0;
          accumulate_metal_bridge<Real, D>(
              rot_coords,
              offset1 + metal1,
              rot_coord_offset[first_rot_for_block[pose_ind][other]] + metal2,
              d1,
              d2,
              bridge_params[key],
              dV_dx,
              compute_derivs,
              Real(1),
              &V[0][pose_ind][Vlo][Vhi]);
        });
  });

  DeviceDispatch<D>::template forall_grouped<launch_t>(
      mgr, n_poses, max_n_blocks, eval_energies);

  return {V_t, dV_dx_t};
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto MetalCoordinationPoseScoreDispatch<DeviceDispatch, D, Real, Int>::backward(
    ContextManager& mgr,
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

    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Int, 2, D> conn_atom,
    TView<Int, 2, D> conn_metal,
    TView<Int, 2, D> conn_virt,
    TView<Int, 2, D> conn_key,
    TView<MetalSiteParams<Real>, 3, D> site_params,
    TView<Vec<Int, 2>, 2, D> fan_atoms,
    TView<MetalFanParams<Real>, 2, D> fan_params,
    TView<Int, 3, D> bridge_internal_metal,
    TView<Real, 3, D> bridge_internal_d0,
    TView<MetalBridgeParams<Real>, 1, D> bridge_params,
    TView<Real, 4, D> dTdV) -> TPack<Vec<Real, 3>, 2, D> {
  int const n_atoms = rot_coords.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);
  int const max_n_conns = conn_virt.size(1);
  int const max_n_fan = fan_atoms.size(1);
  int const max_n_internal = bridge_internal_metal.size(2);

  assert(dTdV.size(0) == 1);
  assert(dTdV.size(1) == n_poses);
  assert(dTdV.size(2) == max_n_blocks);
  assert(dTdV.size(3) == max_n_blocks);

  auto dV_dx_t = TPack<Vec<Real, 3>, 2, D>::zeros({1, n_atoms});
  auto dV_dx = dV_dx_t.view;

  LAUNCH_BOX_32;

  auto eval_derivs = ([=] TMOL_DEVICE_FUNC(int ind) {
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
    int const offset1 = rot_coord_offset[rot_ind1];

    Real const dTdV_fan = dTdV[0][pose_ind][block_ind1][block_ind1];
    if (dTdV_fan != 0) {
      for (int i = 0; i < max_n_fan; i++) {
        Vec<Int, 2> const pair = fan_atoms[block_type1][i];
        if (pair[0] < 0) {
          break;
        }
        accumulate_metal_fan_pair<Real, D>(
            rot_coords,
            offset1 + pair[0],
            offset1 + pair[1],
            fan_params[block_type1][i],
            dV_dx,
            true,
            dTdV_fan,
            nullptr);
      }
    }

    for (int conn1 = 0; conn1 < max_n_conns; conn1++) {
      int const virt = conn_virt[block_type1][conn1];
      if (virt == NOT_A_SITE) {
        continue;
      }
      int const block_ind2 =
          pose_stack_inter_block_connections[pose_ind][block_ind1][conn1][0];
      if (block_ind2 < 0) {
        continue;
      }
      int const lo = block_ind1 < block_ind2 ? block_ind1 : block_ind2;
      int const hi = block_ind1 < block_ind2 ? block_ind2 : block_ind1;
      Real const dTdV_site = dTdV[0][pose_ind][lo][hi];
      if (dTdV_site == 0) {
        continue;
      }
      int const conn2 =
          pose_stack_inter_block_connections[pose_ind][block_ind1][conn1][1];
      int const rot_ind2 = first_rot_for_block[pose_ind][block_ind2];
      int const block_type2 = first_rot_block_type[pose_ind][block_ind2];
      if (rot_ind2 < 0 || block_type2 < 0) {
        continue;
      }
      int const key = conn_key[block_type2][conn2];
      if (key < 0) {
        continue;
      }
      accumulate_metal_site<Real, D>(
          rot_coords,
          offset1 + conn_metal[block_type1][conn1],
          virt >= 0 ? offset1 + virt : -1,
          rot_coord_offset[rot_ind2] + conn_atom[block_type2][conn2],
          site_params[block_type1][conn1][key],
          dV_dx,
          true,
          dTdV_site,
          nullptr);
    }

    for_each_bridge<Int, Real, D>(
        pose_ind,
        block_ind1,
        block_type1,
        max_n_conns,
        max_n_internal,
        first_rot_block_type,
        pose_stack_inter_block_connections,
        conn_atom,
        conn_metal,
        conn_virt,
        conn_key,
        site_params,
        bridge_internal_metal,
        bridge_internal_d0,
        [&](int,
            int other,
            int,
            int metal1,
            int metal2,
            Real d1,
            Real d2,
            int key) {
          int const lo = block_ind1 < other ? block_ind1 : other;
          int const hi = block_ind1 < other ? other : block_ind1;
          Real const dTdV_bridge = dTdV[0][pose_ind][lo][hi];
          if (dTdV_bridge == 0) {
            return;
          }
          accumulate_metal_bridge<Real, D>(
              rot_coords,
              offset1 + metal1,
              rot_coord_offset[first_rot_for_block[pose_ind][other]] + metal2,
              d1,
              d2,
              bridge_params[key],
              dV_dx,
              true,
              dTdV_bridge,
              nullptr);
        });
  });

  DeviceDispatch<D>::template forall_grouped<launch_t>(
      mgr, n_poses, max_n_blocks, eval_derivs);

  return dV_dx_t;
}

// terms_for_dispatch rows: metal rotamer, partner rotamer, metal connection;
// a connection of -1 marks the metal's fan, scored as a one-body entry.
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto MetalCoordinationRotamerScoreDispatch<DeviceDispatch, D, Real, Int>::
    forward(
        ContextManager& mgr,
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

        TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
        TView<Int, 2, D> conn_atom,
        TView<Int, 2, D> conn_metal,
        TView<Int, 2, D> conn_virt,
        TView<Int, 2, D> conn_key,
        TView<MetalSiteParams<Real>, 3, D> site_params,
        TView<Vec<Int, 2>, 2, D> fan_atoms,
        TView<MetalFanParams<Real>, 2, D> fan_params,
        TView<Int, 3, D> bridge_internal_metal,
        TView<Real, 3, D> bridge_internal_d0,
        TView<MetalBridgeParams<Real>, 1, D> bridge_params,
        bool output_block_pair_energies,
        bool compute_derivs)
        -> std::tuple<
            TPack<Real, 2, D>,
            TPack<Vec<Real, 3>, 2, D>,
            TPack<Int, 2, D>,
            TPack<Int, 2, D>> {
  int const n_atoms = rot_coords.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_conns = conn_virt.size(1);
  int const max_n_fan = fan_atoms.size(1);
  int const max_n_internal = bridge_internal_metal.size(2);

  auto n_energies_for_rot_t = TPack<int64_t, 1, D>::zeros({n_rots});
  auto n_energies_for_rot = n_energies_for_rot_t.view;

  LAUNCH_BOX_32;

  auto count_dispatch_indices = ([=] TMOL_DEVICE_FUNC(int rot_ind) {
    int const pose_ind = pose_ind_for_rot[rot_ind];
    int const block_ind = block_ind_for_rot[rot_ind];
    int const block_type = block_type_ind_for_rot[rot_ind];
    if (pose_ind < 0 || block_ind < 0 || block_type < 0) {
      return;
    }
    int64_t n_energies = fan_atoms[block_type][0][0] >= 0 ? 1 : 0;
    for (int conn = 0; conn < max_n_conns; conn++) {
      if (conn_virt[block_type][conn] == NOT_A_SITE) {
        continue;
      }
      int const other_block =
          pose_stack_inter_block_connections[pose_ind][block_ind][conn][0];
      if (other_block < 0) {
        continue;
      }
      n_energies += n_rots_for_block[pose_ind][other_block];
    }
    for_each_bridge<Int, Real, D>(
        pose_ind,
        block_ind,
        block_type,
        max_n_conns,
        max_n_internal,
        first_rot_block_type,
        pose_stack_inter_block_connections,
        conn_atom,
        conn_metal,
        conn_virt,
        conn_key,
        site_params,
        bridge_internal_metal,
        bridge_internal_d0,
        [&](int, int other, int, int, int, Real, Real, int) {
          n_energies += n_rots_for_block[pose_ind][other];
        });
    n_energies_for_rot[rot_ind] = n_energies;
  });
  DeviceDispatch<D>::template forall<launch_t>(
      mgr, n_rots, count_dispatch_indices);

  int64_t const max_n_energies_for_rot_64 = DeviceDispatch<D>::reduce(
      mgr, n_energies_for_rot.data(), n_rots, mgpu::maximum_t<int64_t>());
  int const max_n_energies_for_rot = score::common::checked_dispatch_size(
      max_n_energies_for_rot_64, "metal coordination per-rotamer dispatch");
  int const candidate_dispatch = score::common::checked_dispatch_product(
      n_rots, max_n_energies_for_rot, "metal coordination candidate dispatch");

  auto n_energies_for_rot_offset_t = TPack<int64_t, 1, D>::zeros({n_rots});
  auto n_energies_for_rot_offset = n_energies_for_rot_offset_t.view;
  int64_t const n_dispatch_total_64 =
      DeviceDispatch<D>::template scan_and_return_total<mgpu::scan_type_exc>(
          mgr,
          n_energies_for_rot.data(),
          n_energies_for_rot_offset.data(),
          n_rots,
          mgpu::plus_t<int64_t>());
  int const n_dispatch_total = score::common::checked_dispatch_size(
      n_dispatch_total_64, "metal coordination output dispatch");

  auto dispatch_indices_t = TPack<Int, 2, D>::zeros({3, n_dispatch_total});
  auto terms_for_dispatch_t = TPack<Int, 2, D>::zeros({3, n_dispatch_total});
  auto V_t = output_block_pair_energies
                 ? TPack<Real, 2, D>::zeros({1, n_dispatch_total})
                 : TPack<Real, 2, D>::zeros({1, n_poses});
  auto dV_dx_t = compute_derivs
                     ? TPack<Vec<Real, 3>, 2, D>::zeros({1, n_atoms})
                     : TPack<Vec<Real, 3>, 2, D>::empty({1, n_atoms});

  auto V = V_t.view;
  auto dV_dx = dV_dx_t.view;
  auto dispatch_indices = dispatch_indices_t.view;
  auto terms_for_dispatch = terms_for_dispatch_t.view;

  auto mark_dispatch_indices = ([=] TMOL_DEVICE_FUNC(int ind) {
    int const rot_ind1 = ind / max_n_energies_for_rot;
    int const local_ind = ind % max_n_energies_for_rot;
    if (local_ind >= n_energies_for_rot[rot_ind1]) {
      return;
    }
    int const pose_ind = pose_ind_for_rot[rot_ind1];
    int const block_ind1 = block_ind_for_rot[rot_ind1];
    int const block_type1 = block_type_ind_for_rot[rot_ind1];
    int const sparse_ind = n_energies_for_rot_offset[rot_ind1] + local_ind;
    dispatch_indices[0][sparse_ind] = pose_ind;

    int remaining = local_ind;
    if (fan_atoms[block_type1][0][0] >= 0) {
      if (remaining == 0) {
        dispatch_indices[1][sparse_ind] = rot_ind1;
        dispatch_indices[2][sparse_ind] = rot_ind1;
        terms_for_dispatch[0][sparse_ind] = rot_ind1;
        terms_for_dispatch[1][sparse_ind] = rot_ind1;
        terms_for_dispatch[2][sparse_ind] = -1;
        return;
      }
      remaining -= 1;
    }
    for (int conn1 = 0; conn1 < max_n_conns; conn1++) {
      if (conn_virt[block_type1][conn1] == NOT_A_SITE) {
        continue;
      }
      int const block_ind2 =
          pose_stack_inter_block_connections[pose_ind][block_ind1][conn1][0];
      if (block_ind2 < 0) {
        continue;
      }
      int const n_rots2 = n_rots_for_block[pose_ind][block_ind2];
      if (remaining < n_rots2) {
        int const rot_ind2 =
            first_rot_for_block[pose_ind][block_ind2] + remaining;
        // lower block first, as the other pair terms report it
        bool const metal_first = block_ind1 < block_ind2;
        dispatch_indices[1][sparse_ind] = metal_first ? rot_ind1 : rot_ind2;
        dispatch_indices[2][sparse_ind] = metal_first ? rot_ind2 : rot_ind1;
        terms_for_dispatch[0][sparse_ind] = rot_ind1;
        terms_for_dispatch[1][sparse_ind] = rot_ind2;
        terms_for_dispatch[2][sparse_ind] = conn1;
        return;
      }
      remaining -= n_rots2;
    }
    // bridges follow, coded -2 - code: -1 is the fan
    bool placed = false;
    for_each_bridge<Int, Real, D>(
        pose_ind,
        block_ind1,
        block_type1,
        max_n_conns,
        max_n_internal,
        first_rot_block_type,
        pose_stack_inter_block_connections,
        conn_atom,
        conn_metal,
        conn_virt,
        conn_key,
        site_params,
        bridge_internal_metal,
        bridge_internal_d0,
        [&](int code, int other, int, int, int, Real, Real, int) {
          if (placed) {
            return;
          }
          int const n_rots2 = n_rots_for_block[pose_ind][other];
          if (remaining < n_rots2) {
            int const rot_ind2 =
                first_rot_for_block[pose_ind][other] + remaining;
            bool const first = block_ind1 < other;
            dispatch_indices[1][sparse_ind] = first ? rot_ind1 : rot_ind2;
            dispatch_indices[2][sparse_ind] = first ? rot_ind2 : rot_ind1;
            terms_for_dispatch[0][sparse_ind] = rot_ind1;
            terms_for_dispatch[1][sparse_ind] = rot_ind2;
            terms_for_dispatch[2][sparse_ind] = -2 - code;
            placed = true;
            return;
          }
          remaining -= n_rots2;
        });
  });
  DeviceDispatch<D>::template forall<launch_t>(
      mgr, candidate_dispatch, mark_dispatch_indices);

  auto eval_energies = ([=] TMOL_DEVICE_FUNC(int dispatch_ind) {
    int const pose_ind = dispatch_indices[0][dispatch_ind];
    int const rot_ind1 = terms_for_dispatch[0][dispatch_ind];
    int const rot_ind2 = terms_for_dispatch[1][dispatch_ind];
    int const conn1 = terms_for_dispatch[2][dispatch_ind];
    int const block_type1 = block_type_ind_for_rot[rot_ind1];
    int const offset1 = rot_coord_offset[rot_ind1];
    Real* V_out =
        output_block_pair_energies ? &V[0][dispatch_ind] : &V[0][pose_ind];

    if (conn1 <= -2) {
      int const bridge_pose = pose_ind_for_rot[rot_ind1];
      auto const bridge = bridge_from_code<Int, Real, D>(
          -2 - conn1,
          bridge_pose,
          block_ind_for_rot[rot_ind1],
          block_type1,
          block_type_ind_for_rot[rot_ind2],
          max_n_conns,
          max_n_internal,
          first_rot_block_type,
          pose_stack_inter_block_connections,
          conn_metal,
          conn_key,
          site_params,
          bridge_internal_metal,
          bridge_internal_d0);
      if (bridge.valid) {
        accumulate_metal_bridge<Real, D>(
            rot_coords,
            offset1 + bridge.metal1,
            rot_coord_offset[rot_ind2] + bridge.metal2,
            bridge.d1,
            bridge.d2,
            bridge_params[bridge.key],
            dV_dx,
            compute_derivs,
            Real(1),
            V_out);
      }
      return;
    }

    if (conn1 < 0) {
      for (int i = 0; i < max_n_fan; i++) {
        Vec<Int, 2> const pair = fan_atoms[block_type1][i];
        if (pair[0] < 0) {
          break;
        }
        accumulate_metal_fan_pair<Real, D>(
            rot_coords,
            offset1 + pair[0],
            offset1 + pair[1],
            fan_params[block_type1][i],
            dV_dx,
            compute_derivs,
            Real(1),
            V_out);
      }
      return;
    }

    int const block_ind1 = block_ind_for_rot[rot_ind1];
    int const conn2 =
        pose_stack_inter_block_connections[pose_ind][block_ind1][conn1][1];
    int const block_type2 = block_type_ind_for_rot[rot_ind2];
    int const key = conn_key[block_type2][conn2];
    if (key < 0) {
      return;
    }
    int const virt = conn_virt[block_type1][conn1];
    accumulate_metal_site<Real, D>(
        rot_coords,
        offset1 + conn_metal[block_type1][conn1],
        virt >= 0 ? offset1 + virt : -1,
        rot_coord_offset[rot_ind2] + conn_atom[block_type2][conn2],
        site_params[block_type1][conn1][key],
        dV_dx,
        compute_derivs,
        Real(1),
        V_out);
  });
  DeviceDispatch<D>::template forall<launch_t>(
      mgr, n_dispatch_total, eval_energies);

  return {V_t, dV_dx_t, dispatch_indices_t, terms_for_dispatch_t};
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto MetalCoordinationRotamerScoreDispatch<DeviceDispatch, D, Real, Int>::
    backward(
        ContextManager& mgr,
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

        TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
        TView<Int, 2, D> conn_atom,
        TView<Int, 2, D> conn_metal,
        TView<Int, 2, D> conn_virt,
        TView<Int, 2, D> conn_key,
        TView<MetalSiteParams<Real>, 3, D> site_params,
        TView<Vec<Int, 2>, 2, D> fan_atoms,
        TView<MetalFanParams<Real>, 2, D> fan_params,
        TView<Int, 3, D> bridge_internal_metal,
        TView<Real, 3, D> bridge_internal_d0,
        TView<MetalBridgeParams<Real>, 1, D> bridge_params,
        TView<Int, 2, D> terms_for_dispatch,
        TView<Real, 2, D> dTdV) -> TPack<Vec<Real, 3>, 2, D> {
  int const n_atoms = rot_coords.size(0);
  int const n_dispatch_total = terms_for_dispatch.size(1);
  int const max_n_fan = fan_atoms.size(1);
  int const max_n_conns = conn_virt.size(1);
  int const max_n_internal = bridge_internal_metal.size(2);

  assert(dTdV.size(0) == 1);
  assert(dTdV.size(1) == n_dispatch_total);

  auto dV_dx_t = TPack<Vec<Real, 3>, 2, D>::zeros({1, n_atoms});
  auto dV_dx = dV_dx_t.view;

  LAUNCH_BOX_32;

  auto eval_derivs = ([=] TMOL_DEVICE_FUNC(int dispatch_ind) {
    Real const weight = dTdV[0][dispatch_ind];
    if (weight == 0) {
      return;
    }
    int const rot_ind1 = terms_for_dispatch[0][dispatch_ind];
    int const rot_ind2 = terms_for_dispatch[1][dispatch_ind];
    int const conn1 = terms_for_dispatch[2][dispatch_ind];
    int const block_type1 = block_type_ind_for_rot[rot_ind1];
    int const offset1 = rot_coord_offset[rot_ind1];

    if (conn1 <= -2) {
      int const bridge_pose = pose_ind_for_rot[rot_ind1];
      auto const bridge = bridge_from_code<Int, Real, D>(
          -2 - conn1,
          bridge_pose,
          block_ind_for_rot[rot_ind1],
          block_type1,
          block_type_ind_for_rot[rot_ind2],
          max_n_conns,
          max_n_internal,
          first_rot_block_type,
          pose_stack_inter_block_connections,
          conn_metal,
          conn_key,
          site_params,
          bridge_internal_metal,
          bridge_internal_d0);
      if (bridge.valid) {
        accumulate_metal_bridge<Real, D>(
            rot_coords,
            offset1 + bridge.metal1,
            rot_coord_offset[rot_ind2] + bridge.metal2,
            bridge.d1,
            bridge.d2,
            bridge_params[bridge.key],
            dV_dx,
            true,
            weight,
            nullptr);
      }
      return;
    }

    if (conn1 < 0) {
      for (int i = 0; i < max_n_fan; i++) {
        Vec<Int, 2> const pair = fan_atoms[block_type1][i];
        if (pair[0] < 0) {
          break;
        }
        accumulate_metal_fan_pair<Real, D>(
            rot_coords,
            offset1 + pair[0],
            offset1 + pair[1],
            fan_params[block_type1][i],
            dV_dx,
            true,
            weight,
            nullptr);
      }
      return;
    }

    int const pose_ind = pose_ind_for_rot[rot_ind1];
    int const block_ind1 = block_ind_for_rot[rot_ind1];
    int const conn2 =
        pose_stack_inter_block_connections[pose_ind][block_ind1][conn1][1];
    int const block_type2 = block_type_ind_for_rot[rot_ind2];
    int const key = conn_key[block_type2][conn2];
    if (key < 0) {
      return;
    }
    int const virt = conn_virt[block_type1][conn1];
    accumulate_metal_site<Real, D>(
        rot_coords,
        offset1 + conn_metal[block_type1][conn1],
        virt >= 0 ? offset1 + virt : -1,
        rot_coord_offset[rot_ind2] + conn_atom[block_type2][conn2],
        site_params[block_type1][conn1][key],
        dV_dx,
        true,
        weight,
        nullptr);
  });
  DeviceDispatch<D>::template forall<launch_t>(
      mgr, n_dispatch_total, eval_derivs);

  return dV_dx_t;
}

}  // namespace potentials
}  // namespace metal
}  // namespace score
}  // namespace tmol
