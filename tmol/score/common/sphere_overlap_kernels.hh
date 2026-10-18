#pragma once

// Bodies of the two sphere-overlap kernels the pose-scoring terms need.
//
// Separated from sphere_overlap.impl.hh -- which holds the host launchers and
// therefore reaches the torch tensor layer -- so that a Metal translation
// unit can include just the kernels. Everything named here is freestanding.
//
// The bodies are templated on their context type, so the CPU and CUDA builds
// pass TView-backed state while a `.metal` entry passes MetalView-backed
// state and runs the same code.

#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/sphere_overlap_ctx.hh>
#include <tmol/score/common/upper_triangle_indices.hh>
#include <tmol/score/common/workgroup_lane.hh>
#include <tmol/utility/portable/device.hh>
#include <tmol/utility/portable/math.hh>
#include <tmol/utility/portable/matrix.hh>
#include <tmol/utility/portable/metal_view.hh>
#include <tmol/utility/portable/reduce_ops.hh>

namespace tmol {
namespace score {
namespace common {
namespace sphere_overlap {

// Bounding sphere of one block: the centre of mass of its atoms and the
// distance from that centre to the furthest atom.
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    typename launch_t,
    typename Ctx>
TMOL_HD void compute_block_spheres_body(
    TMOL_THREAD Ctx const& ctx, int cta, WorkgroupLane<D> lane = {}) {
  CTA_LAUNCH_T_PARAMS;

  int const pose_ind = ctx.pose_ind_for_rot[cta];
  int const block_ind = ctx.block_ind_for_rot[cta];
  int const block_type = ctx.block_type_ind_for_rot[cta];
  int const coord_offset = ctx.rot_coord_offset[cta];

  if (block_type < 0) return;
  int const n_atoms = ctx.block_type_n_atoms[block_type];

  tmol::portable::Vec<Real, 3> local_coords{0, 0, 0};
  auto per_thread_com = ([&] TMOL_DEVICE_FUNC(int tid) {
    for (int i = tid; i < n_atoms; i += nt) {
      auto ci = ctx.rot_coords[coord_offset + i];
      for (int j = 0; j < 3; ++j) {
        local_coords[j] += ci[j];
      }
    }
    for (int j = 0; j < 3; ++j) {
      local_coords[j] /= n_atoms;
    }
  });
  DeviceDispatch<D>::template for_each_in_workgroup<nt>(per_thread_com, lane);

  DeviceDispatch<D>::synchronize_workgroup(lane);
  tmol::portable::Vec<Real, 3> com =
      DeviceDispatch<D>::template shuffle_reduce_and_broadcast_in_workgroup<nt>(
          local_coords, tmol::portable::plus_t<Real>(), lane);

  Real dmax(0);
  Real d2max(0);
  auto per_thread_dist_to_com = ([&] TMOL_DEVICE_FUNC(int tid) {
    for (int i = tid; i < n_atoms; i += nt) {
      auto ci = ctx.rot_coords[coord_offset + i];
      Real d2 =
          ((ci[0] - com[0]) * (ci[0] - com[0])
           + (ci[1] - com[1]) * (ci[1] - com[1])
           + (ci[2] - com[2]) * (ci[2] - com[2]));
      if (d2 > d2max) {
        d2max = d2;
      }
    }
    // The scalar workgroup implementation is faster taking the square root
    // before the reduction; the parallel backends reduce squared distances so
    // that one square root runs per block rather than one per thread.
    if (D == tmol::Device::CPU) {
      dmax = tmol::portable::math::sqrt(d2max);
    }
  });
  DeviceDispatch<D>::template for_each_in_workgroup<nt>(
      per_thread_dist_to_com, lane);

  if (D == tmol::Device::CPU) {
    dmax = DeviceDispatch<D>::template shuffle_reduce_in_workgroup<nt>(
        dmax, tmol::portable::maximum_t<Real>(), lane);
  } else {
    d2max = DeviceDispatch<D>::template shuffle_reduce_in_workgroup<nt>(
        d2max, tmol::portable::maximum_t<Real>(), lane);
    dmax = tmol::portable::math::sqrt(d2max);
  }

  auto thread0_write_out_result = ([&] TMOL_DEVICE_FUNC(int tid) {
    if (tid == 0) {
      ctx.block_spheres[pose_ind][block_ind][0] = com[0];
      ctx.block_spheres[pose_ind][block_ind][1] = com[1];
      ctx.block_spheres[pose_ind][block_ind][2] = com[2];
      ctx.block_spheres[pose_ind][block_ind][3] = dmax;
    }
  });
  DeviceDispatch<D>::template for_each_in_workgroup<nt>(
      thread0_write_out_result, lane);
}

// Whether two blocks are close enough that any of their atom pairs can
// contribute. One thread per candidate pair.
template <tmol::Device D, typename Real, typename Int, typename Ctx>
TMOL_HD void detect_block_neighbors_body(
    TMOL_THREAD Ctx const& ctx, int index) {
  int const pose_ind = index / ctx.block_pairs_per_pose;
  int const pair = index % ctx.block_pairs_per_pose;
  int const block_ind1 = pair / ctx.max_n_blocks;
  int const block_ind2 = pair % ctx.max_n_blocks;

  // The CPU backend keeps zero-initialized scratch and writes only the cells
  // that are neighbors; the parallel backends allocate uninitialized scratch
  // and write every cell, which saves a separate fill.
  bool const writes_every_cell = D != tmol::Device::CPU;

  if (block_ind1 > block_ind2) {
    if (writes_every_cell) {
      ctx.block_neighbors[pose_ind][block_ind1][block_ind2] = 0;
    }
    return;
  }

  int const block_type1 = ctx.pose_stack_block_type[pose_ind][block_ind1];
  int const block_type2 =
      block_type1 < 0 ? -1 : ctx.pose_stack_block_type[pose_ind][block_ind2];
  if (block_type1 < 0 || block_type2 < 0) {
    if (writes_every_cell) {
      ctx.block_neighbors[pose_ind][block_ind1][block_ind2] = 0;
    }
    return;
  }

  Real sphere1[4];
  Real sphere2[4];
  for (int i = 0; i < 4; ++i) {
    sphere1[i] = ctx.block_spheres[pose_ind][block_ind1][i];
    sphere2[i] = ctx.block_spheres[pose_ind][block_ind2][i];
  }

  Real const d2 =
      ((sphere1[0] - sphere2[0]) * (sphere1[0] - sphere2[0])
       + (sphere1[1] - sphere2[1]) * (sphere1[1] - sphere2[1])
       + (sphere1[2] - sphere2[2]) * (sphere1[2] - sphere2[2]));
  Real const d_threshold = sphere1[3] + sphere2[3] + ctx.reach;

  if (d2 < d_threshold * d_threshold) {
    ctx.block_neighbors[pose_ind][block_ind1][block_ind2] = 1;
  } else if (writes_every_cell) {
    ctx.block_neighbors[pose_ind][block_ind1][block_ind2] = 0;
  }
}


// ---------------------------------------------------------------------------
// Rotamer-pair dispatch.
//
// Unlike the two kernels above -- whose host launchers still carry their own
// copy of the body as a device lambda -- these four are called from both
// paths: the generic launcher builds a TView-backed context and invokes the
// same function the Metal entry point does. One body, three backends.

// Largest value the dispatch index type can hold, without <limits>, which
// Metal has no standard library for.
template <typename Int>
TMOL_HD Int sphere_overlap_int_max() {
  return Int(~(Int(1) << (sizeof(Int) * 8 - 1)));
}

// Bounding sphere of one rotamer: the centre of mass of its atoms and the
// distance from that centre to the furthest atom. One workgroup per rotamer.
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    typename launch_t,
    typename Ctx>
TMOL_HD void compute_rot_spheres_body(
    TMOL_THREAD Ctx const& ctx, int cta, WorkgroupLane<D> lane = {}) {
  CTA_LAUNCH_T_PARAMS;

  int const rot_ind = cta;
  int const coord_offset = ctx.rot_coord_offset[rot_ind];
  int const block_type = ctx.block_type_ind_for_rot[rot_ind];

  if (block_type < 0) return;
  int const n_atoms = ctx.block_type_n_atoms[block_type];

  tmol::portable::Vec<Real, 3> local_coords{0, 0, 0};
  auto per_thread_com = ([&] TMOL_DEVICE_FUNC(int tid) {
    for (int i = tid; i < n_atoms; i += nt) {
      auto ci = ctx.rot_coords[coord_offset + i];
      for (int j = 0; j < 3; ++j) {
        local_coords[j] += ci[j];
      }
    }
    for (int j = 0; j < 3; ++j) {
      local_coords[j] /= n_atoms;
    }
  });
  DeviceDispatch<D>::template for_each_in_workgroup<nt>(per_thread_com, lane);

  DeviceDispatch<D>::synchronize_workgroup(lane);
  tmol::portable::Vec<Real, 3> com =
      DeviceDispatch<D>::template shuffle_reduce_and_broadcast_in_workgroup<nt>(
          local_coords, tmol::portable::plus_t<Real>(), lane);

  Real dmax(0);
  Real d2max(0);
  auto per_thread_dist_to_com = ([&] TMOL_DEVICE_FUNC(int tid) {
    for (int i = tid; i < n_atoms; i += nt) {
      auto ci = ctx.rot_coords[coord_offset + i];
      Real d2 =
          ((ci[0] - com[0]) * (ci[0] - com[0])
           + (ci[1] - com[1]) * (ci[1] - com[1])
           + (ci[2] - com[2]) * (ci[2] - com[2]));
      if (d2 > d2max) {
        d2max = d2;
      }
    }
    // See compute_block_spheres_body: the scalar backend is faster taking the
    // square root before the reduction, the parallel ones after.
    if (D == tmol::Device::CPU) {
      dmax = tmol::portable::math::sqrt(d2max);
    }
  });
  DeviceDispatch<D>::template for_each_in_workgroup<nt>(
      per_thread_dist_to_com, lane);

  if (D == tmol::Device::CPU) {
    dmax = DeviceDispatch<D>::template shuffle_reduce_in_workgroup<nt>(
        dmax, tmol::portable::maximum_t<Real>(), lane);
  } else {
    d2max = DeviceDispatch<D>::template shuffle_reduce_in_workgroup<nt>(
        d2max, tmol::portable::maximum_t<Real>(), lane);
    dmax = tmol::portable::math::sqrt(d2max);
  }

  auto thread0_write_out_result = ([&] TMOL_DEVICE_FUNC(int tid) {
    if (tid == 0) {
      ctx.rot_spheres[rot_ind][0] = com[0];
      ctx.rot_spheres[rot_ind][1] = com[1];
      ctx.rot_spheres[rot_ind][2] = com[2];
      ctx.rot_spheres[rot_ind][3] = dmax;
    }
  });
  DeviceDispatch<D>::template for_each_in_workgroup<nt>(
      thread0_write_out_result, lane);
}

// A block's bounding sphere from the spheres of the rotamers that can occupy
// it. One thread per (pose, block); the serial loop over rotamers is what
// keeps this free of the write races compute_block_spheres would have when
// launched per rotamer.
template <tmol::Device D, typename Real, typename Int, typename Ctx>
TMOL_HD void compute_block_spheres_from_rot_body(
    TMOL_THREAD Ctx const& ctx, int index) {
  int const pose = index / ctx.max_n_blocks;
  int const block = index % ctx.max_n_blocks;
  int const n_rots = ctx.n_rots_for_block[pose][block];
  if (n_rots <= 0) return;
  int const rot_start = ctx.rot_offset_for_block[pose][block];
  if (rot_start < 0) return;

  Real cx = 0, cy = 0, cz = 0;
  for (int r = 0; r < n_rots; ++r) {
    cx += ctx.rot_spheres[rot_start + r][0];
    cy += ctx.rot_spheres[rot_start + r][1];
    cz += ctx.rot_spheres[rot_start + r][2];
  }
  cx /= n_rots;
  cy /= n_rots;
  cz /= n_rots;

  Real rmax = 0;
  for (int r = 0; r < n_rots; ++r) {
    Real dx = ctx.rot_spheres[rot_start + r][0] - cx;
    Real dy = ctx.rot_spheres[rot_start + r][1] - cy;
    Real dz = ctx.rot_spheres[rot_start + r][2] - cz;
    Real d = tmol::portable::math::sqrt(dx * dx + dy * dy + dz * dz)
             + ctx.rot_spheres[rot_start + r][3];
    if (d > rmax) rmax = d;
  }

  ctx.block_spheres[pose][block][0] = cx;
  ctx.block_spheres[pose][block][1] = cy;
  ctx.block_spheres[pose][block][2] = cz;
  ctx.block_spheres[pose][block][3] = rmax;
}

// Whether two rotamers' bounding spheres reach each other.
template <typename RotSpheresView, typename Real>
TMOL_HD bool rot_spheres_overlap_view(
    TMOL_THREAD RotSpheresView const& rot_spheres,
    int rot1,
    int rot2,
    Real reach) {
  Real const dx = rot_spheres[rot1][0] - rot_spheres[rot2][0];
  Real const dy = rot_spheres[rot1][1] - rot_spheres[rot2][1];
  Real const dz = rot_spheres[rot1][2] - rot_spheres[rot2][2];
  Real const threshold = rot_spheres[rot1][3] + rot_spheres[rot2][3] + reach;
  return dx * dx + dy * dy + dz * dz < threshold * threshold;
}

// Shared prologue of the count and fill passes: resolve a linear candidate
// index into the pose and the two blocks of a packed upper-triangle pair, and
// read the per-block rotamer ranges. Returns false when the candidate is dead.
template <tmol::Device D, typename Real, typename Int, typename Ctx>
TMOL_HD bool rot_pair_candidate_blocks(
    TMOL_THREAD Ctx const& ctx,
    int candidate,
    TMOL_THREAD int& pose,
    TMOL_THREAD int& b1,
    TMOL_THREAD int& b2) {
  int const global_candidate = ctx.candidate_begin + candidate;
  pose = global_candidate / ctx.block_pairs_per_pose;
  auto pair = common::upper_triangle_inds_from_linear_index(
      global_candidate % ctx.block_pairs_per_pose, ctx.max_n_blocks + 1);
  b1 = common::get<0>(pair);
  b2 = common::get<1>(pair) - 1;
  return ctx.pose_stack_block_type[pose][b1] >= 0
         && ctx.pose_stack_block_type[pose][b2] >= 0;
}

// Pass 1: how many rotamer pairs each candidate block pair contributes.
// Diagonal pairs contribute self-pairs only; blocks in one lockstep group
// contribute matching rotamer indices; otherwise the overlapping spheres.
template <tmol::Device D, typename Real, typename Int, typename Ctx>
TMOL_HD void rot_pair_counts_body(TMOL_THREAD Ctx const& ctx, int candidate) {
  int pose, b1, b2;
  if (!rot_pair_candidate_blocks<D, Real, Int>(ctx, candidate, pose, b1, b2)) {
    return;
  }

  Real const dx =
      ctx.block_spheres[pose][b1][0] - ctx.block_spheres[pose][b2][0];
  Real const dy =
      ctx.block_spheres[pose][b1][1] - ctx.block_spheres[pose][b2][1];
  Real const dz =
      ctx.block_spheres[pose][b1][2] - ctx.block_spheres[pose][b2][2];
  Real const block_threshold = ctx.block_spheres[pose][b1][3]
                               + ctx.block_spheres[pose][b2][3] + ctx.reach;
  if (!(dx * dx + dy * dy + dz * dz < block_threshold * block_threshold)) {
    return;
  }

  int const nr1 = ctx.n_rots_for_block[pose][b1];
  int const nr2 = ctx.n_rots_for_block[pose][b2];
  int const off1 = ctx.rot_offset_for_block[pose][b1];
  int const off2 = ctx.rot_offset_for_block[pose][b2];
  if (nr1 <= 0 || nr2 <= 0 || off1 < 0 || off2 < 0) return;
  int const g1 = ctx.lockstep_group_for_block[pose][b1];
  int const g2 = ctx.lockstep_group_for_block[pose][b2];

  int64_t count = 0;
  if (b1 == b2 || (g1 >= 0 && g1 == g2)) {
    count = nr1 < nr2 ? nr1 : nr2;
  } else {
    for (int i = 0; i < nr1; ++i) {
      for (int j = 0; j < nr2; ++j) {
        count += rot_spheres_overlap_view(
            ctx.rot_spheres, off1 + i, off2 + j, ctx.reach);
      }
    }
  }
  if (count > int64_t(sphere_overlap_int_max<Int>())) {
    // Every writer stores the same value, so a plain store is enough; the
    // host reads the flag after the dispatch completes.
    ctx.pair_count_overflow[0] = 1;
    return;
  }
  ctx.pair_counts[candidate] = Int(count);
}

// Pass 2: write the (pose, rot1, rot2) triples, starting where the exclusive
// scan of the counts says this candidate's run begins.
template <tmol::Device D, typename Real, typename Int, typename Ctx>
TMOL_HD void rot_pair_fill_body(TMOL_THREAD Ctx const& ctx, int candidate) {
  if (ctx.pair_counts[candidate] == 0) return;
  int pose, b1, b2;
  // The count pass already rejected dead candidates by leaving the count at
  // zero, so the return value is not re-tested here.
  rot_pair_candidate_blocks<D, Real, Int>(ctx, candidate, pose, b1, b2);

  int const nr1 = ctx.n_rots_for_block[pose][b1];
  int const nr2 = ctx.n_rots_for_block[pose][b2];
  int const off1 = ctx.rot_offset_for_block[pose][b1];
  int const off2 = ctx.rot_offset_for_block[pose][b2];
  if (off1 < 0 || off2 < 0) return;

  int const g1 = ctx.lockstep_group_for_block[pose][b1];
  int const g2 = ctx.lockstep_group_for_block[pose][b2];

  Int offset = ctx.pair_offsets[candidate];
  if (b1 == b2) {
    for (int i = 0; i < nr1; ++i) {
      ctx.indices[0][offset] = pose;
      ctx.indices[1][offset] = off1 + i;
      ctx.indices[2][offset] = off1 + i;  // same rot
      ++offset;
    }
  } else if (g1 >= 0 && g1 == g2) {
    // lockstep: any pair of differing rotamer indices describes a conformer
    // combination the group cannot adopt
    int const n = nr1 < nr2 ? nr1 : nr2;
    for (int i = 0; i < n; ++i) {
      ctx.indices[0][offset] = pose;
      ctx.indices[1][offset] = off1 + i;
      ctx.indices[2][offset] = off2 + i;
      ++offset;
    }
  } else {
    for (int i = 0; i < nr1; ++i) {
      for (int j = 0; j < nr2; ++j) {
        if (!rot_spheres_overlap_view(
                ctx.rot_spheres, off1 + i, off2 + j, ctx.reach)) {
          continue;
        }
        ctx.indices[0][offset] = pose;
        ctx.indices[1][offset] = off1 + i;
        ctx.indices[2][offset] = off2 + j;
        ++offset;
      }
    }
  }
}

}  // namespace sphere_overlap
}  // namespace common
}  // namespace score
}  // namespace tmol
