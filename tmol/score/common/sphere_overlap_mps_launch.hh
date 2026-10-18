#pragma once

// The calls the generic sphere-overlap launchers make into the Metal
// backend.
//
// sphere_overlap.impl.hh is compiled for every backend, so it cannot name
// Metal-only types. These entries compile to nothing unless D is MPS -- and
// to nothing at all in a build without the Metal backend.

#include <tmol/utility/portable/device.hh>

#include <cstdint>
#include <stdexcept>

#ifdef WITH_MPS
#include <tmol/score/common/mps_launch.hh>
#include <tmol/score/common/sphere_overlap_ctx.hh>
#include <tmol/score/common/sphere_overlap_mps_kernels_source.h>
#include <tmol/score/common/workgroup_lane.hh>
#endif

namespace tmol {
namespace score {
namespace common {
namespace sphere_overlap {

// One threadgroup per rotamer.
template <tmol::Device D, typename Ctx>
void block_spheres_mps_launch_if(Ctx& ctx, int64_t n_rots) {
  if constexpr (D == tmol::Device::MPS) {
#ifdef WITH_MPS
    auto args = mps_collect(ctx);
    int64_t const group = TMOL_MPS_THREADGROUP_SIZE;
    mps_launch(
        kSphereOverlapMetalShaderSource,
        "compute_block_spheres_f32",
        args,
        n_rots * group,
        group);
#else
    throw std::runtime_error(
        "tmol: an MPS tensor reached sphere_overlap, but this build has no "
        "Metal backend");
#endif
  } else {
    (void)ctx;
    (void)n_rots;
  }
}

// One thread per candidate block pair. The grid is rounded up to whole
// threadgroups; the kernel drops the tail.
template <tmol::Device D, typename Ctx>
void block_neighbors_mps_launch_if(Ctx& ctx, int64_t n_block_pairs) {
  if constexpr (D == tmol::Device::MPS) {
#ifdef WITH_MPS
    auto args = mps_collect(ctx);
    args.set_scalar(Ctx::n_views, Ctx::scalar_max_n_blocks, ctx.max_n_blocks);
    args.set_scalar(
        Ctx::n_views,
        Ctx::scalar_block_pairs_per_pose,
        ctx.block_pairs_per_pose);
    args.set_scalar(
        Ctx::n_views, Ctx::scalar_reach, tmol_pack_float(float(ctx.reach)));

    int64_t const group = TMOL_MPS_THREADGROUP_SIZE;
    int64_t const rounded = ((n_block_pairs + group - 1) / group) * group;
    mps_launch(
        kSphereOverlapMetalShaderSource,
        "detect_block_neighbors_f32",
        args,
        rounded,
        group);
#else
    throw std::runtime_error(
        "tmol: an MPS tensor reached sphere_overlap, but this build has no "
        "Metal backend");
#endif
  } else {
    (void)ctx;
    (void)n_block_pairs;
  }
}


// ---------------------------------------------------------------------------
// Rotamer-pair dispatch.

// One threadgroup per rotamer.
template <tmol::Device D, typename Ctx>
void rot_spheres_mps_launch_if(Ctx& ctx, int64_t n_rots) {
  if constexpr (D == tmol::Device::MPS) {
#ifdef WITH_MPS
    auto args = mps_collect(ctx);
    int64_t const group = TMOL_MPS_THREADGROUP_SIZE;
    mps_launch(
        kSphereOverlapMetalShaderSource,
        "compute_rot_spheres_f32",
        args,
        n_rots * group,
        group);
#else
    throw std::runtime_error(
        "tmol: an MPS tensor reached sphere_overlap, but this build has no "
        "Metal backend");
#endif
  } else {
    (void)ctx;
    (void)n_rots;
  }
}

// One thread per (pose, block).
template <tmol::Device D, typename Ctx>
void block_spheres_from_rot_mps_launch_if(Ctx& ctx, int64_t n_pose_blocks) {
  if constexpr (D == tmol::Device::MPS) {
#ifdef WITH_MPS
    auto args = mps_collect(ctx);
    args.set_scalar(Ctx::n_views, Ctx::scalar_max_n_blocks, ctx.max_n_blocks);
    int64_t const group = TMOL_MPS_THREADGROUP_SIZE;
    int64_t const rounded = ((n_pose_blocks + group - 1) / group) * group;
    mps_launch(
        kSphereOverlapMetalShaderSource,
        "compute_block_spheres_from_rot_f32",
        args,
        rounded,
        group);
#else
    throw std::runtime_error(
        "tmol: an MPS tensor reached sphere_overlap, but this build has no "
        "Metal backend");
#endif
  } else {
    (void)ctx;
    (void)n_pose_blocks;
  }
}

// One thread per candidate block pair. The count and fill passes bind the
// same context and differ only in the entry point they name.
template <tmol::Device D, typename Ctx>
void rot_pair_list_mps_launch_if(
    Ctx& ctx, int64_t n_candidates, char const* entry) {
  if constexpr (D == tmol::Device::MPS) {
#ifdef WITH_MPS
    auto args = mps_collect(ctx);
    args.set_scalar(Ctx::n_views, Ctx::scalar_max_n_blocks, ctx.max_n_blocks);
    args.set_scalar(
        Ctx::n_views,
        Ctx::scalar_block_pairs_per_pose,
        ctx.block_pairs_per_pose);
    args.set_scalar(
        Ctx::n_views, Ctx::scalar_candidate_begin, ctx.candidate_begin);
    args.set_scalar(
        Ctx::n_views, Ctx::scalar_reach, tmol_pack_float(float(ctx.reach)));

    int64_t const group = TMOL_MPS_THREADGROUP_SIZE;
    int64_t const rounded = ((n_candidates + group - 1) / group) * group;
    mps_launch(
        kSphereOverlapMetalShaderSource, entry, args, rounded, group);
#else
    throw std::runtime_error(
        "tmol: an MPS tensor reached sphere_overlap, but this build has no "
        "Metal backend");
#endif
  } else {
    (void)ctx;
    (void)n_candidates;
    (void)entry;
  }
}

}  // namespace sphere_overlap
}  // namespace common
}  // namespace score
}  // namespace tmol
