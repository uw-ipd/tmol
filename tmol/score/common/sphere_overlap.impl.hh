#include <tmol/utility/portable/reduce_ops.hh>
#include <tmol/utility/portable/matrix.hh>
#include <tmol/utility/tensor/native.hh>
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/counting.hh>

#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/sphere_overlap_ctx.hh>
#include <tmol/score/common/sphere_overlap_kernels.hh>
#include <tmol/score/common/tview_binder.hh>
#include <tmol/score/common/sphere_overlap_mps_launch.hh>
#include <tmol/score/common/upper_triangle_indices.hh>

#include <tmol/utility/tensor/context_manager.hh>

namespace tmol {
namespace score {
namespace common {
namespace sphere_overlap {

using tmol::portable::Vec;

inline bool should_compact_block_neighbors(
    int n_poses, int max_n_blocks, bool computes_derivatives) {
  // Compaction pays for one large inference pose or a wide batch of smaller
  // poses. Derivative kernels carry more useful work per CTA, so require both
  // a large pose and a genuinely wide workload.
  constexpr int min_blocks_per_pose = 256;
  constexpr int min_inference_candidates = 1 << 16;
  constexpr int min_batched_blocks_for_derivatives = 1000;
  int64_t const pairs_per_pose =
      (int64_t(max_n_blocks) * (int64_t(max_n_blocks) + 1)) / 2;
  bool const enough_inference_candidates =
      n_poses > 0
      && pairs_per_pose
             >= (min_inference_candidates + int64_t(n_poses) - 1) / n_poses;
  bool const large_inference_workload =
      max_n_blocks >= min_blocks_per_pose || enough_inference_candidates;
  if (!large_inference_workload) return false;
  return !computes_derivatives
         || (max_n_blocks >= min_blocks_per_pose
             && int64_t(n_poses) * max_n_blocks
                    >= min_batched_blocks_for_derivatives);
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct compute_rot_spheres {
  // Templated on the coordinate view so terms can move from Eigen to
  // tmol::portable::Vec independently; both spell a 3-vector of Real.
  template <typename CoordTView>
  static void f(
      ContextManager& mgr,
      CoordTView rot_coords,
      TView<Int, 1, D> rot_coord_offset,
      TView<Int, 1, D> block_type_ind_for_rot,
      TView<Int, 1, D> block_type_n_atoms,
      TView<Real, 2, D> rot_spheres) {
    LAUNCH_BOX_32;

    // One context, three backends: the generic launch captures it in a device
    // lambda, the Metal launch binds its views as buffers, and both reach the
    // same body in sphere_overlap_kernels.hh.
    RotSpheresCtxT<
        tmol::score::common::TViewBinder<D>::template View,
        Real,
        Int>
        ctx;
    ctx.rot_coords = rot_coords;
    ctx.rot_coord_offset = rot_coord_offset;
    ctx.block_type_ind_for_rot = block_type_ind_for_rot;
    ctx.block_type_n_atoms = block_type_n_atoms;
    ctx.rot_spheres = rot_spheres;

    if (D == tmol::Device::MPS) {
      rot_spheres_mps_launch_if<D>(ctx, rot_coord_offset.size(0));
    } else {
      auto compute_spheres = ([=] TMOL_DEVICE_FUNC(int cta) {
        compute_rot_spheres_body<DeviceDispatch, D, Real, Int, launch_t>(
            ctx, cta);
      });
      DeviceDispatch<D>::template foreach_independent_workgroup<launch_t>(
          mgr, rot_coord_offset.size(0), compute_spheres);
    }
  }
};

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    bool ResetCount = false>
struct compute_block_spheres {
  // Templated on the coordinate view so terms can move from Eigen to
  // tmol::portable::Vec independently; both spell a 3-vector of Real.
  template <typename CoordTView>
  static void f(
      ContextManager& mgr,
      CoordTView rot_coords,
      TView<Int, 1, D> rot_coord_offset,
      TView<Int, 1, D> block_ind_for_rot,
      TView<Int, 1, D> pose_ind_for_rot,
      TView<Int, 1, D> block_type_ind_for_rot,
      TView<Int, 1, D> block_type_n_atoms,
      TView<Real, 3, D> block_spheres,
      Int* count_to_reset = nullptr) {
    LAUNCH_BOX_32;

    auto compute_spheres = ([=] TMOL_DEVICE_FUNC(int cta) {
      CTA_LAUNCH_T_PARAMS;

      // Define this outside the constexpr branch so NVCC captures the pointer
      // in an ordinary extended-lambda context.
      auto reset_count = ([=] TMOL_DEVICE_FUNC(int tid) {
        if (cta == 0 && tid == 0) count_to_reset[0] = 0;
      });
      if constexpr (ResetCount) {
        DeviceDispatch<D>::template for_each_in_workgroup<nt>(reset_count);
      }

      int const pose_ind = pose_ind_for_rot[cta];
      int const block_ind = block_ind_for_rot[cta];
      int const block_type = block_type_ind_for_rot[cta];
      int const coord_offset = rot_coord_offset[cta];

      if (block_type < 0) return;
      int const n_atoms = block_type_n_atoms[block_type];
      Vec<Real, 3> local_coords{0, 0, 0};

      auto per_thread_com = ([&] TMOL_DEVICE_FUNC(int tid) {
        for (int i = tid; i < n_atoms; i += nt) {
          auto ci = rot_coords[coord_offset + i];
          for (int j = 0; j < 3; ++j) {
            local_coords[j] += ci[j];
          }
        }
        for (int j = 0; j < 3; ++j) {
          local_coords[j] /= n_atoms;
        }
      });

      DeviceDispatch<D>::template for_each_in_workgroup<nt>(per_thread_com);

      // CPU's scalar workgroup implementation is faster with the square root
      // before reduction; CUDA reduces squared distances to execute one sqrt.
      Real dmax(0);

      DeviceDispatch<D>::synchronize_workgroup();
      Vec<Real, 3> com =
          DeviceDispatch<D>::template shuffle_reduce_and_broadcast_in_workgroup<
              nt>(local_coords, tmol::portable::plus_t<Real>());

      Real d2max = 0;
      // Now find maximum distance
      auto per_thread_dist_to_com = ([&] TMOL_DEVICE_FUNC(int tid) {
        for (int i = tid; i < n_atoms; i += nt) {
          auto ci = rot_coords[coord_offset + i];
          Real d2 =
              ((ci[0] - com[0]) * (ci[0] - com[0])
               + (ci[1] - com[1]) * (ci[1] - com[1])
               + (ci[2] - com[2]) * (ci[2] - com[2]));
          if (d2 > d2max) {
            d2max = d2;
          }
        }
        if constexpr (D == Device::CPU) {
          dmax = sqrt(d2max);
        }
      });
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(
          per_thread_dist_to_com);

      if constexpr (D == Device::CPU) {
        dmax = DeviceDispatch<D>::template shuffle_reduce_in_workgroup<nt>(
            dmax, tmol::portable::maximum_t<Real>());
      } else {
        d2max = DeviceDispatch<D>::template shuffle_reduce_in_workgroup<nt>(
            d2max, tmol::portable::maximum_t<Real>());
      }

      auto thread0_write_out_result = ([=] TMOL_DEVICE_FUNC(int tid) {
        if (tid == 0) {
          block_spheres[pose_ind][block_ind][0] = com[0];
          block_spheres[pose_ind][block_ind][1] = com[1];
          block_spheres[pose_ind][block_ind][2] = com[2];
          if constexpr (D == Device::CPU) {
            block_spheres[pose_ind][block_ind][3] = dmax;
          } else {
            // Reduce squared distances first and take one square root per
            // block, rather than one per participating GPU thread.
            block_spheres[pose_ind][block_ind][3] = sqrt(d2max);
          }
        }
      });
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(
          thread0_write_out_result);
    });

    if (D == tmol::Device::MPS) {
      // Metal cannot launch a device lambda, so the same body is reached
      // through a named entry point instead of through compute_spheres.
      BlockSpheresCtxT<
          tmol::score::common::TViewBinder<D>::template View,
          Real,
          Int>
          ctx;
      ctx.rot_coords = rot_coords;
      ctx.rot_coord_offset = rot_coord_offset;
      ctx.block_ind_for_rot = block_ind_for_rot;
      ctx.pose_ind_for_rot = pose_ind_for_rot;
      ctx.block_type_ind_for_rot = block_type_ind_for_rot;
      ctx.block_type_n_atoms = block_type_n_atoms;
      ctx.block_spheres = block_spheres;
      block_spheres_mps_launch_if<D>(ctx, block_ind_for_rot.size(0));
    } else {
      DeviceDispatch<D>::template foreach_workgroup<launch_t>(
          mgr, block_ind_for_rot.size(0), compute_spheres);
    }
  }
};

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct detect_rot_neighbors {
  static void f(
      ContextManager& mgr,
      Int max_n_rots_per_pose,
      TView<Int, 1, D> block_ind_for_rot,
      TView<Int, 1, D> block_type_ind_for_rot,
      TView<Int, 1, D> block_type_n_atoms,
      TView<Int, 1, D> n_rots_for_pose,
      TView<Int, 1, D> rot_offset_for_pose,
      TView<Int, 2, D> n_rots_for_block,
      TView<Real, 2, D> rot_spheres,
      TView<Int, 3, D> rot_neighbors,
      Real reach) {
    LAUNCH_BOX_32;

    int const n_poses = n_rots_for_block.size(0);
    int const max_n_rots = max_n_rots_per_pose;
    int const rot_pairs_per_pose = common::checked_dispatch_product(
        max_n_rots, max_n_rots, "rotamer-neighbor candidates per pose");
    int const n_rot_pairs = common::checked_dispatch_product(
        n_poses, rot_pairs_per_pose, "rotamer-neighbor candidates");

    auto detect_neighbors = ([=] TMOL_DEVICE_FUNC(int ind) {
      int const n_block_types = block_type_n_atoms.size(0);

      int const pose_ind = ind / rot_pairs_per_pose;
      int const rot_pair_ind = ind % rot_pairs_per_pose;
      int const rot_ind1 = rot_pair_ind / max_n_rots;
      int const rot_ind2 = rot_pair_ind % max_n_rots;

      if (rot_ind1 >= n_rots_for_pose[pose_ind]
          || rot_ind2 >= n_rots_for_pose[pose_ind]) {
        if (D == Device::CUDA) {
          rot_neighbors[pose_ind][rot_ind1][rot_ind2] = 0;
        }
        return;
      }

      if (rot_ind1 > rot_ind2) {
        if (D == Device::CUDA) {
          rot_neighbors[pose_ind][rot_ind1][rot_ind2] = 0;
        }
        return;
      }

      int const global_rot_ind1 = rot_ind1 + rot_offset_for_pose[pose_ind];
      int const global_rot_ind2 = rot_ind2 + rot_offset_for_pose[pose_ind];

      int const block_ind1 = block_ind_for_rot[global_rot_ind1];
      int const block_ind2 = block_ind_for_rot[global_rot_ind2];

      bool same_rot = rot_ind1 == rot_ind2;
      bool same_block = block_ind1 == block_ind2;
      if (same_block && !same_rot) {
        if (D == Device::CUDA) {
          rot_neighbors[pose_ind][rot_ind1][rot_ind2] = 0;
        }
        return;
      }

      int const block_type1 = block_type_ind_for_rot[global_rot_ind1];
      if (block_type1 < 0) {
        if (D == Device::CUDA) {
          rot_neighbors[pose_ind][rot_ind1][rot_ind2] = 0;
        }
        return;
      }
      int const block_type2 = block_type_ind_for_rot[global_rot_ind2];
      if (block_type2 < 0) {
        if (D == Device::CUDA) {
          rot_neighbors[pose_ind][rot_ind1][rot_ind2] = 0;
        }
        return;
      }

      Vec<Real, 4> sphere1{0, 0, 0, 0};
      Vec<Real, 4> sphere2{0, 0, 0, 0};

      for (int i = 0; i < 4; ++i) {
        sphere1[i] = rot_spheres[global_rot_ind1][i];
        sphere2[i] = rot_spheres[global_rot_ind2][i];
      }

      Real d2 =
          ((sphere1[0] - sphere2[0]) * (sphere1[0] - sphere2[0])
           + (sphere1[1] - sphere2[1]) * (sphere1[1] - sphere2[1])
           + (sphere1[2] - sphere2[2]) * (sphere1[2] - sphere2[2]));

      Real d_threshold = sphere1[3] + sphere2[3] + reach;

      // CUDA writes every cell so its caller can avoid a separate fill. CPU
      // retains zero-initialized scratch because skipping non-neighbor stores
      // is faster there.
      if (d2 < d_threshold * d_threshold) {
        rot_neighbors[pose_ind][rot_ind1][rot_ind2] = 1;
      } else if (D == Device::CUDA) {
        rot_neighbors[pose_ind][rot_ind1][rot_ind2] = 0;
      }
    });
    DeviceDispatch<D>::template forall_independent<launch_t>(
        mgr, n_rot_pairs, detect_neighbors);
  }
};

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct detect_block_neighbors {
  static void f(
      ContextManager& mgr,
      TView<Int, 2, D> pose_stack_block_type,
      TView<Real, 3, D> block_spheres,
      TView<Int, 3, D> block_neighbors,
      Real reach) {
    LAUNCH_BOX_32;

    int const n_poses = pose_stack_block_type.size(0);
    int const max_n_blocks = pose_stack_block_type.size(1);
    int const block_pairs_per_pose = common::checked_dispatch_product(
        max_n_blocks, max_n_blocks, "block-neighbor candidates per pose");
    int const n_block_pairs = common::checked_dispatch_product(
        n_poses, block_pairs_per_pose, "block-neighbor candidates");
    auto detect_neighbors = ([=] TMOL_DEVICE_FUNC(int index) {
      int const pose_ind = index / block_pairs_per_pose;
      int const pair = index % block_pairs_per_pose;
      int const block_ind1 = pair / max_n_blocks;
      int const block_ind2 = pair % max_n_blocks;

      if (block_ind1 > block_ind2) {
        if (D == Device::CUDA) {
          block_neighbors[pose_ind][block_ind1][block_ind2] = 0;
        }
        return;
      }

      int const block_type1 = pose_stack_block_type[pose_ind][block_ind1];
      if (block_type1 < 0) {
        if (D == Device::CUDA) {
          block_neighbors[pose_ind][block_ind1][block_ind2] = 0;
        }
        return;
      }
      int const block_type2 = pose_stack_block_type[pose_ind][block_ind2];
      if (block_type2 < 0) {
        if (D == Device::CUDA) {
          block_neighbors[pose_ind][block_ind1][block_ind2] = 0;
        }
        return;
      }

      Vec<Real, 4> sphere1{0, 0, 0, 0};
      Vec<Real, 4> sphere2{0, 0, 0, 0};

      for (int i = 0; i < 4; ++i) {
        sphere1[i] = block_spheres[pose_ind][block_ind1][i];
        sphere2[i] = block_spheres[pose_ind][block_ind2][i];
      }

      Real d2 =
          ((sphere1[0] - sphere2[0]) * (sphere1[0] - sphere2[0])
           + (sphere1[1] - sphere2[1]) * (sphere1[1] - sphere2[1])
           + (sphere1[2] - sphere2[2]) * (sphere1[2] - sphere2[2]));

      Real d_threshold = sphere1[3] + sphere2[3] + reach;

      // CUDA writes every cell so its caller can avoid a separate fill. CPU
      // retains zero-initialized scratch because skipping non-neighbor stores
      // is faster there.
      if (d2 < d_threshold * d_threshold) {
        block_neighbors[pose_ind][block_ind1][block_ind2] = 1;
      } else if (D == Device::CUDA) {
        block_neighbors[pose_ind][block_ind1][block_ind2] = 0;
      }
    });
    if (D == tmol::Device::MPS) {
      BlockNeighborsCtxT<
          tmol::score::common::TViewBinder<D>::template View,
          Real,
          Int>
          ctx;
      ctx.pose_stack_block_type = pose_stack_block_type;
      ctx.block_spheres = block_spheres;
      ctx.block_neighbors = block_neighbors;
      ctx.max_n_blocks = max_n_blocks;
      ctx.block_pairs_per_pose = block_pairs_per_pose;
      ctx.reach = reach;
      block_neighbors_mps_launch_if<D>(ctx, n_block_pairs);
    } else {
      DeviceDispatch<D>::template forall_independent<launch_t>(
          mgr, n_block_pairs, detect_neighbors);
    }
  }
};

template <typename Real, typename Int>
bool try_cpu_spatial_compact_block_neighbors(
    TView<Int, 2, tmol::Device::CPU> pose_stack_block_type,
    TView<Real, 3, tmol::Device::CPU> block_spheres,
    Real reach,
    std::vector<Int>& retained) {
  retained.clear();
  int const n_poses = pose_stack_block_type.size(0);
  int const max_n_blocks = pose_stack_block_type.size(1);
  constexpr int min_spatial_sweep_blocks = 150;
  if (n_poses != 1 || max_n_blocks < min_spatial_sweep_blocks) {
    return false;
  }

  std::vector<Int> ordered_blocks;
  ordered_blocks.reserve(max_n_blocks);
  Real min_center[3] = {
      std::numeric_limits<Real>::infinity(),
      std::numeric_limits<Real>::infinity(),
      std::numeric_limits<Real>::infinity()};
  Real max_center[3] = {
      -std::numeric_limits<Real>::infinity(),
      -std::numeric_limits<Real>::infinity(),
      -std::numeric_limits<Real>::infinity()};
  bool finite_spheres = true;
  for (int block = 0; block < max_n_blocks; ++block) {
    if (pose_stack_block_type[0][block] < 0) continue;
    ordered_blocks.push_back(block);
    for (int axis = 0; axis < 3; ++axis) {
      Real const center = block_spheres[0][block][axis];
      finite_spheres = finite_spheres && std::isfinite(center);
      min_center[axis] = std::min(min_center[axis], center);
      max_center[axis] = std::max(max_center[axis], center);
    }
    finite_spheres =
        finite_spheres && std::isfinite(block_spheres[0][block][3]);
  }

  // NaN/Inf coordinates follow the established quadratic comparison behavior
  // rather than entering a comparator without a strict order.
  if (!finite_spheres || ordered_blocks.size() < min_spatial_sweep_blocks) {
    return false;
  }

  int sweep_axis = 0;
  for (int axis = 1; axis < 3; ++axis) {
    if (max_center[axis] - min_center[axis]
        > max_center[sweep_axis] - min_center[sweep_axis]) {
      sweep_axis = axis;
    }
  }
  std::sort(
      ordered_blocks.begin(), ordered_blocks.end(), [=](Int left, Int right) {
        Real const left_center = block_spheres[0][left][sweep_axis];
        Real const right_center = block_spheres[0][right][sweep_axis];
        return left_center < right_center
               || (left_center == right_center && left < right);
      });

  int const n_valid_blocks = static_cast<int>(ordered_blocks.size());
  std::vector<Real> suffix_max_radius(n_valid_blocks);
  Real suffix_radius = 0;
  for (int position = n_valid_blocks - 1; position >= 0; --position) {
    suffix_radius =
        std::max(suffix_radius, block_spheres[0][ordered_blocks[position]][3]);
    suffix_max_radius[position] = suffix_radius;
  }

  int64_t const n_pairs =
      (int64_t(max_n_blocks) * (int64_t(max_n_blocks) + 1)) / 2;
  retained.reserve(
      std::min<int64_t>(
          n_pairs, std::max<int64_t>(16, int64_t(n_valid_blocks) * 32)));
  int const triangle_dimension = max_n_blocks + 1;
  for (int left_position = 0; left_position < n_valid_blocks; ++left_position) {
    Int const left = ordered_blocks[left_position];
    Real const left_axis_center = block_spheres[0][left][sweep_axis];
    Real const left_radius = block_spheres[0][left][3];
    for (int right_position = left_position; right_position < n_valid_blocks;
         ++right_position) {
      Int const right = ordered_blocks[right_position];
      Real const axis_separation =
          block_spheres[0][right][sweep_axis] - left_axis_center;
      if (right_position != left_position
          && axis_separation
                 >= left_radius + suffix_max_radius[right_position] + reach) {
        break;
      }

      Real const dx = block_spheres[0][left][0] - block_spheres[0][right][0];
      Real const dy = block_spheres[0][left][1] - block_spheres[0][right][1];
      Real const dz = block_spheres[0][left][2] - block_spheres[0][right][2];
      Real const d2 = dx * dx + dy * dy + dz * dz;
      Real const threshold = left_radius + block_spheres[0][right][3] + reach;
      if (d2 >= threshold * threshold) continue;

      Int const block1 = std::min(left, right);
      Int const block2 = std::max(left, right);
      int64_t const candidate =
          int64_t(block1) * (2 * int64_t(triangle_dimension) - block1 - 1) / 2
          + block2 - block1;
      retained.push_back(static_cast<Int>(candidate));
    }
  }
  std::sort(retained.begin(), retained.end());
  return true;
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct detect_compact_block_neighbors {
  static void f(
      ContextManager& mgr,
      TView<Int, 2, D> pose_stack_block_type,
      TView<Real, 3, D> block_spheres,
      TView<Int, 1, D> neighbor_indices,
      Real reach) {
    LAUNCH_BOX_32;

    int const n_poses = pose_stack_block_type.size(0);
    int const max_n_blocks = pose_stack_block_type.size(1);
    int const n_pairs = common::checked_dispatch_size(
        common::checked_count_product(
            max_n_blocks,
            int64_t(max_n_blocks) + 1,
            "compact block-neighbor candidates per pose")
            / 2,
        "compact block-neighbor candidates per pose");
    int const n_candidates = common::checked_dispatch_product(
        n_poses, n_pairs, "compact block-neighbor candidates");
    if constexpr (D == tmol::Device::CPU) {
      // Sorting retained candidate IDs restores the exact canonical order used
      // by the quadratic path, so downstream accumulation is unchanged.
      std::vector<Int> retained;
      if (try_cpu_spatial_compact_block_neighbors(
              pose_stack_block_type, block_spheres, reach, retained)) {
        neighbor_indices[0] = static_cast<Int>(retained.size());
        if (!retained.empty()) {
          std::memcpy(
              neighbor_indices.data() + 1,
              retained.data(),
              retained.size() * sizeof(Int));
        }
        return;
      }

      auto detect_neighbors =
          ([=] TMOL_DEVICE_FUNC(
               int candidate, int pose_ind, int block_ind1, int block_ind2) {
            // CPU candidates run concurrently. Classify into disjoint slots;
            // compact them deterministically after the parallel loop instead of
            // racing on the intentionally non-atomic CPU accumulator.
            neighbor_indices[candidate + 1] = -1;
            int const block_type1 = pose_stack_block_type[pose_ind][block_ind1];
            if (block_type1 < 0) return;
            int const block_type2 = pose_stack_block_type[pose_ind][block_ind2];
            if (block_type2 < 0) return;

            Vec<Real, 4> sphere1{0, 0, 0, 0};
            Vec<Real, 4> sphere2{0, 0, 0, 0};
            for (int i = 0; i < 4; ++i) {
              sphere1[i] = block_spheres[pose_ind][block_ind1][i];
              sphere2[i] = block_spheres[pose_ind][block_ind2][i];
            }
            Real const d2 =
                ((sphere1[0] - sphere2[0]) * (sphere1[0] - sphere2[0])
                 + (sphere1[1] - sphere2[1]) * (sphere1[1] - sphere2[1])
                 + (sphere1[2] - sphere2[2]) * (sphere1[2] - sphere2[2]));
            Real const threshold = sphere1[3] + sphere2[3] + reach;
            if (d2 >= threshold * threshold) return;

            neighbor_indices[candidate + 1] = candidate;
          });
      // Decode one triangular index per chunk, then advance through adjacent
      // pairs. Chunks own disjoint output slots and retain canonical ordering.
      constexpr int chunk_size = 128;
      int const n_chunks =
          int((int64_t(n_candidates) + chunk_size - 1) / chunk_size);
      auto detect_chunk = ([=] TMOL_DEVICE_FUNC(int chunk) {
        int const first = chunk * chunk_size;
        int const end = first + min(chunk_size, n_candidates - first);
        int pose = first / n_pairs;
        auto pair = common::upper_triangle_inds_from_linear_index(
            first % n_pairs, max_n_blocks + 1);
        int block1 = common::get<0>(pair);
        int block2 = common::get<1>(pair) - 1;
        for (int candidate = first; candidate < end; ++candidate) {
          detect_neighbors(candidate, pose, block1, block2);
          if (++block2 == max_n_blocks) {
            if (++block1 == max_n_blocks) {
              ++pose;
              block1 = 0;
            }
            block2 = block1;
          }
        }
      });
      DeviceDispatch<D>::template foreach_independent_workgroup<launch_t>(
          mgr, n_chunks, detect_chunk);
      Int n_neighbors = 0;
      for (int candidate = 0; candidate < n_candidates; ++candidate) {
        Int const value = neighbor_indices[candidate + 1];
        if (value >= 0) neighbor_indices[++n_neighbors] = value;
      }
      neighbor_indices[0] = n_neighbors;
    } else {
      auto detect_neighbors = ([=] TMOL_DEVICE_FUNC(int candidate) {
        int const pose_ind = candidate / n_pairs;
        auto pair = common::upper_triangle_inds_from_linear_index(
            candidate % n_pairs, max_n_blocks + 1);
        int const block_ind1 = common::get<0>(pair);
        int const block_ind2 = common::get<1>(pair) - 1;

        int const block_type1 = pose_stack_block_type[pose_ind][block_ind1];
        if (block_type1 < 0) return;
        int const block_type2 = pose_stack_block_type[pose_ind][block_ind2];
        if (block_type2 < 0) return;

        Vec<Real, 4> sphere1{0, 0, 0, 0};
        Vec<Real, 4> sphere2{0, 0, 0, 0};
        for (int i = 0; i < 4; ++i) {
          sphere1[i] = block_spheres[pose_ind][block_ind1][i];
          sphere2[i] = block_spheres[pose_ind][block_ind2][i];
        }
        Real const d2 =
            ((sphere1[0] - sphere2[0]) * (sphere1[0] - sphere2[0])
             + (sphere1[1] - sphere2[1]) * (sphere1[1] - sphere2[1])
             + (sphere1[2] - sphere2[2]) * (sphere1[2] - sphere2[2]));
        Real const threshold = sphere1[3] + sphere2[3] + reach;
        if (d2 >= threshold * threshold) return;

        Int const output = accumulate<D, Int>::add(neighbor_indices[0], Int(1));
        neighbor_indices[output + 1] = candidate;
      });
      DeviceDispatch<D>::template forall_independent<launch_t>(
          mgr, n_candidates, detect_neighbors);
    }
  }
};

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Int>
struct compact_block_neighbors {
  static void f(
      ContextManager& mgr,
      TView<Int, 3, D> block_neighbors,
      TView<Int, 1, D> neighbor_indices,
      TView<Int, 1, D> n_neighbors) {
    LAUNCH_BOX_32;

    int const n_poses = block_neighbors.size(0);
    int const max_n_blocks = block_neighbors.size(1);
    int const n_pairs = common::checked_dispatch_size(
        common::checked_count_product(
            max_n_blocks,
            int64_t(max_n_blocks) + 1,
            "compact block-neighbor candidates per pose")
            / 2,
        "compact block-neighbor candidates per pose");
    int const n_candidates = common::checked_dispatch_product(
        n_poses, n_pairs, "compact block-neighbor candidates");
    if constexpr (D == tmol::Device::CPU) {
      auto compact = ([=] TMOL_DEVICE_FUNC(int candidate) {
        neighbor_indices[candidate] = -1;
        int const pose = candidate / n_pairs;
        auto pair = common::upper_triangle_inds_from_linear_index(
            candidate % n_pairs, max_n_blocks + 1);
        int const block1 = common::get<0>(pair);
        int const block2 = common::get<1>(pair) - 1;
        if (block_neighbors[pose][block1][block2] == 0) return;
        neighbor_indices[candidate] = candidate;
      });
      DeviceDispatch<D>::template forall_independent<launch_t>(
          mgr, n_candidates, compact);
      Int count = 0;
      for (int candidate = 0; candidate < n_candidates; ++candidate) {
        Int const value = neighbor_indices[candidate];
        if (value >= 0) neighbor_indices[count++] = value;
      }
      n_neighbors[0] = count;
    } else {
      auto compact = ([=] TMOL_DEVICE_FUNC(int candidate) {
        int const pose = candidate / n_pairs;
        auto pair = common::upper_triangle_inds_from_linear_index(
            candidate % n_pairs, max_n_blocks + 1);
        int const block1 = common::get<0>(pair);
        int const block2 = common::get<1>(pair) - 1;
        if (block_neighbors[pose][block1][block2] == 0) return;
        int const output = accumulate<D, Int>::add(n_neighbors[0], Int(1));
        neighbor_indices[output] = candidate;
      });
      DeviceDispatch<D>::template forall_independent<launch_t>(
          mgr, n_candidates, compact);
    }
  }
};

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Launch,
    typename Int,
    typename Eval>
void launch_compact_block_neighbors(
    ContextManager& mgr, TView<Int, 3, D> block_neighbors, Eval eval) {
  // Keep the compact count on device so this remains capture-safe. A bounded
  // grid then strides over only neighboring pairs without a host sync.
  int const n_poses = block_neighbors.size(0);
  int const max_n_blocks = block_neighbors.size(1);
  int const n_pairs = common::checked_dispatch_size(
      common::checked_count_product(
          max_n_blocks,
          int64_t(max_n_blocks) + 1,
          "compact block-neighbor candidates per pose")
          / 2,
      "compact block-neighbor candidates per pose");
  int const n_candidates = common::checked_dispatch_product(
      n_poses, n_pairs, "compact block-neighbor candidates");
  if (n_candidates == 0) return;
  auto neighbor_indices_t = TPack<Int, 1, D>::empty({n_candidates});
  auto neighbor_indices = neighbor_indices_t.view;
  auto n_neighbors_t = TPack<Int, 1, D>::zeros({1});
  auto n_neighbors = n_neighbors_t.view;
  compact_block_neighbors<DeviceDispatch, D, Int>::f(
      mgr, block_neighbors, neighbor_indices, n_neighbors);

  constexpr int wide_batch_candidates = 1 << 20;
  constexpr int normal_workgroups = 1 << 14;
  constexpr int wide_batch_workgroups = 1 << 15;
  int const max_workgroups = n_candidates >= wide_batch_candidates
                                 ? wide_batch_workgroups
                                 : normal_workgroups;
  int const n_workgroups =
      n_candidates < max_workgroups ? n_candidates : max_workgroups;
  auto eval_compact = ([=] TMOL_DEVICE_FUNC(int cta) {
    for (int index = cta; index < n_candidates; index += n_workgroups) {
      if (index >= n_neighbors[0]) return;
      eval(neighbor_indices[index]);
    }
  });
  DeviceDispatch<D>::template foreach_workgroup<Launch>(
      mgr, n_workgroups, eval_compact);
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Launch,
    typename Int,
    typename Eval>
void launch_precomputed_block_neighbors(
    ContextManager& mgr,
    TView<Int, 1, D> neighbor_indices,
    int n_poses,
    int max_n_blocks,
    Eval eval) {
  int const n_candidates = neighbor_indices.size(0) - 1;
  if (n_candidates <= 0) return;
#ifdef __NVCC__
  constexpr int normal_workgroups = 1 << 14;
  int const n_workgroups =
      n_candidates < normal_workgroups ? n_candidates : normal_workgroups;
  auto eval_compact = ([=] TMOL_DEVICE_FUNC(int cta) {
    for (int index = cta; index < n_candidates; index += n_workgroups) {
      if (index >= neighbor_indices[0]) return;
      eval(neighbor_indices[index + 1]);
    }
  });
  DeviceDispatch<D>::template foreach_workgroup<Launch>(
      mgr, n_workgroups, eval_compact);
#else
  int const n_neighbors = neighbor_indices[0];
  bool ordered = n_poses > 1;
  for (int index = 2; ordered && index <= n_neighbors; ++index) {
    ordered = neighbor_indices[index - 1] <= neighbor_indices[index];
  }
  if (ordered) {
    int const pairs_per_pose = common::checked_triangular_size(
        max_n_blocks, true, "compact CPU block-pair dispatch");
    common::checked_dispatch_product(
        n_poses, pairs_per_pose, "compact CPU block-pair dispatch");
    // CPU neighbor construction retains ascending candidate IDs. Partition
    // this list by pose: coordinate/energy outputs are disjoint across poses,
    // while each pose keeps the original pair accumulation order. Explicit
    // caller-provided lists with another order retain the serial fallback.
    auto lower_bound = ([=](int candidate) {
      int first = 1;
      int last = n_neighbors + 1;
      while (first < last) {
        int const middle = first + (last - first) / 2;
        if (neighbor_indices[middle] < candidate) {
          first = middle + 1;
        } else {
          last = middle;
        }
      }
      return first;
    });
    auto eval_pose = ([=](int pose) {
      int const first = lower_bound(pose * pairs_per_pose);
      int const last = lower_bound((pose + 1) * pairs_per_pose);
      for (int index = first; index < last; ++index) {
        eval(neighbor_indices[index]);
      }
    });
    DeviceDispatch<D>::template foreach_pose_workgroup<Launch>(
        mgr, n_poses, 1, eval_pose);
  } else {
    auto eval_compact = ([=] TMOL_DEVICE_FUNC(int index) {
      eval(neighbor_indices[index + 1]);
    });
    DeviceDispatch<D>::template foreach_workgroup<Launch>(
        mgr, n_neighbors, eval_compact);
  }
#endif
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Int>
struct rot_neighbor_indices {
  static auto f(
      ContextManager& mgr,
      TView<Int, 3, D> rot_neighbors,
      TView<Int, 1, D> rot_offset_for_pose) -> TPack<Int, 2, D> {
    LAUNCH_BOX_32;

    int n_pose = rot_neighbors.size(0);
    int n_rot = rot_neighbors.size(1);

    int const pose_rot_cells = common::checked_dispatch_product(
        n_pose, n_rot, "rotamer-neighbor dispatch candidates");
    int const n_cells = common::checked_dispatch_product(
        pose_rot_cells, n_rot, "rotamer-neighbor dispatch candidates");
    auto count_for_cell_tp =
        TPack<int64_t, 3, D>::zeros({n_pose, n_rot, n_rot});
    auto count_for_cell = count_for_cell_tp.view;
    auto copy_counts = ([=] TMOL_DEVICE_FUNC(int ind) {
      count_for_cell.data()[ind] = rot_neighbors.data()[ind];
    });
    DeviceDispatch<D>::template forall_independent<launch_t>(
        mgr, n_cells, copy_counts);

    auto offset_for_cell_tp =
        TPack<int64_t, 3, D>::empty({n_pose, n_rot, n_rot});
    auto offset_for_cell = offset_for_cell_tp.view;

    int64_t const n_dispatch_total_64 =
        DeviceDispatch<D>::template scan_and_return_total<
            tmol::portable::scan_type::exclusive>(
            mgr,
            count_for_cell.data(),
            offset_for_cell.data(),
            n_cells,
            tmol::portable::plus_t<int64_t>());
    int const n_dispatch_total = common::checked_dispatch_size(
        n_dispatch_total_64, "rotamer-neighbor dispatch");

    auto rot_neighbor_indices =
        TPack<Int, 2, D>::full({3, n_dispatch_total}, -1);
    auto rot_neighbor_indices_v = rot_neighbor_indices.view;

    auto fill_indices = ([=] TMOL_DEVICE_FUNC(int ind) {
      int pose = ind / (n_rot * n_rot);
      ind = ind % (n_rot * n_rot);
      int rot1 = ind / n_rot;
      int rot2 = ind % n_rot;

      if (rot_neighbors[pose][rot1][rot2]) {
        int offset = offset_for_cell[pose][rot1][rot2];
        rot_neighbor_indices_v[0][offset] = pose;
        rot_neighbor_indices_v[1][offset] = rot1 + rot_offset_for_pose[pose];
        rot_neighbor_indices_v[2][offset] = rot2 + rot_offset_for_pose[pose];
      }
    });

    DeviceDispatch<D>::template forall_independent<launch_t>(
        mgr, n_cells, fill_indices);

    return rot_neighbor_indices;
  }
};

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Int>
struct block_neighbor_indices {
  static auto f(ContextManager& mgr, TView<Int, 3, D> block_neighbors)
      -> TPack<Int, 2, D> {
    LAUNCH_BOX_32;

    int n_pose = block_neighbors.size(0);
    int n_res = block_neighbors.size(1);

    int const pose_res_cells = common::checked_dispatch_product(
        n_pose, n_res, "block-neighbor dispatch candidates");
    int const n_cells = common::checked_dispatch_product(
        pose_res_cells, n_res, "block-neighbor dispatch candidates");
    auto count_for_cell_tp =
        TPack<int64_t, 3, D>::zeros({n_pose, n_res, n_res});
    auto count_for_cell = count_for_cell_tp.view;
    auto copy_counts = ([=] TMOL_DEVICE_FUNC(int ind) {
      count_for_cell.data()[ind] = block_neighbors.data()[ind];
    });
    DeviceDispatch<D>::template forall_independent<launch_t>(
        mgr, n_cells, copy_counts);

    auto offset_for_cell_tp =
        TPack<int64_t, 3, D>::empty({n_pose, n_res, n_res});
    auto offset_for_cell = offset_for_cell_tp.view;

    int64_t const n_dispatch_total_64 =
        DeviceDispatch<D>::template scan_and_return_total<
            tmol::portable::scan_type::exclusive>(
            mgr,
            count_for_cell.data(),
            offset_for_cell.data(),
            n_cells,
            tmol::portable::plus_t<int64_t>());
    int const n_dispatch_total = common::checked_dispatch_size(
        n_dispatch_total_64, "block-neighbor dispatch");

    auto block_neighbor_indices =
        TPack<Int, 2, D>::full({3, n_dispatch_total}, -1);
    auto block_neighbor_indices_v = block_neighbor_indices.view;

    auto fill_indices = ([=] TMOL_DEVICE_FUNC(int ind) {
      int pose = ind / (n_res * n_res);
      ind = ind % (n_res * n_res);
      int res1 = ind / n_res;
      int res2 = ind % n_res;

      if (block_neighbors[pose][res1][res2]) {
        int offset = offset_for_cell[pose][res1][res2];
        block_neighbor_indices_v[0][offset] = pose;
        block_neighbor_indices_v[1][offset] = res1;
        block_neighbor_indices_v[2][offset] = res2;
      }
    });

    DeviceDispatch<D>::template forall_independent<launch_t>(
        mgr, n_cells, fill_indices);

    return block_neighbor_indices;
  }
};

// Compute block-level bounding spheres that enclose all rotamers for each
// block, using pre-computed per-rotamer spheres.  One thread per block; serial
// loop over n_rots_for_block[pose][block] rotamers.  Safe for packing (no
// write races) unlike compute_block_spheres which launches per rotamer.
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct compute_block_spheres_from_rot_spheres {
  static void f(
      ContextManager& mgr,
      TView<Real, 2, D> rot_spheres,          // [n_rots_global, 4]
      TView<Int, 2, D> n_rots_for_block,      // [n_poses, max_n_blocks]
      TView<Int, 2, D> rot_offset_for_block,  // [n_poses, max_n_blocks] global
      TView<Real, 3, D> block_spheres         // [n_poses, max_n_blocks, 4] out
  ) {
    LAUNCH_BOX_32;

    int const n_poses = n_rots_for_block.size(0);
    int const max_n_blocks = n_rots_for_block.size(1);
    int const n_pose_blocks = common::checked_dispatch_product(
        n_poses, max_n_blocks, "rotamer block-sphere dispatch");

    BlockSpheresFromRotCtxT<
        tmol::score::common::TViewBinder<D>::template View,
        Real,
        Int>
        ctx;
    ctx.rot_spheres = rot_spheres;
    ctx.n_rots_for_block = n_rots_for_block;
    ctx.rot_offset_for_block = rot_offset_for_block;
    ctx.block_spheres = block_spheres;
    ctx.max_n_blocks = max_n_blocks;

    if (D == tmol::Device::MPS) {
      block_spheres_from_rot_mps_launch_if<D>(ctx, n_pose_blocks);
    } else {
      auto compute = ([=] TMOL_DEVICE_FUNC(int ind) {
        compute_block_spheres_from_rot_body<D, Real, Int>(ctx, ind);
      });
      DeviceDispatch<D>::template forall_independent<launch_t>(
          mgr, n_pose_blocks, compute);
    }
  }
};

// Convert block and rotamer bounding spheres into rotamer-pair dispatch indices
// (the same [3, n_pairs] format as rot_neighbor_indices). Canonical packed
// upper-triangle block pairs avoid all square block-pair scratch while
// retaining the original pose-major, b1-major, b2-major ordering.
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct rot_neighbor_indices_from_block_neighbors {
  // Exclusive scan of the per-candidate counts.
  //
  // DeviceDispatch::scan takes raw pointers, which Metal cannot honour: an
  // MPS tensor's data pointer is an MTLBuffer handle, not an address a kernel
  // can be handed, and the stable ABI exposes no cumsum. The scan runs on the
  // host there instead -- which costs no extra synchronization, because the
  // caller has already blocked on the overflow flag and the count sum
  // immediately above.
  static auto exclusive_scan_counts(
      ContextManager& mgr, TPack<Int, 1, D>& counts_t, int n_candidates)
      -> TPack<Int, 1, D> {
    if (D == tmol::Device::MPS) {
      auto host_counts =
          counts_t.tensor.to(tmol::native::Device(tmol::native::kCPU));
      auto host_offsets_t =
          TPack<Int, 1, tmol::Device::CPU>::empty({int64_t(n_candidates)});
      Int const* counts = host_counts.template data_ptr<Int>();
      Int* offsets = host_offsets_t.tensor.template data_ptr<Int>();
      Int running = 0;
      for (int i = 0; i < n_candidates; ++i) {
        offsets[i] = running;
        running += counts[i];
      }
      return TPack<Int, 1, D>(
          host_offsets_t.tensor.to(counts_t.tensor.device()));
    }
    auto offsets_t = TPack<Int, 1, D>::zeros({n_candidates});
    DeviceDispatch<D>::template scan<tmol::portable::scan_type::exclusive>(
        mgr,
        counts_t.view.data(),
        offsets_t.view.data(),
        n_candidates,
        tmol::portable::plus_t<Int>());
    return offsets_t;
  }

  static auto f(
      ContextManager& mgr,
      TView<Int, 2, D> pose_stack_block_type,
      TView<Real, 3, D> block_spheres,
      TView<Int, 2, D> n_rots_for_block,      // [n_poses, max_n_blocks]
      TView<Int, 2, D> rot_offset_for_block,  // [n_poses, max_n_blocks] global
      TView<Real, 2, D> rot_spheres,          // [n_rots_global, 4]
      TView<Int, 2, D> lockstep_group_for_block,
      Real reach,
      int64_t candidate_begin = 0,
      int64_t candidate_count = -1) -> TPack<Int, 2, D> {
    LAUNCH_BOX_32;

    int const n_poses = pose_stack_block_type.size(0);
    int const max_n_blocks = pose_stack_block_type.size(1);
    int const block_pairs_per_pose = common::checked_triangular_size(
        max_n_blocks, true, "rotamer block-pair dispatch candidates per pose");
    int const total_candidates = common::checked_dispatch_product(
        n_poses,
        block_pairs_per_pose,
        "rotamer block-pair dispatch candidates");
    if (candidate_count < 0) {
      candidate_count = int64_t(total_candidates) - candidate_begin;
    }
    if (candidate_begin < 0 || candidate_count < 0
        || candidate_begin + candidate_count > total_candidates) {
      throw std::out_of_range("rotamer block-pair dispatch window is invalid");
    }
    int const n_candidates = common::checked_dispatch_size(
        candidate_count, "rotamer block-pair dispatch window");
    if (n_candidates == 0) {
      return TPack<Int, 2, D>::empty({3, 0});
    }

    // Step 1: per-block-pair rotamer pair counts.
    // For diagonal (b1==b2): only self-pairs (r,r), count = n_rots[b1].
    // For off-diagonal (b1<b2): count the overlapping rotamer spheres.
    auto pair_counts_t = TPack<Int, 1, D>::zeros({n_candidates});
    auto pair_count_overflow_t = TPack<int64_t, 1, D>::zeros({1});

    RotPairListCtxT<
        tmol::score::common::TViewBinder<D>::template View,
        Real,
        Int>
        ctx;
    ctx.pose_stack_block_type = pose_stack_block_type;
    ctx.block_spheres = block_spheres;
    ctx.n_rots_for_block = n_rots_for_block;
    ctx.rot_offset_for_block = rot_offset_for_block;
    ctx.rot_spheres = rot_spheres;
    ctx.lockstep_group_for_block = lockstep_group_for_block;
    ctx.pair_counts = pair_counts_t.view;
    ctx.pair_count_overflow = pair_count_overflow_t.view;
    ctx.max_n_blocks = max_n_blocks;
    ctx.block_pairs_per_pose = block_pairs_per_pose;
    ctx.candidate_begin = int(candidate_begin);
    ctx.reach = reach;
    // The fill pass reuses this context; its two remaining views are bound
    // once the scan and the output allocation exist. They are bound to the
    // counts for now so the Metal collector never sees an undefined view.
    ctx.pair_offsets = pair_counts_t.view;
    auto indices_placeholder_t = TPack<Int, 2, D>::empty({3, 1});
    ctx.indices = indices_placeholder_t.view;

    if (D == tmol::Device::MPS) {
      rot_pair_list_mps_launch_if<D>(ctx, n_candidates, "rot_pair_counts_f32");
    } else {
      auto compute_counts = ([=] TMOL_DEVICE_FUNC(int candidate) {
        rot_pair_counts_body<D, Real, Int>(ctx, candidate);
      });
      DeviceDispatch<D>::template forall_independent<launch_t>(
          mgr, n_candidates, compute_counts);
    }

    // Step 2: check an int64 reduction before scanning int32 offsets. This
    // preserves the signed-32-bit dispatch limit without two int64 pair arrays.
    if (pair_count_overflow_t.tensor.template item<int64_t>() != 0) {
      throw std::overflow_error(
          "rotamer block-pair dispatch exceeds the signed 32-bit native "
          "dispatch limit (2147483647)");
    }
    int64_t const total_64 =
        pair_counts_t.tensor.sum(tmol::native::ScalarType::Long)
            .template item<int64_t>();
    int const total =
        common::checked_dispatch_size(total_64, "rotamer block-pair dispatch");

    auto pair_offsets_t =
        exclusive_scan_counts(mgr, pair_counts_t, n_candidates);

    // Step 3: allocate output [3, total]
    auto indices_t = TPack<Int, 2, D>::full({3, total}, -1);
    ctx.pair_offsets = pair_offsets_t.view;
    ctx.indices = indices_t.view;

    // Step 4: fill -- one thread per block pair, serial loop over rot pairs.
    // Diagonal (b1==b2): only (r,r) self-pairs (intrares scoring).
    // Same lockstep group: matching states; otherwise overlapping spheres.
    if (D == tmol::Device::MPS) {
      rot_pair_list_mps_launch_if<D>(ctx, n_candidates, "rot_pair_fill_f32");
    } else {
      auto fill = ([=] TMOL_DEVICE_FUNC(int candidate) {
        rot_pair_fill_body<D, Real, Int>(ctx, candidate);
      });
      DeviceDispatch<D>::template forall_independent<launch_t>(
          mgr, n_candidates, fill);
    }

    return indices_t;
  }
};

}  // namespace sphere_overlap
}  // namespace common
}  // namespace score
}  // namespace tmol
