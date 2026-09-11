#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/counting.hh>

#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/upper_triangle_indices.hh>

#include <moderngpu/operators.hxx>
#include <moderngpu/scan_types.hxx>
#include <tmol/utility/tensor/context_manager.hh>

namespace tmol {
namespace score {
namespace common {
namespace sphere_overlap {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

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
  static void f(
      ContextManager& mgr,
      TView<Vec<Real, 3>, 1, D> rot_coords,
      TView<Int, 1, D> rot_coord_offset,
      TView<Int, 1, D> block_type_ind_for_rot,
      TView<Int, 1, D> block_type_n_atoms,
      TView<Real, 2, D> rot_spheres) {
    LAUNCH_BOX_32;

    auto compute_spheres = ([=] TMOL_DEVICE_FUNC(int cta) {
      CTA_LAUNCH_T_PARAMS;

      int const rot_ind = cta;
      int const coord_offset = rot_coord_offset[rot_ind];
      int const block_type = block_type_ind_for_rot[rot_ind];

      if (block_type < 0) return;
      int const n_atoms = block_type_n_atoms[block_type];
      Vec<Real, 3> local_coords(0, 0, 0);

      auto per_thread_com = ([&] TMOL_DEVICE_FUNC(int tid) {
        for (int i = tid; i < n_atoms; i += nt) {
          Vec<Real, 3> ci = rot_coords[coord_offset + i];
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
              nt>(local_coords, mgpu::plus_t<Real>());

      Real d2max = 0;
      // Now find maximum distance
      auto per_thread_dist_to_com = ([&] TMOL_DEVICE_FUNC(int tid) {
        for (int i = tid; i < n_atoms; i += nt) {
          Vec<Real, 3> ci = rot_coords[coord_offset + i];
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
            dmax, mgpu::maximum_t<Real>());
      } else {
        d2max = DeviceDispatch<D>::template shuffle_reduce_in_workgroup<nt>(
            d2max, mgpu::maximum_t<Real>());
      }

      auto thread0_write_out_result = ([=] TMOL_DEVICE_FUNC(int tid) {
        if (tid == 0) {
          rot_spheres[rot_ind][0] = com[0];
          rot_spheres[rot_ind][1] = com[1];
          rot_spheres[rot_ind][2] = com[2];
          if constexpr (D == Device::CPU) {
            rot_spheres[rot_ind][3] = dmax;
          } else {
            // Reduce squared distances first and take one square root per
            // block, rather than one per participating GPU thread.
            rot_spheres[rot_ind][3] = sqrt(d2max);
          }
        }
      });
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(
          thread0_write_out_result);
    });

    DeviceDispatch<D>::template foreach_independent_workgroup<launch_t>(
        mgr, rot_coord_offset.size(0), compute_spheres);
  }
};

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    bool ResetCount = false>
struct compute_block_spheres {
  static void f(
      ContextManager& mgr,
      TView<Vec<Real, 3>, 1, D> rot_coords,
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
      Vec<Real, 3> local_coords(0, 0, 0);

      auto per_thread_com = ([&] TMOL_DEVICE_FUNC(int tid) {
        for (int i = tid; i < n_atoms; i += nt) {
          Vec<Real, 3> ci = rot_coords[coord_offset + i];
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
              nt>(local_coords, mgpu::plus_t<Real>());

      Real d2max = 0;
      // Now find maximum distance
      auto per_thread_dist_to_com = ([&] TMOL_DEVICE_FUNC(int tid) {
        for (int i = tid; i < n_atoms; i += nt) {
          Vec<Real, 3> ci = rot_coords[coord_offset + i];
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
            dmax, mgpu::maximum_t<Real>());
      } else {
        d2max = DeviceDispatch<D>::template shuffle_reduce_in_workgroup<nt>(
            d2max, mgpu::maximum_t<Real>());
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

    DeviceDispatch<D>::template foreach_workgroup<launch_t>(
        mgr, block_ind_for_rot.size(0), compute_spheres);
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

      Vec<Real, 4> sphere1(0, 0, 0, 0);
      Vec<Real, 4> sphere2(0, 0, 0, 0);

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

      Vec<Real, 4> sphere1(0, 0, 0, 0);
      Vec<Real, 4> sphere2(0, 0, 0, 0);

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
    DeviceDispatch<D>::template forall_independent<launch_t>(
        mgr, n_block_pairs, detect_neighbors);
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

      auto detect_neighbors = ([=] TMOL_DEVICE_FUNC(int candidate) {
        // CPU candidates run concurrently. Classify into disjoint slots;
        // compact them deterministically after the parallel loop instead of
        // racing on the intentionally non-atomic CPU accumulator.
        neighbor_indices[candidate + 1] = -1;
        int const pose_ind = candidate / n_pairs;
        auto pair = common::upper_triangle_inds_from_linear_index(
            candidate % n_pairs, max_n_blocks + 1);
        int const block_ind1 = common::get<0>(pair);
        int const block_ind2 = common::get<1>(pair) - 1;

        int const block_type1 = pose_stack_block_type[pose_ind][block_ind1];
        if (block_type1 < 0) return;
        int const block_type2 = pose_stack_block_type[pose_ind][block_ind2];
        if (block_type2 < 0) return;

        Vec<Real, 4> sphere1(0, 0, 0, 0);
        Vec<Real, 4> sphere2(0, 0, 0, 0);
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
      DeviceDispatch<D>::template forall_independent<launch_t>(
          mgr, n_candidates, detect_neighbors);
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

        Vec<Real, 4> sphere1(0, 0, 0, 0);
        Vec<Real, 4> sphere2(0, 0, 0, 0);
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
        DeviceDispatch<D>::template scan_and_return_total<mgpu::scan_type_exc>(
            mgr,
            count_for_cell.data(),
            offset_for_cell.data(),
            n_cells,
            mgpu::plus_t<int64_t>());
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
        DeviceDispatch<D>::template scan_and_return_total<mgpu::scan_type_exc>(
            mgr,
            count_for_cell.data(),
            offset_for_cell.data(),
            n_cells,
            mgpu::plus_t<int64_t>());
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

    auto compute = ([=] TMOL_DEVICE_FUNC(int ind) {
      int const pose = ind / max_n_blocks;
      int const block = ind % max_n_blocks;
      int const n_rots = n_rots_for_block[pose][block];
      if (n_rots <= 0) return;
      int const rot_start = rot_offset_for_block[pose][block];
      if (rot_start < 0) return;

      Real cx = 0, cy = 0, cz = 0;
      for (int r = 0; r < n_rots; ++r) {
        cx += rot_spheres[rot_start + r][0];
        cy += rot_spheres[rot_start + r][1];
        cz += rot_spheres[rot_start + r][2];
      }
      cx /= n_rots;
      cy /= n_rots;
      cz /= n_rots;

      Real rmax = 0;
      for (int r = 0; r < n_rots; ++r) {
        Real dx = rot_spheres[rot_start + r][0] - cx;
        Real dy = rot_spheres[rot_start + r][1] - cy;
        Real dz = rot_spheres[rot_start + r][2] - cz;
        Real d =
            sqrt(dx * dx + dy * dy + dz * dz) + rot_spheres[rot_start + r][3];
        if (d > rmax) rmax = d;
      }

      block_spheres[pose][block][0] = cx;
      block_spheres[pose][block][1] = cy;
      block_spheres[pose][block][2] = cz;
      block_spheres[pose][block][3] = rmax;
    });

    DeviceDispatch<D>::template forall_independent<launch_t>(
        mgr, n_pose_blocks, compute);
  }
};

template <tmol::Device D, typename Real>
inline TMOL_DEVICE_FUNC bool rot_spheres_overlap(
    TView<Real, 2, D> rot_spheres, int rot1, int rot2, Real reach) {
  Real const dx = rot_spheres[rot1][0] - rot_spheres[rot2][0];
  Real const dy = rot_spheres[rot1][1] - rot_spheres[rot2][1];
  Real const dz = rot_spheres[rot1][2] - rot_spheres[rot2][2];
  Real const threshold = rot_spheres[rot1][3] + rot_spheres[rot2][3] + reach;
  return dx * dx + dy * dy + dz * dz < threshold * threshold;
}

// Convert a block-level neighbor matrix into rotamer-pair dispatch indices
// (the same [3, n_pairs] format as rot_neighbor_indices). The block spheres
// cheaply reject whole block pairs; the rotamer spheres then omit individual
// pairs that cannot interact. This avoids the O(max_n_rots^2) dense matrix
// used by rot_neighbor_indices without retaining guaranteed-zero score-table
// entries.
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct rot_neighbor_indices_from_block_neighbors {
  static auto
  f(ContextManager& mgr,
    TView<Int, 3, D> block_neighbors,   // [n_poses, max_n_blocks, max_n_blocks]
    TView<Int, 2, D> n_rots_for_block,  // [n_poses, max_n_blocks]
    TView<Int, 2, D> rot_offset_for_block,  // [n_poses, max_n_blocks] global
    TView<Real, 2, D> rot_spheres,          // [n_rots_global, 4]
    Real reach) -> TPack<Int, 2, D> {
    LAUNCH_BOX_32;

    int const n_poses = block_neighbors.size(0);
    int const max_n_blocks = block_neighbors.size(1);
    int const block_pair_cells = common::checked_dispatch_product(
        max_n_blocks,
        max_n_blocks,
        "rotamer block-pair dispatch candidates per pose");
    int const n_cells = common::checked_dispatch_product(
        n_poses, block_pair_cells, "rotamer block-pair dispatch candidates");

    // Step 1: per-block-pair rotamer pair counts.
    // For diagonal (b1==b2): only self-pairs (r,r), count = n_rots[b1].
    // For off-diagonal (b1<b2): count the overlapping rotamer spheres.
    auto pair_counts_t =
        TPack<int64_t, 3, D>::zeros({n_poses, max_n_blocks, max_n_blocks});
    auto pair_counts = pair_counts_t.view;

    auto compute_counts = ([=] TMOL_DEVICE_FUNC(int ind) {
      int const pose = ind / block_pair_cells;
      int const bp = ind % block_pair_cells;
      int const b1 = bp / max_n_blocks;
      int const b2 = bp % max_n_blocks;
      if (block_neighbors[pose][b1][b2]) {
        int const nr1 = n_rots_for_block[pose][b1];
        int const nr2 = n_rots_for_block[pose][b2];
        int const off1 = rot_offset_for_block[pose][b1];
        int const off2 = rot_offset_for_block[pose][b2];
        if (nr1 <= 0 || nr2 <= 0 || off1 < 0 || off2 < 0) return;
        if (b1 == b2) {
          pair_counts[pose][b1][b2] = nr1;
        } else {
          int64_t count = 0;
          for (int i = 0; i < nr1; ++i) {
            for (int j = 0; j < nr2; ++j) {
              count += rot_spheres_overlap<D>(
                  rot_spheres, off1 + i, off2 + j, reach);
            }
          }
          pair_counts[pose][b1][b2] = count;
        }
      }
    });
    DeviceDispatch<D>::template forall_independent<launch_t>(
        mgr, n_cells, compute_counts);

    // Step 2: prefix scan → per-block-pair offsets and total
    auto pair_offsets_t =
        TPack<int64_t, 3, D>::zeros({n_poses, max_n_blocks, max_n_blocks});
    auto pair_offsets = pair_offsets_t.view;

    int64_t const total_64 =
        DeviceDispatch<D>::template scan_and_return_total<mgpu::scan_type_exc>(
            mgr,
            pair_counts.data(),
            pair_offsets.data(),
            n_cells,
            mgpu::plus_t<int64_t>());
    int const total =
        common::checked_dispatch_size(total_64, "rotamer block-pair dispatch");

    // Step 3: allocate output [3, total]
    auto indices_t = TPack<Int, 2, D>::full({3, total}, -1);
    auto indices = indices_t.view;

    // Step 4: fill — one thread per block pair, serial loop over rot pairs.
    // Diagonal (b1==b2): only (r,r) self-pairs (intrares scoring).
    // Off-diagonal (b1<b2): only overlapping rotamer spheres.
    auto fill = ([=] TMOL_DEVICE_FUNC(int ind) {
      int const pose = ind / block_pair_cells;
      int const bp = ind % block_pair_cells;
      int const b1 = bp / max_n_blocks;
      int const b2 = bp % max_n_blocks;
      if (!block_neighbors[pose][b1][b2]) return;

      int const nr1 = n_rots_for_block[pose][b1];
      int const nr2 = n_rots_for_block[pose][b2];
      int const off1 = rot_offset_for_block[pose][b1];
      int const off2 = rot_offset_for_block[pose][b2];
      if (off1 < 0 || off2 < 0) return;

      int64_t offset = pair_offsets[pose][b1][b2];
      if (b1 == b2) {
        for (int i = 0; i < nr1; ++i) {
          indices[0][offset] = pose;
          indices[1][offset] = off1 + i;
          indices[2][offset] = off1 + i;  // same rot
          ++offset;
        }
      } else {
        for (int i = 0; i < nr1; ++i) {
          for (int j = 0; j < nr2; ++j) {
            if (!rot_spheres_overlap<D>(
                    rot_spheres, off1 + i, off2 + j, reach)) {
              continue;
            }
            indices[0][offset] = pose;
            indices[1][offset] = off1 + i;
            indices[2][offset] = off2 + j;
            ++offset;
          }
        }
      }
    });
    DeviceDispatch<D>::template forall_independent<launch_t>(
        mgr, n_cells, fill);

    return indices_t;
  }
};

}  // namespace sphere_overlap
}  // namespace common
}  // namespace score
}  // namespace tmol
