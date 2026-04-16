#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/score/common/accumulate.hh>

#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>

#include <moderngpu/operators.hxx>

namespace tmol {
namespace score {
namespace common {
namespace sphere_overlap {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct compute_rot_spheres {
  static void f(
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

      // The center of mass
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
        dmax = sqrt(d2max);
      });
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(
          per_thread_dist_to_com);

      dmax = DeviceDispatch<D>::template shuffle_reduce_in_workgroup<nt>(
          dmax, mgpu::maximum_t<Real>());

      auto thread0_write_out_result = ([=] TMOL_DEVICE_FUNC(int tid) {
        if (tid == 0) {
          rot_spheres[rot_ind][0] = com[0];
          rot_spheres[rot_ind][1] = com[1];
          rot_spheres[rot_ind][2] = com[2];
          rot_spheres[rot_ind][3] = dmax;
        }
      });
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(
          thread0_write_out_result);
    });

    DeviceDispatch<D>::template foreach_workgroup<launch_t>(
        rot_coord_offset.size(0), compute_spheres);
  }
};

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct compute_block_spheres {
  static void f(
      TView<Vec<Real, 3>, 1, D> rot_coords,
      TView<Int, 1, D> rot_coord_offset,
      TView<Int, 1, D> block_ind_for_rot,
      TView<Int, 1, D> pose_ind_for_rot,
      TView<Int, 1, D> block_type_ind_for_rot,
      TView<Int, 1, D> block_type_n_atoms,
      TView<Real, 3, D> block_spheres) {
    LAUNCH_BOX_32;

    auto compute_spheres = ([=] TMOL_DEVICE_FUNC(int cta) {
      CTA_LAUNCH_T_PARAMS;

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

      // The center of mass
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
        dmax = sqrt(d2max);
      });
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(
          per_thread_dist_to_com);

      dmax = DeviceDispatch<D>::template shuffle_reduce_in_workgroup<nt>(
          dmax, mgpu::maximum_t<Real>());

      auto thread0_write_out_result = ([=] TMOL_DEVICE_FUNC(int tid) {
        if (tid == 0) {
          block_spheres[pose_ind][block_ind][0] = com[0];
          block_spheres[pose_ind][block_ind][1] = com[1];
          block_spheres[pose_ind][block_ind][2] = com[2];
          block_spheres[pose_ind][block_ind][3] = dmax;
        }
      });
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(
          thread0_write_out_result);
    });

    DeviceDispatch<D>::template foreach_workgroup<launch_t>(
        block_ind_for_rot.size(0), compute_spheres);
  }
};

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct detect_rot_neighbors {
  static void f(
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

    auto detect_neighbors = ([=] TMOL_DEVICE_FUNC(int ind) {
      int const n_poses = n_rots_for_block.size(0);
      int const max_n_rots = max_n_rots_per_pose;
      int const n_block_types = block_type_n_atoms.size(0);

      if (ind >= n_poses * max_n_rots * max_n_rots) return;

      int const pose_ind = ind / (max_n_rots * max_n_rots);
      int const rot_pair_ind = ind % (max_n_rots * max_n_rots);
      int const rot_ind1 = rot_pair_ind / max_n_rots;
      int const rot_ind2 = rot_pair_ind % max_n_rots;

      if (rot_ind1 >= n_rots_for_pose[pose_ind]
          || rot_ind2 >= n_rots_for_pose[pose_ind]) {
        return;
      }

      if (rot_ind1 > rot_ind2) {
        return;
      }

      int const global_rot_ind1 = rot_ind1 + rot_offset_for_pose[pose_ind];
      int const global_rot_ind2 = rot_ind2 + rot_offset_for_pose[pose_ind];

      int const block_ind1 = block_ind_for_rot[global_rot_ind1];
      int const block_ind2 = block_ind_for_rot[global_rot_ind2];

      bool same_rot = rot_ind1 == rot_ind2;
      bool same_block = block_ind1 == block_ind2;
      if (same_block && !same_rot) {
        return;
      }

      int const block_type1 = block_type_ind_for_rot[global_rot_ind1];
      if (block_type1 < 0) {
        return;
      }
      int const block_type2 = block_type_ind_for_rot[global_rot_ind2];
      if (block_type2 < 0) {
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

      if (d2 < d_threshold * d_threshold) {
        rot_neighbors[pose_ind][rot_ind1][rot_ind2] = 1;
      }
    });
    std::uint64_t n_rot_pairs = std::uint64_t(n_rots_for_block.size(0))
                                * max_n_rots_per_pose * max_n_rots_per_pose;

    DeviceDispatch<D>::template forall<launch_t>(n_rot_pairs, detect_neighbors);
  }
};

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct detect_block_neighbors {
  static void f(
      TView<Int, 2, D> pose_stack_block_type,
      TView<Real, 3, D> block_spheres,
      TView<Int, 3, D> block_neighbors,
      Real reach) {
    LAUNCH_BOX_32;

    auto detect_neighbors = ([=] TMOL_DEVICE_FUNC(int ind) {
      int const n_poses = pose_stack_block_type.size(0);
      int const max_n_blocks = pose_stack_block_type.size(1);

      if (ind >= n_poses * max_n_blocks * max_n_blocks) return;

      int const pose_ind = ind / (max_n_blocks * max_n_blocks);
      int const block_pair_ind = ind % (max_n_blocks * max_n_blocks);
      int const block_ind1 = block_pair_ind / max_n_blocks;
      int const block_ind2 = block_pair_ind % max_n_blocks;

      if (block_ind1 > block_ind2) {
        return;
      }

      int const block_type1 = pose_stack_block_type[pose_ind][block_ind1];
      if (block_type1 < 0) {
        return;
      }
      int const block_type2 = pose_stack_block_type[pose_ind][block_ind2];
      if (block_type2 < 0) {
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

      if (d2 < d_threshold * d_threshold) {
        block_neighbors[pose_ind][block_ind1][block_ind2] = 1;
      }
    });
    int n_block_pairs = pose_stack_block_type.size(0)
                        * pose_stack_block_type.size(1)
                        * pose_stack_block_type.size(1);

    DeviceDispatch<D>::template forall<launch_t>(
        n_block_pairs, detect_neighbors);
  }
};

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Int>
struct rot_neighbor_indices {
  static auto f(
      TView<Int, 3, D> rot_neighbors, TView<Int, 1, D> rot_offset_for_pose)
      -> TPack<Int, 2, D> {
    LAUNCH_BOX_32;

    int n_pose = rot_neighbors.size(0);
    int n_rot = rot_neighbors.size(1);

    int n_cells = n_pose * n_rot * n_rot;
    auto offset_for_cell_tp = TPack<Int, 3, D>::zeros_like(rot_neighbors);
    auto offset_for_cell = offset_for_cell_tp.view;

    int n_dispatch_total =
        DeviceDispatch<D>::template scan_and_return_total<mgpu::scan_type_exc>(
            rot_neighbors.data(),
            offset_for_cell.data(),
            n_cells,
            mgpu::plus_t<Int>());

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

    DeviceDispatch<D>::template forall<launch_t>(n_cells, fill_indices);

    return rot_neighbor_indices;
  }
};

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Int>
struct block_neighbor_indices {
  static auto f(TView<Int, 3, D> block_neighbors) -> TPack<Int, 2, D> {
    LAUNCH_BOX_32;

    int n_pose = block_neighbors.size(0);
    int n_res = block_neighbors.size(1);

    int n_cells = n_pose * n_res * n_res;
    auto offset_for_cell_tp = TPack<Int, 3, D>::zeros_like(block_neighbors);
    auto offset_for_cell = offset_for_cell_tp.view;

    int n_dispatch_total =
        DeviceDispatch<D>::template scan_and_return_total<mgpu::scan_type_exc>(
            block_neighbors.data(),
            offset_for_cell.data(),
            n_cells,
            mgpu::plus_t<Int>());

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

    DeviceDispatch<D>::template forall<launch_t>(n_cells, fill_indices);

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
      TView<Real, 2, D> rot_spheres,          // [n_rots_global, 4]
      TView<Int, 2, D> n_rots_for_block,      // [n_poses, max_n_blocks]
      TView<Int, 2, D> rot_offset_for_block,  // [n_poses, max_n_blocks] global
      TView<Real, 3, D> block_spheres         // [n_poses, max_n_blocks, 4] out
  ) {
    LAUNCH_BOX_32;

    int const n_poses = n_rots_for_block.size(0);
    int const max_n_blocks = n_rots_for_block.size(1);

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

    DeviceDispatch<D>::template forall<launch_t>(
        n_poses * max_n_blocks, compute);
  }
};

// Convert a block-level neighbor matrix into rotamer-pair dispatch indices
// (the same [3, n_pairs] format as rot_neighbor_indices), expanding each
// neighboring block pair (b1, b2) into all n_rots[b1]*n_rots[b2] pairs.
// This avoids the O(max_n_rots^2) dense matrix used by rot_neighbor_indices.
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Int>
struct rot_neighbor_indices_from_block_neighbors {
  static auto
  f(TView<Int, 3, D> block_neighbors,   // [n_poses, max_n_blocks, max_n_blocks]
    TView<Int, 2, D> n_rots_for_block,  // [n_poses, max_n_blocks]
    TView<Int, 2, D> rot_offset_for_block  // [n_poses, max_n_blocks] global
    ) -> TPack<Int, 2, D> {
    LAUNCH_BOX_32;

    int const n_poses = block_neighbors.size(0);
    int const max_n_blocks = block_neighbors.size(1);
    int const n_cells = n_poses * max_n_blocks * max_n_blocks;

    // Step 1: per-block-pair rotamer pair counts.
    // For diagonal (b1==b2): only self-pairs (r,r), count = n_rots[b1].
    // For off-diagonal (b1<b2): all pairs, count = n_rots[b1]*n_rots[b2].
    auto pair_counts_t = TPack<Int, 3, D>::zeros_like(block_neighbors);
    auto pair_counts = pair_counts_t.view;

    auto compute_counts = ([=] TMOL_DEVICE_FUNC(int ind) {
      int const pose = ind / (max_n_blocks * max_n_blocks);
      int const bp = ind % (max_n_blocks * max_n_blocks);
      int const b1 = bp / max_n_blocks;
      int const b2 = bp % max_n_blocks;
      if (block_neighbors[pose][b1][b2]) {
        if (b1 == b2) {
          pair_counts[pose][b1][b2] = n_rots_for_block[pose][b1];
        } else {
          pair_counts[pose][b1][b2] =
              n_rots_for_block[pose][b1] * n_rots_for_block[pose][b2];
        }
      }
    });
    DeviceDispatch<D>::template forall<launch_t>(n_cells, compute_counts);

    // Step 2: prefix scan → per-block-pair offsets and total
    auto pair_offsets_t = TPack<Int, 3, D>::zeros_like(block_neighbors);
    auto pair_offsets = pair_offsets_t.view;

    int const total =
        DeviceDispatch<D>::template scan_and_return_total<mgpu::scan_type_exc>(
            pair_counts.data(),
            pair_offsets.data(),
            n_cells,
            mgpu::plus_t<Int>());

    // Step 3: allocate output [3, total]
    auto indices_t = TPack<Int, 2, D>::full({3, total}, -1);
    auto indices = indices_t.view;

    // Step 4: fill — one thread per block pair, serial loop over rot pairs.
    // Diagonal (b1==b2): only (r,r) self-pairs (intrares scoring).
    // Off-diagonal (b1<b2): all nr1*nr2 pairs.
    auto fill = ([=] TMOL_DEVICE_FUNC(int ind) {
      int const pose = ind / (max_n_blocks * max_n_blocks);
      int const bp = ind % (max_n_blocks * max_n_blocks);
      int const b1 = bp / max_n_blocks;
      int const b2 = bp % max_n_blocks;
      if (!block_neighbors[pose][b1][b2]) return;

      int const nr1 = n_rots_for_block[pose][b1];
      int const nr2 = n_rots_for_block[pose][b2];
      int const off1 = rot_offset_for_block[pose][b1];
      int const off2 = rot_offset_for_block[pose][b2];
      if (off1 < 0 || off2 < 0) return;

      int offset = pair_offsets[pose][b1][b2];
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
            indices[0][offset] = pose;
            indices[1][offset] = off1 + i;
            indices[2][offset] = off2 + j;
            ++offset;
          }
        }
      }
    });
    DeviceDispatch<D>::template forall<launch_t>(n_cells, fill);

    return indices_t;
  }
};

}  // namespace sphere_overlap
}  // namespace common
}  // namespace score
}  // namespace tmol
