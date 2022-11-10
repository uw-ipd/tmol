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
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct compute_block_spheres {
  static void f(
      TView<Vec<Real, 3>, 2, D> coords,
      TView<Int, 2, D> pose_stack_block_coord_offset,
      TView<Int, 2, D> pose_stack_block_type,
      TView<Int, 1, D> block_type_n_atoms,
      TView<Real, 3, D> block_spheres) {
    LAUNCH_BOX_32;

    auto compute_spheres = ([=] TMOL_DEVICE_FUNC(int cta) {
      CTA_LAUNCH_T_PARAMS;

      int const n_poses = coords.size(0);
      int const max_n_pose_atoms = coords.size(1);
      int const max_n_blocks = pose_stack_block_type.size(1);

      int const pose_ind = cta / max_n_blocks;
      int const block_ind = cta % max_n_blocks;

      if (pose_ind >= n_poses) return;

      int const block_type = pose_stack_block_type[pose_ind][block_ind];

      if (block_type < 0) return;
      int const block_coord_offset =
          pose_stack_block_coord_offset[pose_ind][block_ind];
      int const n_atoms = block_type_n_atoms[block_type];
      Vec<Real, 3> local_coords(0, 0, 0);

      auto per_thread_com = ([&] TMOL_DEVICE_FUNC(int tid) {
        for (int i = tid; i < n_atoms; i += nt) {
          Vec<Real, 3> ci = coords[pose_ind][block_coord_offset + i];
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

      // __syncthreads();
      DeviceDispatch<D>::synchronize_workgroup();
      Vec<Real, 3> com =
          DeviceDispatch<D>::template shuffle_reduce_and_broadcast_in_workgroup<
              nt>(local_coords, mgpu::plus_t<Real>());

      Real d2max = 0;
      // Now find maximum distance
      auto per_thread_dist_to_com = ([&] TMOL_DEVICE_FUNC(int tid) {
        for (int i = tid; i < n_atoms; i += nt) {
          Vec<Real, 3> ci = coords[pose_ind][block_coord_offset + i];
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
        coords.size(0) * pose_stack_block_type.size(1), compute_spheres);
  }
};

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct detect_block_neighbors {
  static void f(
      TView<Vec<Real, 3>, 2, D> coords,
      TView<Int, 2, D> pose_stack_block_coord_offset,
      TView<Int, 2, D> pose_stack_block_type,
      TView<Int, 1, D> block_type_n_atoms,
      TView<Real, 3, D> block_spheres,
      TView<Int, 3, D> block_neighbors,
      Real reach) {
    LAUNCH_BOX_32;

    auto detect_neighbors = ([=] TMOL_DEVICE_FUNC(int ind) {
      int const n_poses = coords.size(0);
      int const max_n_pose_atoms = coords.size(1);
      int const max_n_blocks = pose_stack_block_type.size(1);
      int const n_block_types = block_type_n_atoms.size(0);

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

}  // namespace sphere_overlap
}  // namespace common
}  // namespace score
}  // namespace tmol
