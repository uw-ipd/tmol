#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/warp_segreduce.hh>
#include <tmol/score/common/warp_stride_reduce.hh>

#include <moderngpu/context.hxx>

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <tmol::Device D, typename Real, typename Int>
__global__ void compute_block_spheres_kernel(
    TView<Vec<Real, 3>, 2, D> coords,
    TView<Int, 2, D> pose_stack_block_coord_offset,
    TView<Int, 2, D> pose_stack_block_type,
    TView<Int, 1, D> block_type_n_atoms,
    TView<Real, 3, D> block_spheres) {
  int const tid = threadIdx.x;
  int const cta = blockIdx.x;

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
  for (int i = tid; i < n_atoms; i += blockDim.x) {
    Vec<Real, 3> ci = coords[pose_ind][block_coord_offset + i];
    for (int j = 0; j < 3; ++j) {
      local_coords[j] += ci[j];
    }
  }

  // The center of mass
  Vec<Real, 3> com;
  Real dmax(0);

  __syncthreads();
  auto g = cooperative_groups::coalesced_threads();
  for (int i = 0; i < 3; ++i) {
    com[i] = tmol::score::common::reduce_tile_shfl(
        g, local_coords[i], mgpu::plus_t<Real>());
    com[i] /= n_atoms;
    com[i] = g.shfl(com[i], 0);
  }

  Real d2max = 0;
  // Now find maximum distance
  for (int i = tid; i < n_atoms; i += blockDim.x) {
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
  dmax =
      tmol::score::common::reduce_tile_shfl(g, dmax, mgpu::maximum_t<Real>());

  if (tid == 0) {
    block_spheres[pose_ind][block_ind][0] = com[0];
    block_spheres[pose_ind][block_ind][1] = com[1];
    block_spheres[pose_ind][block_ind][2] = com[2];
    block_spheres[pose_ind][block_ind][3] = dmax;
  }
}

template <tmol::Device D, typename Real, typename Int>
void __global__ detect_block_neighbors_kernel(
    TView<Vec<Real, 3>, 2, D> coords,
    TView<Int, 2, D> pose_stack_block_coord_offset,
    TView<Int, 2, D> pose_stack_block_type,
    TView<Int, 1, D> block_type_n_atoms,
    TView<Real, 3, D> block_spheres,
    TView<Int, 3, D> block_neighbors,
    Real reach) {
  int const tid = threadIdx.x;
  int const cta = blockIdx.x;

  int const n_poses = coords.size(0);
  int const max_n_pose_atoms = coords.size(1);
  int const max_n_blocks = pose_stack_block_type.size(1);
  int const n_block_types = block_type_n_atoms.size(0);
  int ind = cta * blockDim.x + tid;

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
}

template <tmol::Device D, typename Real, typename Int, typename LaunchType>
void launch_compute_block_spheres(
    TView<Vec<Real, 3>, 2, D> coords,
    TView<Int, 2, D> pose_stack_block_coord_offset,
    TView<Int, 2, D> pose_stack_block_type,
    TView<Int, 1, D> block_type_n_atoms,
    TView<Real, 3, D> block_spheres,
    mgpu::standard_context_t& context) {
  int const n_poses = coords.size(0);
  int const max_n_blocks = pose_stack_block_type.size(1);

  compute_block_spheres_kernel<<<
      n_poses * max_n_blocks,
      LaunchType::sm_ptx::nt,
      0,
      context.stream()>>>(
      coords,
      pose_stack_block_coord_offset,
      pose_stack_block_type,
      block_type_n_atoms,
      block_spheres);
}

template <tmol::Device D, typename Real, typename Int, typename LaunchType>
void launch_detect_block_neighbors(
    TView<Vec<Real, 3>, 2, D> coords,
    TView<Int, 2, D> pose_stack_block_coord_offset,
    TView<Int, 2, D> pose_stack_block_type,
    TView<Int, 1, D> block_type_n_atoms,
    TView<Real, 3, D> block_spheres,
    TView<Int, 3, D> block_neighbors,
    Real reach,
    mgpu::standard_context_t& context) {
  int const n_poses = coords.size(0);
  int const max_n_blocks = pose_stack_block_type.size(1);
  int const n_block_pairs = n_poses * max_n_blocks * max_n_blocks;

  int const n_ctas_detect_block_neighbors =
      (n_block_pairs - 1) / LaunchType::sm_ptx::nt + 1;

  detect_block_neighbors_kernel<<<
      n_poses * max_n_blocks,
      LaunchType::sm_ptx::nt,
      0,
      context.stream()>>>(
      coords,
      pose_stack_block_coord_offset,
      pose_stack_block_type,
      block_type_n_atoms,
      block_spheres,
      block_neighbors,
      reach);
}

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
