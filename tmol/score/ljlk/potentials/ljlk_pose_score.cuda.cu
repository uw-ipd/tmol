#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/coordinate_load.cuh>
#include <tmol/score/common/count_pair.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/warp_segreduce.hh>
#include <tmol/score/common/warp_stride_reduce.hh>

#include <tmol/score/ljlk/potentials/lj.hh>
#include <tmol/score/ljlk/potentials/lk_isotropic.hh>
#include <tmol/score/ljlk/potentials/ljlk_pose_score.hh>

#include <chrono>

#include <tmol/score/common/forall_dispatch.cuda.impl.cuh>

#include <moderngpu/cta_reduce.hxx>
#include <moderngpu/transform.hxx>

// This file moves in more recent versions of Torch
#include <c10/cuda/CUDAStream.h>

// The maximum number of inter-residue chemical bonds
#define MAX_N_CONN 4
#define TILE_SIZE 32

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto LJLKPoseScoreDispatch<DeviceDispatch, D, Real, Int>::f(
    TView<Vec<Real, 3>, 2, D> coords,
    TView<Int, 2, D> pose_stack_block_coord_offset,
    TView<Int, 2, D> pose_stack_block_type,

    // dims: n-poses x max-n-blocks x max-n-blocks
    // Quick lookup: given the inds of two blocks, ask: what is the minimum
    // number of chemical bonds that separate any pair of atoms in those
    // blocks? If this minimum is greater than the crossover, then no further
    // logic for deciding whether two atoms in those blocks should have their
    // interaction energies calculated: all should. intentionally small to
    // (possibly) fit in constant cache
    TView<Int, 3, D> pose_stack_min_bond_separation,

    // dims: n-poses x max-n-blocks x max-n-blocks x
    // max-n-interblock-connections x max-n-interblock-connections
    TView<Int, 5, D> pose_stack_inter_block_bondsep,

    //////////////////////
    // Chemical properties
    // how many atoms for a given block
    // Dimsize n_block_types
    TView<Int, 1, D> block_type_n_atoms,

    TView<Int, 2, D> block_type_n_heavy_atoms_in_tile,
    TView<Int, 2, D> block_type_heavy_atoms_in_tile,

    // what are the atom types for these atoms
    // Dimsize: n_block_types x max_n_atoms
    TView<Int, 2, D> block_type_atom_types,

    // how many inter-block chemical bonds are there
    // Dimsize: n_block_types
    TView<Int, 1, D> block_type_n_interblock_bonds,

    // what atoms form the inter-block chemical bonds
    // Dimsize: n_block_types x max_n_interblock_bonds
    TView<Int, 2, D> block_type_atoms_forming_chemical_bonds,

    // what is the path distance between pairs of atoms in the block
    // Dimsize: n_block_types x max_n_atoms x max_n_atoms
    TView<Int, 3, D> block_type_path_distance,
    //////////////////////

    // LJ parameters
    TView<LJLKTypeParams<Real>, 1, D> type_params,
    TView<LJGlobalParams<Real>, 1, D> global_params

    ) -> std::tuple<TPack<Real, 2, D>, TPack<Vec<Real, 3>, 3, D>> {
  using tmol::score::common::accumulate;
  using Real3 = Vec<Real, 3>;

  int const n_poses = coords.size(0);
  int const max_n_pose_atoms = coords.size(1);
  int const max_n_blocks = pose_stack_block_type.size(1);
  int const max_n_block_atoms = block_type_atom_types.size(1);
  int const n_block_types = block_type_n_atoms.size(0);
  int const max_n_tiles = block_type_n_heavy_atoms_in_tile.size(2);
  int const max_n_interblock_bonds =
      block_type_atoms_forming_chemical_bonds.size(1);
  int64_t const n_atom_types = type_params.size(0);

  assert(max_n_interblock_bonds <= MAX_N_CONN);

  assert(pose_stack_block_type.size(0) == n_poses);

  assert(pose_stack_min_bond_separation.size(0) == n_poses);
  assert(pose_stack_min_bond_separation.size(1) == max_n_blocks);
  assert(pose_stack_min_bond_separation.size(2) == max_n_blocks);

  assert(pose_stack_inter_block_bondsep.size(0) == n_poses);
  assert(pose_stack_inter_block_bondsep.size(1) == max_n_blocks);
  assert(pose_stack_inter_block_bondsep.size(2) == max_n_blocks);
  assert(pose_stack_inter_block_bondsep.size(3) == max_n_interblock_bonds);
  assert(pose_stack_inter_block_bondsep.size(4) == max_n_interblock_bonds);

  assert(block_type_n_heavy_atoms_in_tile.size(0) == n_block_types);

  assert(block_type_heavy_atoms_in_tile.size(0) == n_block_types);
  assert(block_type_heavy_atoms_in_tile.size(1) == TILE_SIZE * max_n_tiles);

  assert(block_type_atom_types.size(0) == n_block_types);
  assert(block_type_atom_types.size(1) == max_n_block_atoms);

  assert(block_type_n_interblock_bonds.size(0) == n_block_types);

  assert(block_type_atoms_forming_chemical_bonds.size(0) == n_block_types);

  assert(block_type_path_distance.size(0) == n_block_types);
  assert(block_type_path_distance.size(1) == max_n_block_atoms);
  assert(block_type_path_distance.size(2) == max_n_block_atoms);

  auto output_t = TPack<Real, 2, D>::zeros({2, n_poses});
  auto output = output_t.view;

  auto dV_dcoords_t =
      TPack<Vec<Real, 3>, 3, D>::zeros({2, n_poses, max_n_pose_atoms});
  auto dV_dcoords = dV_dcoords_t.view;

  auto scratch_block_spheres_t =
      TPack<Real, 3, D>::zeros({n_poses, max_n_blocks, 4});
  auto scratch_block_spheres = scratch_block_spheres_t.view;

  auto scratch_block_neighbors_t =
      TPack<Int, 3, D>::zeros({n_poses, max_n_blocks, max_n_blocks});
  auto scratch_block_neighbors = scratch_block_neighbors_t.view;

  auto compute_block_sphere([=] MGPU_DEVICE(int tid, int cta) {
    // typedef typename launch_t::sm_ptx params_t;
    int const pose_ind = cta / max_n_blocks;
    int const block_ind = cta % max_n_blocks;
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

    // #ifdef __CUDACC__
    __syncthreads();
    auto g = cooperative_groups::coalesced_threads();
    for (int i = 0; i < 3; ++i) {
      com[i] = tmol::score::common::reduce_tile_shfl(
          g, local_coords[i], mgpu::plus_t<Real>());
      com[i] /= n_atoms;
      com[i] = g.shfl(com[i], 0);
    }
    // if (tid == 0) {
    //         printf("center of mass: %d %d (%f %f %f)\n", pose_ind, block_ind,
    // com[0],com[1],com[2]);
    // }
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

    // #endif  // __CUDACC__

    if (tid == 0) {
      scratch_block_spheres[pose_ind][block_ind][0] = com[0];
      scratch_block_spheres[pose_ind][block_ind][1] = com[1];
      scratch_block_spheres[pose_ind][block_ind][2] = com[2];
      scratch_block_spheres[pose_ind][block_ind][3] = dmax;
    }
  });

  auto detect_block_neighbor([=] MGPU_DEVICE(int ind) {
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
      sphere1[i] = scratch_block_spheres[pose_ind][block_ind1][i];
      sphere2[i] = scratch_block_spheres[pose_ind][block_ind2][i];
    }

    Real d2 =
        ((sphere1[0] - sphere2[0]) * (sphere1[0] - sphere2[0])
         + (sphere1[1] - sphere2[1]) * (sphere1[1] - sphere2[1])
         + (sphere1[2] - sphere2[2]) * (sphere1[2] - sphere2[2]));

    // warning: duplication of lennard-jones maximum distance threshold of 6A
    // hard coded here. Please fix!
    Real reach = sphere1[3] + sphere2[3] + 6.0;
    // printf("spheres %d %d %d distance %f vs reach %f; (%f %f %f, %f) and (%f
    // %f %f, %f)\n",
    //   pose_ind, block_ind1, block_ind2,
    //   sqrt(d2), reach,
    //   sphere1[0], sphere1[1], sphere1[2], sphere1[3],
    //   sphere2[0], sphere2[1], sphere2[2], sphere2[3]
    // );
    if (d2 < reach * reach) {
      scratch_block_neighbors[pose_ind][block_ind1][block_ind2] = 1;
    }
  });

  using namespace mgpu;
  typedef launch_box_t<
      arch_20_cta<32, 1>,
      arch_35_cta<32, 1>,
      arch_52_cta<32, 1>>
      launch_t;

  // between one alternate rotamer and its neighbors in the surrounding context
  auto score_inter_pairs_lj =
      ([=] MGPU_DEVICE(
           int pose_ind,
           int block_ind1,
           int block_ind2,
	   int block_coord_offset1,
	   int block_coord_offset2,
           int tid,
           int start_atom1,
           int start_atom2,
           Real *__restrict__ coords1,                  // shared
           Real *__restrict__ coords2,                  // shared
           LJLKTypeParams<Real> *__restrict__ params1,  // shared
           LJLKTypeParams<Real> *__restrict__ params2,  // shared
           int const max_important_bond_separation,
           int const min_separation,

           int const n_atoms1,
           int const n_atoms2,
           int const n_conn1,
           int const n_conn2,
           unsigned char const *__restrict__ path_dist1,  // shared
           unsigned char const *__restrict__ path_dist2,  // shared
           unsigned char const *__restrict__ conn_seps/*,
           Real3 *__restrict__ dlj_dcoords1,
           Real3 *__restrict__ dlj_dcoords2*/) {  // shared
        Real score_total = 0;
        Real3 coord1;
        Real3 coord2;

        int const n_remain1 = min(TILE_SIZE, n_atoms1 - start_atom1);
        int const n_remain2 = min(TILE_SIZE, n_atoms2 - start_atom2);

        int const n_pairs = n_remain1 * n_remain2;

        LJGlobalParams<Real> global_params_local = global_params[0];

        // for (int i = 0; i < 3; ++i) {
        //   dlj_dcoords1[tid][i] = 0;
        //   dlj_dcoords2[tid][i] = 0;
        // }

	// unsigned int active_mask =  __ballot_sync(0xFFFFFFFF, tid < n_pairs);
	
        for (int i = tid; i < n_pairs; i += blockDim.x) {
          auto g = cooperative_groups::coalesced_threads();

          int const atom_tile_ind1 = i / n_remain2;
          int const atom_tile_ind2 = i % n_remain2;
          for (int j = 0; j < 3; ++j) {
            coord1[j] = coords1[3 * atom_tile_ind1 + j];
            coord2[j] = coords2[3 * atom_tile_ind2 + j];
          }
          Real dist2 =
              ((coord1[0] - coord2[0]) * (coord1[0] - coord2[0])
               + (coord1[1] - coord2[1]) * (coord1[1] - coord2[1])
               + (coord1[2] - coord2[2]) * (coord1[2] - coord2[2]));

          // This square distance check cannot be performed if we are using the
          // segmented reduction logic later in this function, which requires
          // that all threads arrive at the warp_segreduce_shfl call.
          // if (dist2 > 36.0) {
          //   // DANGER -- maximum reach of LJ potential hard coded here in a
          //   // second place
          //   continue;
          // }
          auto dist_r = distance<Real>::V_dV(coord1, coord2);
          auto &dist = dist_r.V;
          auto &ddist_dat1 = dist_r.dV_dA;
          auto &ddist_dat2 = dist_r.dV_dB;

          int separation = min_separation;
          if (separation <= max_important_bond_separation) {
            separation =
                common::count_pair::CountPair<D, Int>::inter_block_separation<
                    TILE_SIZE>(
                    max_important_bond_separation,
                    atom_tile_ind1,
                    atom_tile_ind2,
                    n_conn1,
                    n_conn2,
                    path_dist1,
                    path_dist2,
                    conn_seps);
          }
          auto lj = lj_score<Real>::V_dV(
              dist,
              separation,
              params1[atom_tile_ind1].lj_params(),
              params2[atom_tile_ind2].lj_params(),
              global_params_local);
          score_total += lj.V;

          // Run a segmented reduction where the segment boundaries are the
          // threads that are the lowest-indexed threads for a particular atom.
          // All threads that are operating on the same atom will accumulate
          // their derivative vectors together and sum them down to the first
          // thread in the warp that's operating on that atom. Then that thread
          // will perform a single atomicAdd into the global derivative vector
          // tensor. For this logic to work, all threads in the warp must arrive
          // at this call. We therefore cannot perform the square distance check
          // earlier in this function. if (tid == 0) {
          //   printf("warp seg reduce %d %d %d %d\n", pose_ind, block_ind1,
          //   block_ind2, g.size());
          // }

	  // all threads accumulate derivatives for atom 1 to global memory
          Vec<Real, 3> lj_dxyz_at1 = lj.dV_ddist * ddist_dat1;
          for (int j = 0; j < 3; ++j) {
	      if (lj_dxyz_at1[j] != 0) {
              atomicAdd(
                  &dV_dcoords[0][pose_ind][block_coord_offset1 +
                             atom_tile_ind1 + start_atom1][j],
		  // &dlj_dcoords1[atom_tile_ind1][j],
		  lj_dxyz_at1[j]);
            }
          }

	  // all threads accumulate derivatives for atom 2 to shared mem
          Vec<Real, 3> lj_dxyz_at2 = lj.dV_ddist * ddist_dat2;
          for (int j = 0; j < 3; ++j) {
	    if (lj_dxyz_at2[j] != 0) {
              atomicAdd(
                  &dV_dcoords[0][pose_ind][block_coord_offset2 +
                             atom_tile_ind2 + start_atom2][j],
		  // &dlj_dcoords2[atom_tile_ind2][j],
		  lj_dxyz_at2[j]);
            }
          }

	  // Segmented reduction within a warp
	  // Vec<Real, 3> lj_dxyz_at1;
	  // // int const n_repeats = 3;
	  // // for (int repeat = 0; repeat < n_repeats; ++repeat) {
	  //   lj_dxyz_at1 = common::WarpSegReduceShfl<Vec<Real,3>>::segreduce(
	  //     active_mask,
	  //     lj.dV_ddist * ddist_dat1,
	  //     atom_tile_ind2 == 0 || tid == 0,
	  //     mgpu::plus_t<Real>());
	  //   //}
          // if (atom_tile_ind2 == 0 || tid == 0) {
          //   for (int j = 0; j < 3; ++j) {
          //     if (lj_dxyz_at1[j] != 0) {
          //       atomicAdd(&dV_dcoords[0][pose_ind][block_ind1][atom_tile_ind1 + start_atom1][j],
          //         lj_dxyz_at1[j]
          //       );
          //     }
          //   }
          // }

          //
          //
          // Vec<Real, 3> lj_dxyz_at2 = common::WarpStrideReduceShfl<Vec<Real,
          // 3>>::stride_reduce(
          //   g, lj.dV_ddist * ddist_dat2,
          //   n_remain2,
          //   mgpu::plus_t<Real>());
          // if (tid < n_remain2) {
          //   for (int j = 0; j < 3; ++j) {
          //     if (lj_dxyz_at2[j] != 0) {
          //       // atomicAdd(
          //       //   &dV_dcoords[0][pose_ind][block_ind2][atom_tile_ind2 +
          //       start_atom2][j],
          //       //   lj_dxyz_at2[j]
          //       // );
          //     }
          //   }
          // }

          // OK! we'll go ahead and accumulate the derivatives:
          // For each atom index, reduce within the warp, then
          // perform a single (atomic) add if the reduction was
          // non-zero per atom index.
          // This feels expensive!
          // accumulate<D, Vec<Real, 3>>::add_one_dst(
          //     dlj_dcoords1, atom_tile_ind1, lj.dV_ddist * ddist_dat1);
          //
          // accumulate<D, Vec<Real, 3>>::add_one_dst(
          //     dlj_dcoords2, atom_tile_ind2, lj.dV_ddist * ddist_dat2);

	  // Will I be active in the next iteration?
	  // active_mask =  __ballot_sync(active_mask, i + blockDim.x < n_pairs);

	}
	// write derivatives to global memory
	// if (tid < n_remain1) {
	//   for (int j = 0; j < 3; ++j) {
        //       atomicAdd(
        //         &dV_dcoords[0][pose_ind][block_ind1]
	// 	    [tid + start_atom1][j],
	// 	dlj_dcoords1[tid][j]);
	//   }
	// }
	// if (tid < n_remain2) {
	//   for (int j = 0; j < 3; ++j) {
        //       atomicAdd(
        //         &dV_dcoords[0][pose_ind][block_ind2]
	// 	    [tid + start_atom2][j],
	// 	dlj_dcoords2[tid][j]);
	//   }
	// }
	  

	return score_total;
      });

  // auto score_inter_pairs_lj_twice = ([=] MGPU_DEVICE(
  //                                  int pose_ind,
  //                                  int block_ind1,
  //                                  int block_ind2,
  //                                  int tid,
  //                                  int start_atom1,
  //                                  int start_atom2,
  //                                  Real *coords1,                  // shared
  //                                  Real *coords2,                  // shared
  //                                  LJLKTypeParams<Real> *params1,  // shared
  //                                  LJLKTypeParams<Real> *params2,  // shared
  //                                  int const max_important_bond_separation,
  //                                  int const min_separation,
  //
  //                                  int const n_atoms1,
  //                                  int const n_atoms2,
  //                                  int const n_conn1,
  //                                  int const n_conn2,
  //                                  unsigned char const *path_dist1,  //
  //                                  shared unsigned char const *path_dist2, //
  //                                  shared unsigned char const *conn_seps/*,
  //                                  Real3 *dlj_dcoords1,
  //                                  Real3 *dlj_dcoords2*/) {  // shared
  //
  //   Real score_total = 0;
  //   Real3 coord1;
  //   Real3 coord2;
  //
  //   int const n_remain1 = min(TILE_SIZE, n_atoms1 - start_atom1);
  //   int const n_remain2 = min(TILE_SIZE, n_atoms2 - start_atom2);
  //
  //   //int const n_pairs = n_remain1 * n_remain2;
  //
  //   LJGlobalParams<Real> global_params_local = global_params[0];
  //
  //   int const atom_tile_ind1 = tid;
  //   Vec<Real, 3> at1_dscore_dxyz;
  //   at1_dscore_dxyz.setZero();
  //
  //   if (atom_tile_ind1 < n_remain1) {
  //     auto atom_params1 = params1[atom_tile_ind1].lj_params();
  //
  //     // Calculate scores/derivatives for atom 1
  //     for (int i = 0; i < 3; ++i) {
  //       coord1[i] = coords1[3 * atom_tile_ind1 + i];
  //     }
  //
  //     for (int i = 0; i < n_remain2; ++i) {
  //       for (int j = 0; j < 3; ++j) {
  //         coord2[j] = coords2[3 * i + j];
  //       }
  //
  //       Real dist2 =
  //           ((coord1[0] - coord2[0]) * (coord1[0] - coord2[0])
  //            + (coord1[1] - coord2[1]) * (coord1[1] - coord2[1])
  //            + (coord1[2] - coord2[2]) * (coord1[2] - coord2[2]));
  //       auto dist_r = distance<Real>::V_dV(coord1, coord2);
  //       auto &dist = dist_r.V;
  //       auto &ddist_dat1 = dist_r.dV_dA;
  //       auto &ddist_dat2 = dist_r.dV_dB;
  //
  //       int separation = min_separation;
  //       if (separation <= max_important_bond_separation) {
  //         separation =
  //             common::count_pair::CountPair<D, Int>::inter_block_separation<
  //                 TILE_SIZE>(
  //                 max_important_bond_separation,
  //                 atom_tile_ind1,
  //                 i,
  //                 n_conn1,
  //                 n_conn2,
  //                 path_dist1,
  //                 path_dist2,
  //                 conn_seps);
  //       }
  //       auto lj = lj_score<Real>::V_dV(
  //           dist,
  //           separation,
  //           atom_params1,
  //           params2[i].lj_params(),
  //           global_params_local);
  //       score_total += lj.V;
  //       at1_dscore_dxyz += lj.dV_ddist * ddist_dat1;
  //     } // for each atom in residue 2
  //
  //     // Now accumulate the derivatives for atom 1
  //     for (int i = 0; i < 3; ++i) {
  // 	if (at1_dscore_dxyz[i] != 0) {
  // 	  atomicAdd(
  // 	    &dV_dcoords[0][pose_ind][block_ind1][atom_tile_ind1 +
  // start_atom1][i], 	    at1_dscore_dxyz[i]
  // 	  );
  // 	}
  //     }
  //   } // if atom_tile_ind1 < n_atoms1
  //
  //
  //
  //   // Now go back and perform the calculations again, but this time for
  //   atom2's
  //   // benefit.
  //   int const atom_tile_ind2 = tid;
  //   Vec<Real, 3> at2_dscore_dxyz;
  //   at2_dscore_dxyz.setZero();
  //
  //   if (atom_tile_ind2 < n_remain2) {
  //     auto atom_params2 = params2[atom_tile_ind2].lj_params();
  //     // Calculate scores/derivatives for atom 2
  //     for (int i = 0; i < 3; ++i) {
  //       coord2[i] = coords2[3 * atom_tile_ind2 + i];
  //     }
  //
  //     for (int i = 0; i < n_remain1; ++i) {
  //       for (int j = 0; j < 3; ++j) {
  //         coord1[j] = coords1[3 * i + j];
  //       }
  //
  //       Real dist2 =
  //           ((coord1[0] - coord2[0]) * (coord1[0] - coord2[0])
  //            + (coord1[1] - coord2[1]) * (coord1[1] - coord2[1])
  //            + (coord1[2] - coord2[2]) * (coord1[2] - coord2[2]));
  //       auto dist_r = distance<Real>::V_dV(coord1, coord2);
  //       auto &dist = dist_r.V;
  //       auto &ddist_dat1 = dist_r.dV_dA;
  //       auto &ddist_dat2 = dist_r.dV_dB;
  //
  //       int separation = min_separation;
  //       if (separation <= max_important_bond_separation) {
  //         separation =
  //             common::count_pair::CountPair<D, Int>::inter_block_separation<
  //                 TILE_SIZE>(
  //                 max_important_bond_separation,
  //                 i,
  //                 atom_tile_ind2,
  //                 n_conn1,
  //                 n_conn2,
  //                 path_dist1,
  //                 path_dist2,
  //                 conn_seps);
  //       }
  //       auto lj = lj_score<Real>::V_dV(
  //           dist,
  //           separation,
  //           params1[i].lj_params(),
  //           atom_params2,
  //           global_params_local);
  //
  //       at2_dscore_dxyz += lj.dV_ddist * ddist_dat2;
  //     } // for each atom in residue 2
  //
  //     // Now accumulate the derivatives for atom 2
  //     for (int i = 0; i < 3; ++i) {
  // 	if (at2_dscore_dxyz[i] != 0) {
  // 	  atomicAdd(
  // 	    &dV_dcoords[0][pose_ind][block_ind2][atom_tile_ind2 +
  // start_atom2][i], 	    at2_dscore_dxyz[i]
  // 	  );
  // 	}
  //     }
  //   } // if atom_tile_ind1 < n_atoms1
  //
  //   return score_total;
  // });

  auto score_inter_pairs_lk = ([=] MGPU_DEVICE(
      int pose_ind,
      int block_ind1,
      int block_ind2,
	   int block_coord_offset1,
	   int block_coord_offset2,
                                   int tid,
                                   int start_atom1,
                                   int start_atom2,
                                   int n_heavy1,
                                   int n_heavy2,
                                   Real *coords1,                     // shared
                                   Real *coords2,                     // shared
                                   LJLKTypeParams<Real> *params1,     // shared
                                   LJLKTypeParams<Real> *params2,     // shared
                                   unsigned char const *heavy_inds1,  // shared
                                   unsigned char const *heavy_inds2,  // shared
                                   int const max_important_bond_separation,
                                   int const min_separation,
                                   int const n_atoms1,
                                   int const n_atoms2,
                                   int const n_conn1,
                                   int const n_conn2,
                                   unsigned char const *path_dist1,  // shared
                                   unsigned char const *path_dist2,  // shared
                                   unsigned char const *conn_seps/*,
                                   Real3 *dlk_dcoords1,
                                   Real3 *dlk_dcoords2*/) {
    Real score_total = 0;

    Real3 coord1;
    Real3 coord2;

    int const n_pairs = n_heavy1 * n_heavy2;

    LJGlobalParams<Real> global_params_local = global_params[0];

    for (int i = tid; i < n_pairs; i += blockDim.x) {
      //auto g = cooperative_groups::coalesced_threads();

      int const atom_heavy_tile_ind1 = i / n_heavy2;
      int const atom_heavy_tile_ind2 = i % n_heavy2;
      int const atom_tile_ind1 = heavy_inds1[atom_heavy_tile_ind1];
      int const atom_tile_ind2 = heavy_inds2[atom_heavy_tile_ind2];
      // int const atom_ind1 = atom_tile_ind1 + start_atom1;
      // int const atom_ind2 = atom_tile_ind2 + start_atom2;

      // Debugging purposes only
      // if (atom_tile_ind1 < 0 || atom_tile_ind2 < 0
      //     || atom_tile_ind1 >= TILE_SIZE || atom_tile_ind2 >= TILE_SIZE) {
      //   printf(
      //       "bad atom index in lk-inter %d %d\n",
      //       atom_tile_ind1,
      //       atom_tile_ind2);
      //   continue;
      // }

      for (int j = 0; j < 3; ++j) {
        coord1[j] = coords1[3 * atom_tile_ind1 + j];
        coord2[j] = coords2[3 * atom_tile_ind2 + j];
      }
      Real dist2 =
          ((coord1[0] - coord2[0]) * (coord1[0] - coord2[0])
           + (coord1[1] - coord2[1]) * (coord1[1] - coord2[1])
           + (coord1[2] - coord2[2]) * (coord1[2] - coord2[2]));
      // if (dist2 > 36.0) {
      //   // DANGER -- maximum reach of LK potential hard coded here in a
      //   // second place
      //   continue;
      // }
      auto dist_r = distance<Real>::V_dV(coord1, coord2);
      auto &dist = dist_r.V;
      auto &ddist_dat1 = dist_r.dV_dA;
      auto &ddist_dat2 = dist_r.dV_dB;

      int separation = min_separation;
      if (separation <= max_important_bond_separation) {
        separation =
            common::count_pair::CountPair<D, Int>::inter_block_separation<
                TILE_SIZE>(
                max_important_bond_separation,
                atom_tile_ind1,
                atom_tile_ind2,
                n_conn1,
                n_conn2,
                path_dist1,
                path_dist2,
                conn_seps);
      }
      auto lk = lk_isotropic_score<Real>::V_dV(
          dist,
          separation,
          params1[atom_tile_ind1].lk_params(),
          params2[atom_tile_ind2].lk_params(),
          global_params_local);
      score_total += lk.V;

      Vec<Real,3> lk_dxyz_at1 = lk.dV_ddist * ddist_dat1;
      for (int j = 0; j < 3; ++j) {
	if (lk_dxyz_at1[j] != 0) {
	  atomicAdd(
	    &dV_dcoords[1][pose_ind][block_coord_offset1 + atom_tile_ind1 + start_atom1][j],
	    lk_dxyz_at1[j]);
	}
      }

      Vec<Real,3> lk_dxyz_at2 = lk.dV_ddist * ddist_dat2;
      for (int j = 0; j < 3; ++j) {
	if (lk_dxyz_at2[j] != 0) {
	  atomicAdd(
	    &dV_dcoords[1][pose_ind][block_coord_offset2 + atom_tile_ind2 + start_atom2][j],
	    lk_dxyz_at2[j]);
	}
      }



      // Run a segmented reduction where the segment boundaries are the
      // threads that are the lowest-indexed threads for a particular atom.
      // All threads that are operating on the same atom will accumulate their
      // derivative vectors together and sum them down to the first thread
      // in the warp that's operating on that atom. Then that thread will perform
      // a single atomicAdd into the global derivative vector tensor. For
      // this logic to work, all threads in the warp must arrive at this call.
      // We therefore cannot perform the square distance check earlier in this
      // function.
      // if (tid == 0) {
      //   printf("warp seg reduce %d %d %d %d\n", pose_ind, block_ind1, block_ind2, g.size());
      // }
      // Vec<Real, 3> lk_dxyz_at1 = common::WarpSegReduceShfl<Vec<Real, 3>>::segreduce(
      //   g, lk.dV_ddist * ddist_dat1,
      //   atom_heavy_tile_ind2 == 0 || tid == 0,
      //   mgpu::plus_t<Real>());
      // if (atom_heavy_tile_ind2 == 0 || tid == 0) {
      //   for (int j = 0; j < 3; ++j) {
      //     if (lk_dxyz_at1[j] != 0) {
      //       atomicAdd(
      //         &dV_dcoords[1][pose_ind][block_ind1][atom_tile_ind1 + start_atom1][j],
      //         lk_dxyz_at1[j]
      //       );
      //     }
      //   }
      // }
      // 
      // 
      // Vec<Real, 3> lk_dxyz_at2 = common::WarpStrideReduceShfl<Vec<Real, 3>>::stride_reduce(
      //   g, lk.dV_ddist * ddist_dat2,
      //   n_heavy2,
      //   mgpu::plus_t<Real>());
      // if (tid < n_heavy2) {
      //   for (int j = 0; j < 3; ++j) {
      //     if (lk_dxyz_at2[j] != 0) {
      //       atomicAdd(
      //         &dV_dcoords[1][pose_ind][block_ind2][atom_tile_ind2 + start_atom2][j],
      //         lk_dxyz_at2[j]
      //       );
      //     }
      //   }
      // }
    }
    return score_total;
  });

  // between atoms within one block
  auto score_intra_pairs_lj = ([=] MGPU_DEVICE(
                                   int pose_ind,
                                   int block_ind,
                                   int block_coord_offset,
                                   int tid,
                                   int start_atom1,
                                   int start_atom2,
                                   Real *coords1,
                                   Real *coords2,
                                   LJLKTypeParams<Real> *params1,
                                   LJLKTypeParams<Real> *params2,
                                   int const max_important_bond_separation,
                                   int const block_type,
                                   int const n_atoms) {
    Real score_total = 0;
    Real3 coord1;
    Real3 coord2;

    int const n_remain1 = min(TILE_SIZE, n_atoms - start_atom1);
    int const n_remain2 = min(TILE_SIZE, n_atoms - start_atom2);

    int const n_pairs = n_remain1 * n_remain2;
    LJGlobalParams<Real> global_params_local = global_params[0];
    // Real lj_weight = lj_lk_weights[0];

    for (int i = tid; i < n_pairs; i += blockDim.x) {
      // auto g = cooperative_groups::coalesced_threads();

      int const atom_tile_ind1 = i / n_remain2;
      int const atom_tile_ind2 = i % n_remain2;
      int const atom_ind1 = atom_tile_ind1 + start_atom1;
      int const atom_ind2 = atom_tile_ind2 + start_atom2;

      if (atom_ind1 >= atom_ind2) {
        continue;
      }

      for (int j = 0; j < 3; ++j) {
        coord1[j] = coords1[3 * atom_tile_ind1 + j];
        coord2[j] = coords2[3 * atom_tile_ind2 + j];
      }

      // read path distances from global memory
      int const separation =
          block_type_path_distance[block_type][atom_ind1][atom_ind2];

      auto dist_r = distance<Real>::V_dV(coord1, coord2);
      auto &dist = dist_r.V;
      auto &ddist_dat1 = dist_r.dV_dA;
      auto &ddist_dat2 = dist_r.dV_dB;

      auto lj = lj_score<Real>::V_dV(
          dist,
          separation,
          params1[atom_tile_ind1].lj_params(),
          params2[atom_tile_ind2].lj_params(),
          global_params_local);

      score_total += lj.V;

      Vec<Real, 3> lj_dxyz_at1 = lj.dV_ddist * ddist_dat1;
      for (int j = 0; j < 3; ++j) {
        if (lj_dxyz_at1[j] != 0) {
          atomicAdd(
              &dV_dcoords[0][pose_ind][block_coord_offset + atom_ind1][j],
              lj_dxyz_at1[j]);
        }
      }

      Vec<Real, 3> lj_dxyz_at2 = lj.dV_ddist * ddist_dat2;
      for (int j = 0; j < 3; ++j) {
        if (lj_dxyz_at2[j] != 0) {
          atomicAdd(
              &dV_dcoords[0][pose_ind][block_coord_offset + atom_ind2][j],
              lj_dxyz_at2[j]);
        }
      }

      // Vec<Real, 3> lj_dxyz_at1 =
      //     common::WarpSegReduceShfl<Vec<Real, 3>>::segreduce(
      //         g,
      //         lj.dV_ddist * ddist_dat1,
      //         atom_tile_ind2 == 0 || tid == 0,
      //         mgpu::plus_t<Real>());
      // if (atom_tile_ind2 == 0 || tid == 0) {
      //   for (int j = 0; j < 3; ++j) {
      //     if (lj_dxyz_at1[j] != 0) {
      //       atomicAdd(
      //           &dV_dcoords[0][pose_ind][block_ind][atom_ind1][j],
      //           lj_dxyz_at1[j]);
      //     }
      //   }
      // }
      //
      // Vec<Real, 3> lj_dxyz_at2 =
      //     common::WarpStrideReduceShfl<Vec<Real, 3>>::stride_reduce(
      //         g, lj.dV_ddist * ddist_dat2, n_remain2, mgpu::plus_t<Real>());
      // if (tid < n_remain2) {
      //   for (int j = 0; j < 3; ++j) {
      //     if (lj_dxyz_at2[j] != 0) {
      //       atomicAdd(
      //           &dV_dcoords[0][pose_ind][block_ind][atom_ind2][j],
      //           lj_dxyz_at2[j]);
      //     }
      //   }
      // }

      // OK! we'll go ahead and accumulate the derivatives:
      // For each atom index, reduce within the warp, then
      // perform a single (atomic) add if the reduction was
      // non-zero per atom index.
      // This feels expensive!
      // accumulate<D, Vec<Real, 3>>::add_one_dst(
      //     dlj_dcoords1, atom_tile_ind1, lj.dV_ddist * ddist_dat1);
      //
      // accumulate<D, Vec<Real, 3>>::add_one_dst(
      //     dlj_dcoords2, atom_tile_ind2, lj.dV_ddist * ddist_dat2);
    }
    return score_total;
  });

  // between one atoms within an alternate rotamer
  auto score_intra_pairs_lk = ([=] MGPU_DEVICE(
      int pose_ind,
      int block_ind,
      int block_coord_offset,
                                   int tid,
                                   int start_atom1,
                                   int start_atom2,
                                   int n_heavy1,
                                   int n_heavy2,
                                   Real *coords1,
                                   Real *coords2,
                                   LJLKTypeParams<Real> *params1,
                                   LJLKTypeParams<Real> *params2,
                                   unsigned char const *heavy_inds1,
                                   unsigned char const *heavy_inds2,
                                   int const max_important_bond_separation,
                                   int const block_type,
                                   int const n_atoms/*,
                                   Real3 *dlk_dcoords1,
                                   Real3 *dlk_dcoords2*/) {
    Real score_total = 0;

    Real3 coord1;
    Real3 coord2;

    int const n_pairs = n_heavy1 * n_heavy2;
    LJGlobalParams<Real> global_params_local = global_params[0];
    // Real lk_weight = lj_lk_weights[1];

    for (int i = tid; i < n_pairs; i += blockDim.x) {
      //auto g = cooperative_groups::coalesced_threads();

      int const atom_heavy_tile_ind1 = i / n_heavy2;
      int const atom_heavy_tile_ind2 = i % n_heavy2;
      int const atom_tile_ind1 = heavy_inds1[atom_heavy_tile_ind1];
      int const atom_tile_ind2 = heavy_inds2[atom_heavy_tile_ind2];
      int const atom_ind1 = atom_tile_ind1 + start_atom1;
      int const atom_ind2 = atom_tile_ind2 + start_atom2;

      if (atom_ind1 >= atom_ind2) {
        continue;
      }

      for (int j = 0; j < 3; ++j) {
        coord1[j] = coords1[3 * atom_tile_ind1 + j];
        coord2[j] = coords2[3 * atom_tile_ind2 + j];
      }

      // read path distances from global memory
      int const separation =
          block_type_path_distance[block_type][atom_ind1][atom_ind2];

      // Real const dist = sqrt(
      //     (coord1[0] - coord2[0]) * (coord1[0] - coord2[0])
      //     + (coord1[1] - coord2[1]) * (coord1[1] - coord2[1])
      //     + (coord1[2] - coord2[2]) * (coord1[2] - coord2[2]));

      auto dist_r = distance<Real>::V_dV(coord1, coord2);
      auto &dist = dist_r.V;
      auto &ddist_dat1 = dist_r.dV_dA;
      auto &ddist_dat2 = dist_r.dV_dB;

      auto lk = lk_isotropic_score<Real>::V_dV(
          dist,
          separation,
          params1[atom_tile_ind1].lk_params(),
          params2[atom_tile_ind2].lk_params(),
          global_params_local);

      score_total += lk.V;

      Vec<Real,3> lk_dxyz_at1 = lk.dV_ddist * ddist_dat1;
      for (int j = 0; j < 3; ++j) {
	if (lk_dxyz_at1[j] != 0) {
	  atomicAdd(
	    &dV_dcoords[1][pose_ind][block_coord_offset + atom_ind1][j],
	    lk_dxyz_at1[j]);
	}
      }

      Vec<Real,3> lk_dxyz_at2 = lk.dV_ddist * ddist_dat2;
      for (int j = 0; j < 3; ++j) {
	if (lk_dxyz_at2[j] != 0) {
	  atomicAdd(
	    &dV_dcoords[1][pose_ind][block_coord_offset + atom_ind2][j],
	    lk_dxyz_at2[j]);
	}
      }

      // Vec<Real, 3> lk_dxyz_at1 = common::WarpSegReduceShfl<Vec<Real, 3>>::segreduce(
      //   g, lk.dV_ddist * ddist_dat1,
      //   atom_heavy_tile_ind2 == 0 || tid == 0,
      //   mgpu::plus_t<Real>());
      // if (atom_heavy_tile_ind2 == 0 || tid == 0) {
      //   for (int j = 0; j < 3; ++j) {
      //     if (lk_dxyz_at1[j] != 0) {
      //       atomicAdd(
      //         &dV_dcoords[1][pose_ind][block_ind][atom_ind1][j],
      //         lk_dxyz_at1[j]
      //       );
      //     }
      //   }
      // }
      // 
      // 
      // Vec<Real, 3> lk_dxyz_at2 = common::WarpStrideReduceShfl<Vec<Real, 3>>::stride_reduce(
      //   g, lk.dV_ddist * ddist_dat2,
      //   n_heavy2,
      //   mgpu::plus_t<Real>());
      // if (tid < n_heavy2) {
      //   for (int j = 0; j < 3; ++j) {
      //     if (lk_dxyz_at2[j] != 0) {
      //       atomicAdd(
      //         &dV_dcoords[1][pose_ind][block_ind][atom_ind2][j],
      //         lk_dxyz_at2[j]
      //       );
      //     }
      //   }
      // }


      // OK! we'll go ahead and accumulate the derivatives:
      // For each atom index, reduce within the warp, then
      // perform a single (atomic) add if the reduction was
      // non-zero per atom index.
      // This feels expensive!
      // accumulate<D, Vec<Real, 3>>::add_one_dst(
      //     dlk_dcoords1, atom_tile_ind1, lk.dV_ddist * ddist_dat1);
      //
      // accumulate<D, Vec<Real, 3>>::add_one_dst(
      //     dlk_dcoords2, atom_tile_ind2, lk.dV_ddist * ddist_dat2);
    }
    return score_total;
  });

  auto load_block_coords_and_params_into_shared =
      ([=] MGPU_DEVICE(
           int pose_ind,
           int block_coord_offset,
           int n_atoms_to_load,
           int block_type,
           int tid,
           int tile_ind,
           Real *shared_coords,
           LJLKTypeParams<Real> *params,
           unsigned char *heavy_inds) {
        mgpu::mem_to_shared<TILE_SIZE, 3>(
            reinterpret_cast<Real *>(
                &coords[pose_ind][block_coord_offset + TILE_SIZE * tile_ind]),
            tid,
            n_atoms_to_load * 3,
            shared_coords,
            false);

        if (tid < TILE_SIZE) {
          if (tid < n_atoms_to_load) {
            int const atid = TILE_SIZE * tile_ind + tid;
            int const attype = block_type_atom_types[block_type][atid];
            if (attype >= 0) {
              params[tid] = type_params[attype];
            }
            heavy_inds[tid] = block_type_heavy_atoms_in_tile[block_type][atid];
          }
        }
      });

  auto load_block_into_shared = ([=] MGPU_DEVICE(
                                     int pose_ind,
                                     int block_coord_offset,
                                     int n_atoms,
                                     int n_atoms_to_load,
                                     int block_type,
                                     int n_conn,
                                     int tid,
                                     int tile_ind,
                                     bool count_pair_striking_dist,
                                     unsigned char *conn_ats,
                                     Real *shared_coords,
                                     LJLKTypeParams<Real> *params,
                                     unsigned char *heavy_inds,
                                     unsigned char *path_dist  // to conn
                                 ) {
    load_block_coords_and_params_into_shared(
        pose_ind,
        block_coord_offset,
        n_atoms_to_load,
        block_type,
        tid,
        tile_ind,
        shared_coords,
        params,
        heavy_inds);
    if (tid < n_atoms_to_load && count_pair_striking_dist) {
      int const atid = TILE_SIZE * tile_ind + tid;
      for (int j = 0; j < n_conn; ++j) {
        unsigned char ij_path_dist =
            block_type_path_distance[block_type][conn_ats[j]][atid];
        path_dist[j * TILE_SIZE + tid] = ij_path_dist;
      }
    }
  });

  auto eval_energies = ([=] MGPU_DEVICE(int tid, int cta) {
    typedef typename launch_t::sm_ptx params_t;
    enum {
      nt = params_t::nt,
      vt = params_t::vt,
      vt0 = params_t::vt0,
      nv = nt * vt
    };
    typedef mgpu::cta_reduce_t<nt, Real> reduce_t;

    __shared__ union shared_mem_union {
      shared_mem_union() {}
      struct {
        Real coords1[TILE_SIZE * 3];  // 786 bytes for coords
        Real coords2[TILE_SIZE * 3];
        LJLKTypeParams<Real> params1[TILE_SIZE];  // 1536 bytes for params
        LJLKTypeParams<Real> params2[TILE_SIZE];
        unsigned char n_heavy1;
        unsigned char n_heavy2;
        unsigned char heavy_inds1[TILE_SIZE];
        unsigned char heavy_inds2[TILE_SIZE];
        unsigned char conn_ats1[MAX_N_CONN];  // 8 bytes
        unsigned char conn_ats2[MAX_N_CONN];
        unsigned char path_dist1[MAX_N_CONN * TILE_SIZE];  // 256 bytes
        unsigned char path_dist2[MAX_N_CONN * TILE_SIZE];
        unsigned char conn_seps[MAX_N_CONN * MAX_N_CONN];  // 64 bytes

        // temporary location for accumulating atom derivatives
        // Real3 dscore_dxyz1[TILE_SIZE];
        // Real3 dscore_dxyz2[TILE_SIZE];

        // Real3 dlk_dcoords1[TILE_SIZE];
        // Real3 dlk_dcoords2[TILE_SIZE];

      } m;

      typename reduce_t::storage_t reduce;

    } shared;

    Real total_lj = 0;
    Real total_lk = 0;

    int const pose_ind = cta / (max_n_blocks * max_n_blocks);
    int const block_ind_pair = cta % (max_n_blocks * max_n_blocks);
    int const block_ind1 = block_ind_pair / max_n_blocks;
    int const block_ind2 = block_ind_pair % max_n_blocks;
    if (block_ind1 > block_ind2) {
      return;
    }

    if (scratch_block_neighbors[pose_ind][block_ind1][block_ind2] == 0) {
      return;
    }

    int const max_important_bond_separation = 4;

    int const block_type1 = pose_stack_block_type[pose_ind][block_ind1];
    int const block_type2 = pose_stack_block_type[pose_ind][block_ind2];

    if (block_type1 < 0 || block_type2 < 0) {
      return;
    }

    int const n_atoms1 = block_type_n_atoms[block_type1];
    int const n_atoms2 = block_type_n_atoms[block_type2];
    int const block_coord_offset1 =
        pose_stack_block_coord_offset[pose_ind][block_ind1];
    int const block_coord_offset2 =
        pose_stack_block_coord_offset[pose_ind][block_ind2];

    if (block_ind1 != block_ind2) {
      // inter-residue energy evaluation

      int const n_conn1 = block_type_n_interblock_bonds[block_type1];
      int const n_conn2 = block_type_n_interblock_bonds[block_type2];
      int const min_sep =
          pose_stack_min_bond_separation[pose_ind][block_ind1][block_ind2];
      bool const count_pair_striking_dist =
          min_sep <= max_important_bond_separation;

      if (count_pair_striking_dist && tid < n_conn1) {
        shared.m.conn_ats1[tid] =
            block_type_atoms_forming_chemical_bonds[block_type1][tid];
      }
      if (count_pair_striking_dist && tid < n_conn2) {
        shared.m.conn_ats2[tid] =
            block_type_atoms_forming_chemical_bonds[block_type2][tid];
      }

      if (count_pair_striking_dist && tid < n_conn1 * n_conn2) {
        int conn1 = tid / n_conn2;
        int conn2 = tid % n_conn2;
        shared.m.conn_seps[tid] =
            pose_stack_inter_block_bondsep[pose_ind][block_ind1][block_ind2]
                                          [conn1][conn2];
      }

      // Tile the sets of TILE_SIZE atoms
      int const n_iterations1 = (n_atoms1 - 1) / TILE_SIZE + 1;
      int const n_iterations2 = (n_atoms2 - 1) / TILE_SIZE + 1;
      for (int i = 0; i < n_iterations1; ++i) {
        // make sure all threads have completed their work
        // from the previous iteration before we overwrite
        // the contents of shared memory, and, on our first
        // iteration, make sure that the conn_ats arrays
        // have been written to
        __syncthreads();

        int const i_n_atoms_to_load1 =
            max(0, min(Int(TILE_SIZE), Int((n_atoms1 - TILE_SIZE * i))));

        // Let's load coordinates and Lennard-Jones parameters for
        // TILE_SIZE atoms into shared memory

        if (tid == 0) {
          shared.m.n_heavy1 = block_type_n_heavy_atoms_in_tile[block_type1][i];
        }

        load_block_into_shared(
            pose_ind,
            block_coord_offset1,
            n_atoms1,
            i_n_atoms_to_load1,
            block_type1,
            n_conn1,
            tid,
            i,
            count_pair_striking_dist,
            shared.m.conn_ats1,
            shared.m.coords1,
            shared.m.params1,
            shared.m.heavy_inds1,
            shared.m.path_dist1);

        for (int j = 0; j < n_iterations2; ++j) {
          if (j != 0) {
            // make sure that all threads have finished energy
            // calculations from the previous iteration before we
            // overwrite shared memory
            __syncthreads();
          }
          if (tid == 0) {
            shared.m.n_heavy2 =
                block_type_n_heavy_atoms_in_tile[block_type2][j];
            // printf("n heavy other: %d %d %d\n", alt_block_ind,
            // neighb_block_ind, shared.m.union_vals.vals.n_heavy_other);
          }

          int j_n_atoms_to_load2 =
              min(Int(TILE_SIZE), Int((n_atoms2 - TILE_SIZE * j)));
          load_block_into_shared(
              pose_ind,
              block_coord_offset2,
              n_atoms2,
              j_n_atoms_to_load2,
              block_type2,
              n_conn2,
              tid,
              j,
              count_pair_striking_dist,
              shared.m.conn_ats2,
              shared.m.coords2,
              shared.m.params2,
              shared.m.heavy_inds2,
              shared.m.path_dist2);

          // zero the derivative arrays we will be summing into
          // if (tid < TILE_SIZE) {
          //   for (int ii = 0; ii < 3; ++ii) {
          //     shared.m.dlj_dcoords1[tid][ii] = 0;
          //     shared.m.dlj_dcoords2[tid][ii] = 0;
          //     shared.m.dlk_dcoords1[tid][ii] = 0;
          //     shared.m.dlk_dcoords2[tid][ii] = 0;
          //   }
          // }

          // make sure all shared memory writes have completed before we read
          // from it when calculating atom-pair energies.
          __syncthreads();
          int n_heavy1 = shared.m.n_heavy1;
          int n_heavy2 = shared.m.n_heavy2;

          total_lj += score_inter_pairs_lj(
              pose_ind,
              block_ind1,
              block_ind2,
	      block_coord_offset1,
	      block_coord_offset2,
              tid,
              i * TILE_SIZE,
              j * TILE_SIZE,
              shared.m.coords1,
              shared.m.coords2,
              shared.m.params1,
              shared.m.params2,
              max_important_bond_separation,
              min_sep,
              n_atoms1,
              n_atoms2,
              n_conn1,
              n_conn2,
              shared.m.path_dist1,
              shared.m.path_dist2,
              shared.m.conn_seps/*,
              shared.m.dscore_dxyz1,
              shared.m.dscore_dxyz2*/);

          total_lk += score_inter_pairs_lk(
              pose_ind,
              block_ind1,
              block_ind2,
              block_coord_offset1,
              block_coord_offset2,
              tid,
              i * TILE_SIZE,
              j * TILE_SIZE,
              n_heavy1,
              n_heavy2,
              shared.m.coords1,
              shared.m.coords2,
              shared.m.params1,
              shared.m.params2,
              shared.m.heavy_inds1,
              shared.m.heavy_inds2,
              max_important_bond_separation,
              min_sep,
              n_atoms1,
              n_atoms2,
              n_conn1,
              n_conn2,
              shared.m.path_dist1,
              shared.m.path_dist2,
              shared.m.conn_seps/*,
              shared.m.dlk_dcoords1,
              shared.m.dlk_dcoords2*/);

          // now perform atomic adds from shared memory into global
          // memory
          // if (tid < TILE_SIZE) {
          //   for (int ii = 0; ii < 3; ++ii) {
          //     if (shared.m.dlj_dcoords1[tid][ii] != 0) {
          //       atomicAdd(
          //           &dV_dcoords[0][pose_ind][block_ind1][i * TILE_SIZE + tid]
          //                      [ii],
          //           shared.m.dlj_dcoords1[tid][ii]);
          //     }
          //     if (shared.m.dlk_dcoords1[tid][ii] != 0) {
          //       atomicAdd(
          //           &dV_dcoords[1][pose_ind][block_ind1][i * TILE_SIZE + tid]
          //                      [ii],
          //           shared.m.dlk_dcoords1[tid][ii]);
          //     }
          //     if (shared.m.dlj_dcoords2[tid][ii] != 0) {
          //       atomicAdd(
          //           &dV_dcoords[0][pose_ind][block_ind2][j * TILE_SIZE + tid]
          //                      [ii],
          //           shared.m.dlj_dcoords2[tid][ii]);
          //     }
          //     if (shared.m.dlk_dcoords2[tid][ii] != 0) {
          //       atomicAdd(
          //           &dV_dcoords[1][pose_ind][block_ind2][j * TILE_SIZE + tid]
          //                      [ii],
          //           shared.m.dlk_dcoords2[tid][ii]);
          //     }
          //   }
          // }

        }  // for j
      }    // for i
    } else {
      // alt_block_ind == neighb_block_ind

      int const n_iterations = (n_atoms1 - 1) / TILE_SIZE + 1;

      for (int i = 0; i < n_iterations; ++i) {
        if (i != 0) {
          // make sure the calculations for the previous iteration
          // have completed before we overwrite the contents of
          // shared memory
          __syncthreads();
        }
        int const i_n_atoms_to_load1 =
            min(Int(TILE_SIZE), Int((n_atoms1 - TILE_SIZE * i)));

        if (tid == 0) {
          shared.m.n_heavy1 = block_type_n_heavy_atoms_in_tile[block_type1][i];
        }

        load_block_coords_and_params_into_shared(
            pose_ind,
            block_coord_offset1,
            i_n_atoms_to_load1,
            block_type1,
            tid,
            i,
            shared.m.coords1,
            shared.m.params1,
            shared.m.heavy_inds1);

        for (int j = i; j < n_iterations; ++j) {
          int const j_n_atoms_to_load2 =
              min(Int(TILE_SIZE), Int((n_atoms1 - TILE_SIZE * j)));

          if (j != i) {
            // make sure calculations from the previous iteration have
            // completed before we overwrite the contents of shared
            // memory
            __syncthreads();
          }
          if (j != i) {
            if (tid == 0) {
              shared.m.n_heavy2 =
                  block_type_n_heavy_atoms_in_tile[block_type1][j];
            }

            load_block_coords_and_params_into_shared(
                pose_ind,
                block_coord_offset2,
                j_n_atoms_to_load2,
                block_type1,
                tid,
                j,
                shared.m.coords2,
                shared.m.params2,
                shared.m.heavy_inds2);
          }

          // zero the derivative arrays we will be summing into
          // if (tid < TILE_SIZE) {
          //   for (int ii = 0; ii < 3; ++ii) {
          //     shared.m.dlj_dcoords1[tid][ii] = 0;
          //     shared.m.dlk_dcoords1[tid][ii] = 0;
          //     shared.m.dlj_dcoords2[tid][ii] = 0;
          //     shared.m.dlk_dcoords2[tid][ii] = 0;
          //   }
          // }

          // we are guaranteed to hit this syncthreads call; we must wait
          // here before reading from shared memory for the coordinates
          // in shared.coords_alt1 to be loaded, or if j != i, for the
          // coordinates in shared..coords2 to be loaded.
          __syncthreads();
          int const n_heavy1 = shared.m.n_heavy1;
          int const n_heavy2 = (i == j ? n_heavy1 : shared.m.n_heavy2);

          total_lj += score_intra_pairs_lj(
            pose_ind,
            block_ind1,
	    block_coord_offset1,
              tid,
              i * TILE_SIZE,
              j * TILE_SIZE,
              shared.m.coords1,
              (i == j ? shared.m.coords1 : shared.m.coords2),
              shared.m.params1,
              (i == j ? shared.m.params1 : shared.m.params2),
              max_important_bond_separation,
              block_type1,
              n_atoms1/*,
              shared.m.dlj_dcoords1,
              shared.m.dlj_dcoords2*/);

          total_lk += score_intra_pairs_lk(
            pose_ind,
            block_ind1,
	    block_coord_offset1,
              tid,
              i * TILE_SIZE,
              j * TILE_SIZE,
              n_heavy1,
              n_heavy2,
              shared.m.coords1,
              (i == j ? shared.m.coords1 : shared.m.coords2),
              shared.m.params1,
              (i == j ? shared.m.params1 : shared.m.params2),
              shared.m.heavy_inds1,
              (i == j ? shared.m.heavy_inds1 : shared.m.heavy_inds2),
              max_important_bond_separation,
              block_type1,
              n_atoms1/*,
              shared.m.dlk_dcoords1,
              shared.m.dlk_dcoords2*/);

          // now perform atomic adds from shared memory into global
          // memory
          // if (tid < TILE_SIZE) {
          //   if (i == j) {
          //     for (int ii = 0; ii < 3; ++ii) {
          //       shared.m.dlj_dcoords1[tid][ii] +=
          //           shared.m.dlj_dcoords2[tid][ii];
          //       shared.m.dlk_dcoords1[tid][ii] +=
          //           shared.m.dlk_dcoords2[tid][ii];
          //     }
          //   }
          //
          //   for (int ii = 0; ii < 3; ++ii) {
          //     if (shared.m.dlj_dcoords1[tid][ii] != 0) {
          //       atomicAdd(
          //           &dV_dcoords[0][pose_ind][block_ind1][i * TILE_SIZE + tid]
          //                      [ii],
          //           shared.m.dlj_dcoords1[tid][ii]);
          //     }
          //     if (shared.m.dlk_dcoords1[tid][ii] != 0) {
          //       atomicAdd(
          //           &dV_dcoords[1][pose_ind][block_ind1][i * TILE_SIZE + tid]
          //                      [ii],
          //           shared.m.dlk_dcoords1[tid][ii]);
          //     }
          //   }
          //   if (i != j) {
          //     for (int ii = 0; ii < 3; ++ii) {
          //       if (shared.m.dlj_dcoords2[tid][ii] != 0) {
          //         atomicAdd(
          //             &dV_dcoords[0][pose_ind][block_ind2][j * TILE_SIZE +
          //             tid]
          //                        [ii],
          //             shared.m.dlj_dcoords2[tid][ii]);
          //       }
          //       if (shared.m.dlk_dcoords2[tid][ii] != 0) {
          //         atomicAdd(
          //             &dV_dcoords[1][pose_ind][block_ind2][j * TILE_SIZE +
          //             tid]
          //                        [ii],
          //             shared.m.dlk_dcoords2[tid][ii]);
          //       }
          //     }
          //   }
          // }
        }  // for j
      }    // for i
    }      // else

    // Make sure all energy calculations are complete before we overwrite
    // the neighbor-residue data in the shared memory union
    __syncthreads();

    Real const cta_total_lj = reduce_t().reduce(
        tid, total_lj, shared.reduce, nt, mgpu::plus_t<Real>());

    Real const cta_total_lk = reduce_t().reduce(
        tid, total_lk, shared.reduce, nt, mgpu::plus_t<Real>());

    if (tid == 0) {
      atomicAdd(&output[0][pose_ind], cta_total_lj);
      atomicAdd(&output[1][pose_ind], cta_total_lk);
    }
  });

  at::cuda::CUDAStream wrapped_stream = at::cuda::getDefaultCUDAStream();
  mgpu::standard_context_t context(wrapped_stream.stream());
  int const n_block_pairs = n_poses * max_n_blocks * max_n_blocks;

  mgpu::cta_launch<launch_t>(
      compute_block_sphere, n_poses * max_n_blocks, context);
  mgpu::transform<launch_t>(detect_block_neighbor, n_block_pairs, context);
  mgpu::cta_launch<launch_t>(eval_energies, n_block_pairs, context);

  return {output_t, dV_dcoords_t};
}

template struct LJLKPoseScoreDispatch<
    ForallDispatch,
    tmol::Device::CUDA,
    float,
    int>;
template struct LJLKPoseScoreDispatch<
    ForallDispatch,
    tmol::Device::CUDA,
    double,
    int>;

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
