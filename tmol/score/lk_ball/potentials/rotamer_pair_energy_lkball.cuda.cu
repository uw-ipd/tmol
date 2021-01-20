#include <tmol/score/common/forall_dispatch.cuda.impl.cuh>
#include <tmol/score/lk_ball/potentials/rotamer_pair_energy_lkball.impl.hh>

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

template struct LKBallRPEDispatch<
    common::ForallDispatch,
    tmol::Device::CUDA,
    float,
    int,
    4>;
template struct LKBallRPEDispatch<
    common::ForallDispatch,
    tmol::Device::CUDA,
    double,
    int,
    4>;

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol

// #include <Eigen/Core>
// #include <Eigen/Geometry>
//
// #include <tmol/utility/tensor/TensorAccessor.h>
// #include <tmol/utility/tensor/TensorPack.h>
// #include <tmol/utility/tensor/TensorStruct.h>
// #include <tmol/utility/tensor/TensorUtil.h>
// #include <tmol/utility/nvtx.hh>
//
// #include <tmol/score/common/accumulate.hh>
// #include <tmol/score/common/coordinate_load.cuh>
// #include <tmol/score/common/count_pair.hh>
// #include <tmol/score/common/geom.hh>
// #include <tmol/score/common/tuple.hh>
//
// #include <tmol/score/ljlk/potentials/lj.hh>
// #include <tmol/score/ljlk/potentials/rotamer_pair_energy_lj.hh>
//
// #include <chrono>
//
// #include <tmol/score/common/forall_dispatch.cuda.impl.cuh>
//
// #include <moderngpu/cta_load_balance.hxx>
// #include <moderngpu/cta_reduce.hxx>
// #include <moderngpu/cta_scan.hxx>
// #include <moderngpu/cta_segreduce.hxx>
// #include <moderngpu/cta_segscan.hxx>
// #include <moderngpu/memory.hxx>
// #include <moderngpu/search.hxx>
// #include <moderngpu/transform.hxx>
//
// // This file moves in more recent versions of Torch
// #include <ATen/cuda/CUDAStream.h>
//
// // #include <tmol/score/ljlk/potentials/rotamer_pair_energy_lj.impl.hh>
//
// namespace tmol {
// namespace score {
// namespace ljlk {
// namespace potentials {
//
// template <typename Real, int N>
// using Vec = Eigen::Matrix<Real, N, 1>;
//
// template <
//     template <tmol::Device>
//     class DeviceDispatch,
//     tmol::Device D,
//     typename Real,
//     typename Int>
// auto LJRPEDispatch<DeviceDispatch, D, Real, Int>::f(
//     TView<Vec<Real, 3>, 3, D> context_coords,
//     TView<Int, 2, D> context_block_type,
//     TView<Vec<Real, 3>, 2, D> alternate_coords,
//     TView<Vec<Int, 3>, 1, D>
//         alternate_ids,  // 0 == context id; 1 == block id; 2 == block type
//
//     // which system does a given context belong to
//     TView<Int, 1, D> context_system_ids,
//
//     // dims: n-systems x max-n-blocks x max-n-blocks
//     // Quick lookup: given the inds of two blocks, ask: what is the minimum
//     // number of chemical bonds that separate any pair of atoms in those
//     blocks?
//     // If this minimum is greater than the crossover, then no further logic
//     for
//     // deciding whether two atoms in those blocks should have their
//     interaction
//     // energies calculated: all should. intentionally small to (possibly) fit
//     in
//     // constant cache
//     TView<Int, 3, D> system_min_bond_separation,
//
//     // dims: n-systems x max-n-blocks x max-n-blocks x
//     // max-n-interblock-connections x max-n-interblock-connections
//     TView<Int, 5, D> system_inter_block_bondsep,
//
//     // dims n-systems x max-n-blocks x max-n-neighbors
//     // -1 as the sentinel
//     TView<Int, 3, D> system_neighbor_list,
//
//     //////////////////////
//     // Chemical properties
//     // how many atoms for a given block
//     // Dimsize n_block_types
//     TView<Int, 1, D> block_type_n_atoms,
//
//     // what are the atom types for these atoms
//     // Dimsize: n_block_types x max_n_atoms
//     TView<Int, 2, D> block_type_atom_types,
//
//     // how many inter-block chemical bonds are there
//     // Dimsize: n_block_types
//     TView<Int, 1, D> block_type_n_interblock_bonds,
//
//     // what atoms form the inter-block chemical bonds
//     // Dimsize: n_block_types x max_n_interblock_bonds
//     TView<Int, 2, D> block_type_atoms_forming_chemical_bonds,
//
//     // what is the path distance between pairs of atoms in the block
//     // Dimsize: n_block_types x max_n_atoms x max_n_atoms
//     TView<Int, 3, D> block_type_path_distance,
//     //////////////////////
//
//     // LJ parameters
//     TView<LJTypeParams<Real>, 1, D> type_params,
//     TView<LJGlobalParams<Real>, 1, D> global_params,
//     TView<Real, 1, D> lj_lk_weights)
//     -> std::tuple<TPack<Real, 1, D>, TPack<int64_t, 1, D>> {
//   int const n_systems = system_min_bond_separation.size(0);
//   int const n_contexts = context_coords.size(0);
//   int64_t const n_alternate_blocks = alternate_coords.size(0);
//   int const max_n_blocks = context_coords.size(1);
//   int64_t const max_n_atoms = context_coords.size(2);
//   int const n_block_types = block_type_n_atoms.size(0);
//   int const max_n_interblock_bonds =
//       block_type_atoms_forming_chemical_bonds.size(1);
//   int64_t const max_n_neighbors = system_neighbor_list.size(2);
//
//   assert(alternate_coords.size(1) == max_n_atoms);
//   assert(alternate_ids.size(0) == n_alternate_blocks);
//   assert(context_coords.size(0) == context_block_type.size(0));
//   assert(context_system_ids.size(0) == n_contexts);
//
//   assert(system_min_bond_separation.size(1) == max_n_blocks);
//   assert(system_min_bond_separation.size(2) == max_n_blocks);
//
//   assert(system_inter_block_bondsep.size(0) == n_systems);
//   assert(system_inter_block_bondsep.size(1) == max_n_blocks);
//   assert(system_inter_block_bondsep.size(2) == max_n_blocks);
//   assert(system_inter_block_bondsep.size(3) == max_n_interblock_bonds);
//   assert(system_inter_block_bondsep.size(4) == max_n_interblock_bonds);
//   assert(system_neighbor_list.size(0) == n_systems);
//   assert(system_neighbor_list.size(1) == max_n_blocks);
//
//   assert(block_type_atom_types.size(0) == n_block_types);
//   assert(block_type_atom_types.size(1) == max_n_atoms);
//   assert(block_type_n_interblock_bonds.size(0) == n_block_types);
//   assert(block_type_atoms_forming_chemical_bonds.size(0) == n_block_types);
//   assert(block_type_path_distance.size(0) == n_block_types);
//   assert(block_type_path_distance.size(1) == max_n_atoms);
//   assert(block_type_path_distance.size(2) == max_n_atoms);
//
//   assert(lj_lk_weights.size(0) == 2);
//
//   // auto wcts = std::chrono::system_clock::now();
//   // clock_t start_time = clock();
//
//   // Allocate and zero the output tensors in a separate stream
//   at::cuda::CUDAStream wrapped_stream = at::cuda::getStreamFromPool();
//   setCurrentCUDAStream(wrapped_stream);
//
//   auto output_t = TPack<Real, 1, D>::zeros({n_alternate_blocks});
//   auto output = output_t.view;
//   auto count_t = TPack<int, 1, D>::zeros({1});
//   auto count = count_t.view;
//
//   // I'm not sure I want/need events for synchronization
//   auto event_t = TPack<int64_t, 1, D>::zeros({2});
//
//   using namespace mgpu;
//   typedef launch_box_t<
//       arch_20_cta<64, 1>,
//       arch_35_cta<64, 1>,
//       arch_52_cta<64, 1>>
//       launch_t;
//
//   // between one alternate rotamer and its neighbors in the surrounding
//   context auto score_inter_pairs = ([=] MGPU_DEVICE(
//                                 int tid,
//                                 int alt_start_atom,
//                                 int neighb_start_atom,
//                                 Real *alt_coords,
//                                 Real *neighb_coords,
//                                 LJTypeParams<Real> *alt_params,
//                                 LJTypeParams<Real> *neighb_params,
//                                 int const max_important_bond_separation,
//                                 int const alt_block_ind,
//                                 int const neighb_block_ind,
//                                 int const alt_block_type,
//                                 int const neighb_block_type,
//
//                                 int min_separation,
//                                 TensorAccessor<Int, 4, D>
//                                 inter_block_bondsep,
//
//                                 int const alt_n_atoms,
//                                 int const neighb_n_atoms) {
//     Real score_total = 0;
//     Real coord1[3];
//     Real coord2[3];
//
//     int const alt_remain = min(32, alt_n_atoms - alt_start_atom);
//     int const neighb_remain = min(32, neighb_n_atoms - neighb_start_atom);
//
//     int const n_pairs = alt_remain * neighb_remain;
//
//     LJGlobalParams<Real> global_params_local = global_params[0];
//     Real lj_weight = lj_lk_weights[0];
//
//     for (int i = tid; i < n_pairs; i += blockDim.x) {
//       int const alt_atom_tile_ind = i / neighb_remain;
//       int const neighb_atom_tile_ind = i % neighb_remain;
//       int const alt_atom_ind = alt_atom_tile_ind + alt_start_atom;
//       int const neighb_atom_ind = neighb_atom_tile_ind + neighb_start_atom;
//       for (int j = 0; j < 3; ++j) {
//         coord1[j] = alt_coords[3 * alt_atom_tile_ind + j];
//         coord2[j] = neighb_coords[3 * neighb_atom_tile_ind + j];
//       }
//       // int const atom_1_type = alt_atom_type[alt_atom_tile_ind];
//       // int const atom_2_type = neighb_atom_type[neighb_atom_tile_ind];
//
//       int const separation =
//           min_separation > max_important_bond_separation
//               ? max_important_bond_separation
//               : common::count_pair::CountPair<D,
//               Int>::inter_block_separation(
//                     max_important_bond_separation,
//                     alt_block_ind,
//                     neighb_block_ind,
//                     alt_block_type,
//                     neighb_block_type,
//                     alt_atom_ind,
//                     neighb_atom_ind,
//                     inter_block_bondsep,
//                     block_type_n_interblock_bonds,
//                     block_type_atoms_forming_chemical_bonds,
//                     block_type_path_distance);
//
//       Real dist = sqrt(
//           (coord1[0] - coord2[0]) * (coord1[0] - coord2[0])
//           + (coord1[1] - coord2[1]) * (coord1[1] - coord2[1])
//           + (coord1[2] - coord2[2]) * (coord1[2] - coord2[2]));
//
//       Real lj = lj_score<Real>::V(
//           dist,
//           separation,
//           alt_params[alt_atom_tile_ind],
//           neighb_params[neighb_atom_tile_ind],
//           global_params_local);
//       lj *= lj_weight;
//       // if ( lj != 0 ) {
//       //   printf("cuda %d %d %6.3f %6.3f %6.3f vs %6.3f %6.3f %6.3f e=
//       //   %8.4f\n",
//       //     alt_atom_ind, neighb_atom_ind,
//       //     coord1[0], coord1[1], coord1[2],
//       //     coord2[0], coord2[1], coord2[2],
//       //     lj
//       //   );
//       // }
//
//       score_total += lj;
//     }
//     return score_total;
//   });
//
//   // between one atoms within an alternate rotamer
//   auto score_intra_pairs = ([=] MGPU_DEVICE(
//                                 int tid,
//                                 int start_atom1,
//                                 int start_atom2,
//                                 Real *coords1,
//                                 Real *coords2,
//                                 LJTypeParams<Real> *params1,
//                                 LJTypeParams<Real> *params2,
//                                 int const max_important_bond_separation,
//                                 int const block_type,
//                                 int const n_atoms) {
//     Real score_total = 0;
//     Real coord1[3];
//     Real coord2[3];
//
//     int const remain1 = min(32, n_atoms - start_atom1);
//     int const remain2 = min(32, n_atoms - start_atom2);
//
//     int const n_pairs = remain1 * remain2;
//
//     LJGlobalParams<Real> global_params_local = global_params[0];
//     Real lj_weight = lj_lk_weights[0];
//
//     for (int i = tid; i < n_pairs; i += blockDim.x) {
//       int const atom_ind_1_local = i / remain2;
//       int const atom_ind_2_local = i % remain2;
//       int const atom_ind_1 = atom_ind_1_local + start_atom1;
//       int const atom_ind_2 = atom_ind_2_local + start_atom2;
//       if (atom_ind_1 >= atom_ind_2) {
//         continue;
//       }
//
//       for (int j = 0; j < 3; ++j) {
//         coord1[j] = coords1[3 * atom_ind_1_local + j];
//         coord2[j] = coords2[3 * atom_ind_2_local + j];
//       }
//       // int const atom_1_type = atom_type1[atom_ind_1_local];
//       // int const atom_2_type = atom_type2[atom_ind_2_local];
//
//       int const separation =
//           block_type_path_distance[block_type][atom_ind_1][atom_ind_2];
//
//       Real const dist = sqrt(
//           (coord1[0] - coord2[0]) * (coord1[0] - coord2[0])
//           + (coord1[1] - coord2[1]) * (coord1[1] - coord2[1])
//           + (coord1[2] - coord2[2]) * (coord1[2] - coord2[2]));
//
//       Real lj = lj_score<Real>::V(
//           dist,
//           separation,
//           params1[atom_ind_1_local],
//           params2[atom_ind_2_local],
//           global_params_local);
//       lj *= lj_lk_weights[0];
//       score_total += lj;
//     }
//     return score_total;
//   });
//
//   auto eval_energies = ([=] MGPU_DEVICE(int tid, int cta) {
//     typedef typename launch_t::sm_ptx params_t;
//     enum {
//       nt = params_t::nt,
//       vt = params_t::vt,
//       vt0 = params_t::vt0,
//       nv = nt * vt
//     };
//     typedef mgpu::cta_reduce_t<nt, Real> reduce_t;
//
//     __shared__ union {
//       struct {
//         Real coords1[32 * 3];
//         Real coords2[32 * 3];
//         LJTypeParams<Real> params1[32];
//         LJTypeParams<Real> params2[32];
//         Int min_separation;
//       } vals;
//       typename reduce_t::storage_t reduce;
//     } shared;
//
//     Real *coords1 = shared.vals.coords1;
//     Real *coords2 = shared.vals.coords2;
//     LJTypeParams<Real> *params1 = shared.vals.params1;
//     LJTypeParams<Real> *params2 = shared.vals.params2;
//
//     Real cta_totalE = 0;
//
//     for (int iteration = 0; iteration < vt; ++iteration) {
//       Real totalE = 0;
//
//       int alt_ind = (vt * cta + iteration) / max_n_neighbors;
//
//       if (alt_ind >= n_alternate_blocks) {
//         return;
//       }
//
//       int neighb_ind = (vt * cta + iteration) % max_n_neighbors;
//
//       int const max_important_bond_separation = 4;
//       int const alt_context = alternate_ids[alt_ind][0];
//       if (alt_context == -1) {
//         return;
//       }
//
//       int const alt_block_ind = alternate_ids[alt_ind][1];
//       int const alt_block_type = alternate_ids[alt_ind][2];
//       int const system = context_system_ids[alt_context];
//
//       int const neighb_block_ind =
//           system_neighbor_list[system][alt_block_ind][neighb_ind];
//       if (neighb_block_ind == -1) {
//         return;
//       }
//
//       int const n_iterations = (max_n_atoms - 1) / 32 + 1;
//
//       if (alt_block_ind != neighb_block_ind) {
//         int const neighb_block_type =
//             context_block_type[alt_context][neighb_block_ind];
//         int const alt_n_atoms = block_type_n_atoms[alt_block_type];
//         int const neighb_n_atoms = block_type_n_atoms[neighb_block_type];
//
//         if (tid == 0) {
//           int const min_sep =
//           system_min_bond_separation[system][alt_block_ind]
//                                                         [neighb_block_ind];
//           shared.vals.min_separation = min_sep;
//         }
//         __syncthreads();
//         int const min_sep = shared.vals.min_separation;
//
//         // Tile the sets of 32 atoms
//         for (int i = 0; i < n_iterations; ++i) {
//           if (i != 0) {
//             // make sure all threads have completed their work
//             // from the previous iteration before we overwrite
//             // the contents of shared memory
//             __syncthreads();
//           }
//
//           // Let's load coordinates and Lennard-Jones parameters for
//           // 32 atoms into shared memory
//
//           if (tid < 32) {
//             // coalesced read of atom coordinate data
//             common::coalesced_read_of_32_coords_into_shared(
//                 alternate_coords[alt_ind], i * 32, coords1, tid);
//
//             // load the Lennard-Jones parameters for these 32 atoms
//             if (32 * i + tid < max_n_atoms) {
//               int const atid = 32 * i + tid;
//               int const attype = block_type_atom_types[alt_block_type][atid];
//               if (attype >= 0) {
//                 params1[tid] = type_params[attype];
//               }
//             }
//           }
//
//           for (int j = 0; j < n_iterations; ++j) {
//             if (j != 0) {
//               // make sure that all threads have finished energy
//               // calculations from the previous iteration
//               __syncthreads();
//             }
//             if (tid < 32) {
//               // Coalesced read of atom coordinate data
//               common::coalesced_read_of_32_coords_into_shared(
//                   context_coords[alt_context][neighb_block_ind],
//                   j * 32,
//                   coords2,
//                   tid);
//
//               // load the Lennard-Jones parameters for these 32 atoms
//               if (32 * j + tid < max_n_atoms) {
//                 int const atid = 32 * j + tid;
//                 int const attype =
//                     block_type_atom_types[neighb_block_type][atid];
//                 if (attype >= 0) {
//                   params2[tid] = type_params[attype];
//                 }
//               }
//             }
//
//             // make sure shared-memory loading has completed before we
//             proceed
//             // into energy calculations
//             __syncthreads();
//
//             // Now we will calculate the 32x32 atom pair energies
//             totalE += score_inter_pairs(
//                 tid,
//                 i * 32,
//                 j * 32,
//                 coords1,
//                 coords2,
//                 params1,
//                 params2,
//                 max_important_bond_separation,
//                 alt_block_ind,
//                 neighb_block_ind,
//                 alt_block_type,
//                 neighb_block_type,
//                 min_sep,
//                 system_inter_block_bondsep[system],
//                 alt_n_atoms,
//                 neighb_n_atoms);
//           }  // for j
//         }    // for i
//       } else {
//         int const alt_n_atoms = block_type_n_atoms[alt_block_type];
//
//         for (int i = 0; i < n_iterations; ++i) {
//           if (i != 0) {
//             // make sure the calculations for the previous iteration
//             // have completed before we overwrite the contents of
//             // shared memory
//             __syncthreads();
//           }
//           if (tid < 32) {
//             // coalesced reads of coordinate data
//             common::coalesced_read_of_32_coords_into_shared(
//                 alternate_coords[alt_ind], i * 32, coords1, tid);
//             // for (int j = 0; j < 3; ++j) {
//             //   int j_ind = j * 32 + tid;
//             //   int local_atomind = j_ind / 3;
//             //   int atid = local_atomind + i * 32;
//             //   int dim = j_ind % 3;
//             //   if (atid < max_n_atoms) {
//             //     coords1[j_ind] = alternate_coords[alt_ind][atid][dim];
//             //   }
//             // }
//
//             // load Lennard-Jones parameters for the 32 atoms into shared
//             // memory
//             if (i * 32 + tid < max_n_atoms) {
//               int const atind = i * 32 + tid;
//               int const attype =
//               block_type_atom_types[alt_block_type][atind]; if (attype >= 0)
//               {
//                 params1[tid] = type_params[attype];
//               }
//             }
//           }
//           for (int j = i; j < n_iterations; ++j) {
//             if (j != i) {
//               // make sure calculations from the previous iteration have
//               // completed before we overwrite the contents of shared
//               // memory
//               __syncthreads();
//             }
//
//             if (j != i && tid < 32) {
//               // coalesced read of coordinate data
//               common::coalesced_read_of_32_coords_into_shared(
//                   alternate_coords[alt_ind], j * 32, coords2, tid);
//               // for (int k = 0; k < 3; ++k) {
//               //   int k_ind = k * 32 + tid;
//               //   int local_atomind = k_ind / 3;
//               //   int atid = local_atomind + j * 32;
//               //   int dim = k_ind % 3;
//               //   if (atid < max_n_atoms) {
//               //     coords2[k_ind] = alternate_coords[alt_ind][atid][dim];
//               //   }
//               // }
//               if (j * 32 + tid < max_n_atoms) {
//                 int const atind = j * 32 + tid;
//                 int const attype =
//                 block_type_atom_types[alt_block_type][atind]; if (attype >=
//                 0) {
//                   params2[tid] = type_params[attype];
//                 }
//               }
//             }
//             __syncthreads();
//             totalE += score_intra_pairs(
//                 tid,
//                 i * 32,
//                 j * 32,
//                 coords1,
//                 (i == j ? coords1 : coords2),
//                 params1,
//                 (i == j ? params1 : params2),
//                 max_important_bond_separation,
//                 alt_block_type,
//                 alt_n_atoms);
//           }  // for j
//         }    // for i
//       }      // else
//
//       __syncthreads();
//
//       cta_totalE += reduce_t().reduce(
//           tid, totalE, shared.reduce, nt, mgpu::plus_t<Real>());
//
//       if (tid == 0) {
//         // int next_alt_ind = (vt * cta + iteration + 1) / max_n_neighbors;
//
//         // if (cta_totalE != 0 && (iteration+1 == vt || next_alt_ind !=
//         // alt_ind)) {
//         atomicAdd(&output[alt_ind], cta_totalE);
//         cta_totalE = 0;
//         //}
//       }
//     }
//   });
//
//   mgpu::standard_context_t context(wrapped_stream.stream());
//   int const n_ctas =
//       (n_alternate_blocks * max_n_neighbors - 1) / launch_t::sm_ptx::vt + 1;
//   mgpu::cta_launch<launch_t>(eval_energies, n_ctas, context);
//
//   at::cuda::setCurrentCUDAStream(at::cuda::getDefaultCUDAStream());
//
// #ifdef __CUDACC__
//   // float first;
//   // cudaMemcpy(&first, &output[0], sizeof(float), cudaMemcpyDeviceToHost);
//   //
//   // clock_t stop_time = clock();
//   // std::chrono::duration<double> wctduration =
//   // (std::chrono::system_clock::now() - wcts);
//   //
//   // std::cout << n_systems << " " << n_contexts << " " <<n_alternate_blocks
//   <<
//   // " "; std::cout << n_alternate_blocks * max_n_neighbors * max_n_atoms *
//   // max_n_atoms << " "; std::cout << "runtime? " << ((double)stop_time -
//   // start_time) / CLOCKS_PER_SEC
//   //           << " wall time: " << wctduration.count() << " " << first
//   //           << std::endl;
// #endif
//   return {output_t, event_t};
// }
//
// template struct LJRPEDispatch<ForallDispatch, tmol::Device::CUDA, float,
// int>; template struct LJRPEDispatch<ForallDispatch, tmol::Device::CUDA,
// double, int>;
//
// }  // namespace potentials
// }  // namespace ljlk
// }  // namespace score
// }  // namespace tmol
//
