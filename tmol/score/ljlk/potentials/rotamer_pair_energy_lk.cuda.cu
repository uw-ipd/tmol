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

#include <tmol/score/ljlk/potentials/lk_isotropic.hh>
#include <tmol/score/ljlk/potentials/rotamer_pair_energy_lk.hh>

#include <tmol/pack/sim_anneal/compiled/annealer.hh>

#include <chrono>

#include <moderngpu/cta_load_balance.hxx>
#include <moderngpu/cta_reduce.hxx>
#include <moderngpu/cta_scan.hxx>
#include <moderngpu/cta_segreduce.hxx>
#include <moderngpu/cta_segscan.hxx>
#include <moderngpu/memory.hxx>
#include <moderngpu/search.hxx>
#include <moderngpu/transform.hxx>

#include <tmol/score/common/forall_dispatch.cuda.impl.cuh>
// #include <tmol/score/ljlk/potentials/rotamer_pair_energy_lk.impl.hh>

#include <c10/cuda/CUDAStream.h>

// The maximum number of inter-residue chemical bonds
#define MAX_N_CONN 4

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

#define TILE_SIZE 16

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto LKRPEDispatch<DeviceDispatch, D, Real, Int>::f(
    TView<Vec<Real, 3>, 3, D> context_coords,
    TView<Int, 2, D> context_block_type,
    TView<Vec<Real, 3>, 2, D> alternate_coords,
    TView<Vec<Int, 3>, 1, D>
        alternate_ids,  // 0 == context id; 1 == block id; 2 == block type

    // which system does a given context belong to
    TView<Int, 1, D> context_system_ids,

    // dims: n-systems x max-n-blocks x max-n-blocks
    // Quick lookup: given the inds of two blocks, ask: what is the minimum
    // number of chemical bonds that separate any pair of atoms in those blocks?
    // If this minimum is greater than the crossover, then no further logic for
    // deciding whether two atoms in those blocks should have their interaction
    // energies calculated: all should. intentionally small to (possibly) fit in
    // constant cache
    TView<Int, 3, D> system_min_bond_separation,

    // dims: n-systems x max-n-blocks x max-n-blocks x
    // max-n-interblock-connections x max-n-interblock-connections
    TView<Int, 5, D> system_inter_block_bondsep,

    // dims n-systems x max-n-blocks x max-n-neighbors
    // -1 as the sentinel
    TView<Int, 3, D> system_neighbor_list,

    //////////////////////
    // Chemical properties
    // how many atoms for a given block
    // Dimsize n_block_types
    TView<Int, 1, D> block_type_n_heavy_atoms,

    // index of the ith heavy atom in a block type
    TView<Int, 2, D> block_type_heavyatom_index,

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
    TView<LKTypeParams<Real>, 1, D> type_params,
    TView<LJGlobalParams<Real>, 1, D> global_params,
    TView<Real, 1, D> lj_lk_weights,
    TView<Real, 1, D> output_tensor) -> void {
  int const n_systems = system_min_bond_separation.size(0);
  int const n_contexts = context_coords.size(0);
  int64_t const n_alternate_blocks = alternate_coords.size(0);
  int const max_n_blocks = context_coords.size(1);
  int64_t const max_n_atoms = context_coords.size(2);
  int64_t const max_n_heavy_atoms = block_type_heavyatom_index.size(1);
  int const n_block_types = block_type_n_heavy_atoms.size(0);
  int const max_n_interblock_bonds =
      block_type_atoms_forming_chemical_bonds.size(1);
  int64_t const max_n_neighbors = system_neighbor_list.size(2);

  // std::cout << "type params" << type_params.size(0) << std::endl;

  assert(alternate_coords.size(1) == max_n_atoms);
  assert(alternate_ids.size(0) == n_alternate_blocks);
  assert(context_coords.size(0) == context_block_type.size(0));
  assert(context_system_ids.size(0) == n_contexts);

  assert(system_min_bond_separation.size(1) == max_n_blocks);
  assert(system_min_bond_separation.size(2) == max_n_blocks);

  assert(system_inter_block_bondsep.size(0) == n_systems);
  assert(system_inter_block_bondsep.size(1) == max_n_blocks);
  assert(system_inter_block_bondsep.size(2) == max_n_blocks);
  assert(system_inter_block_bondsep.size(3) == max_n_interblock_bonds);
  assert(system_inter_block_bondsep.size(4) == max_n_interblock_bonds);
  assert(system_neighbor_list.size(0) == n_systems);
  assert(system_neighbor_list.size(1) == max_n_blocks);

  assert(block_type_atom_types.size(0) == n_block_types);
  assert(block_type_atom_types.size(1) == max_n_atoms);
  assert(block_type_n_interblock_bonds.size(0) == n_block_types);
  assert(block_type_atoms_forming_chemical_bonds.size(0) == n_block_types);
  assert(block_type_path_distance.size(0) == n_block_types);
  assert(block_type_path_distance.size(1) == max_n_atoms);
  assert(block_type_path_distance.size(2) == max_n_atoms);

  assert(lj_lk_weights.size(0) == 2);

  assert(output_tensor.size(0) == n_alternate_blocks);

  // auto wcts = std::chrono::system_clock::now();
  // clock_t start_time = clock();

  // auto output_t = TPack<Real, 1, D>::zeros({n_alternate_blocks});
  // auto output = output_t.view;
  // auto count_t = TPack<int, 1, D>::zeros({1});
  // auto count = count_t.view;

  using namespace mgpu;
  typedef launch_box_t<
      arch_20_cta<32, 1>,
      arch_35_cta<32, 1>,
      arch_52_cta<32, 1>>
      launch_t;

  // between one alternate rotamer and its neighbors in the surrounding context
  auto score_inter_pairs = ([=] MGPU_DEVICE(
                                int tid,
                                int alt_start_heavy_atom,
                                int neighb_start_heavy_atom,
                                Real *alt_coords,
                                Real *neighb_coords,
                                Int *alt_atom_ind,
                                Int *neighb_atom_ind,
                                // Int *alt_atom_type,
                                // Int *neighb_atom_type,
                                LKTypeParams<Real> *params1,
                                LKTypeParams<Real> *params2,
                                int const max_important_bond_separation,
                                int const alt_block_ind,
                                int const neighb_block_ind,
                                int const alt_block_type,
                                int const neighb_block_type,

                                int const min_separation,
                                TensorAccessor<Int, 4, D> inter_block_bondsep,

                                int const alt_n_heavy_atoms,
                                int const neighb_n_heavy_atoms,
                                int const n_conn1,
                                int const n_conn2,
                                int const *path_dist1,
                                int const *path_dist2,
                                int const *conn_seps) {
    Real score_total = 0;
    Real coord1[3];
    Real coord2[3];

    int const alt_remain =
        min(TILE_SIZE, alt_n_heavy_atoms - alt_start_heavy_atom);
    int const neighb_remain =
        min(TILE_SIZE, neighb_n_heavy_atoms - neighb_start_heavy_atom);

    int const n_pairs = alt_remain * neighb_remain;

    LJGlobalParams<Real> global_params_local = global_params[0];
    Real const lk_weight = lj_lk_weights[1];

    for (int i = tid; i < n_pairs; i += blockDim.x) {
      // Tile numbering
      int const alt_heavy_atom_tile_ind = i / neighb_remain;
      int const neighb_heavy_atom_tile_ind = i % neighb_remain;

      // Block-type numbering
      int const alt_heavy_atom_ind =
          alt_heavy_atom_tile_ind + alt_start_heavy_atom;
      int const neighb_heavy_atom_ind =
          neighb_heavy_atom_tile_ind + neighb_start_heavy_atom;

      // Load atom data from shared
      for (int j = 0; j < 3; ++j) {
        coord1[j] = alt_coords[3 * alt_heavy_atom_tile_ind + j];
        coord2[j] = neighb_coords[3 * neighb_heavy_atom_tile_ind + j];
      }
      Real d2 =
          ((coord1[0] - coord2[0]) * (coord1[0] - coord2[0])
           + (coord1[1] - coord2[1]) * (coord1[1] - coord2[1])
           + (coord1[2] - coord2[2]) * (coord1[2] - coord2[2]));
      if (d2 > 36) {
        // DANGER! Duplication of max-dist = 6A here
        continue;
      }
      Real dist = sqrt(d2);

      int const atom_ind1 = alt_atom_ind[alt_heavy_atom_tile_ind];
      int const atom_ind2 = neighb_atom_ind[neighb_heavy_atom_tile_ind];
      // int const atom_1_type = alt_atom_type[alt_heavy_atom_tile_ind];
      // int const atom_2_type = neighb_atom_type[neighb_heavy_atom_tile_ind];

      int separation = min_separation;
      if (separation <= max_important_bond_separation) {
        separation =
            common::count_pair::CountPair<D, Int>::inter_block_separation<
                TILE_SIZE>(
                max_important_bond_separation,
                alt_heavy_atom_tile_ind,
                neighb_heavy_atom_tile_ind,
                n_conn1,
                n_conn2,
                path_dist1,
                path_dist2,
                conn_seps);
      }
      // int const separation =
      //     min_separation > max_important_bond_separation
      //         ? max_important_bond_separation
      //         : common::count_pair::CountPair<D,
      //         Int>::inter_block_separation(
      //               max_important_bond_separation,
      //               alt_block_ind,
      //               neighb_block_ind,
      //               alt_block_type,
      //               neighb_block_type,
      //               atom_ind1,
      //               atom_ind2,
      //               inter_block_bondsep,
      //               block_type_n_interblock_bonds,
      //               block_type_atoms_forming_chemical_bonds,
      //               block_type_path_distance);

      LKTypeParams<Real> p1 = params1[alt_heavy_atom_tile_ind];
      LKTypeParams<Real> p2 = params2[neighb_heavy_atom_tile_ind];

      Real lk = lk_isotropic_score<Real>::V(
          dist,
          separation,
          // type_params[atom_1_type],
          // type_params[atom_2_type],
          p1,
          p2,
          global_params_local);

      lk *= lk_weight;
      score_total += lk;
    }
    return score_total;
  });

  // between one atoms within an alternate rotamer
  auto score_intra_pairs = ([=] MGPU_DEVICE(
                                int tid,
                                int start_heavy_atom1,
                                int start_heavy_atom2,
                                Real *coords1,
                                Real *coords2,
                                Int *atom_ind1,
                                Int *atom_ind2,
                                // Int *atom_type1,
                                // Int *atom_type2,
                                LKTypeParams<Real> *params1,
                                LKTypeParams<Real> *params2,
                                int const max_important_bond_separation,
                                int const block_type,
                                int const n_heavy_atoms) {
    Real score_total = 0;
    Real coord1[3];
    Real coord2[3];

    int const remain1 = min(TILE_SIZE, n_heavy_atoms - start_heavy_atom1);
    int const remain2 = min(TILE_SIZE, n_heavy_atoms - start_heavy_atom2);

    int const n_pairs = remain1 * remain2;

    LJGlobalParams<Real> global_params_local = global_params[0];
    Real const lk_weight = lj_lk_weights[1];

    for (int i = tid; i < n_pairs; i += blockDim.x) {
      int const heavy_atom_tile_ind_1 = i / remain2;
      int const heavy_atom_tile_ind_2 = i % remain2;
      int const heavy_atom_ind_1 = heavy_atom_tile_ind_1 + start_heavy_atom1;
      int const heavy_atom_ind_2 = heavy_atom_tile_ind_2 + start_heavy_atom2;
      if (heavy_atom_ind_1 >= heavy_atom_ind_2) {
        continue;
      }
      int const atom_ind_1 = atom_ind1[heavy_atom_tile_ind_1];
      int const atom_ind_2 = atom_ind2[heavy_atom_tile_ind_2];

      for (int j = 0; j < 3; ++j) {
        coord1[j] = coords1[3 * heavy_atom_tile_ind_1 + j];
        coord2[j] = coords2[3 * heavy_atom_tile_ind_2 + j];
      }
      // int const atom_1_type = atom_type1[heavy_atom_tile_ind_1];
      // int const atom_2_type = atom_type2[heavy_atom_tile_ind_2];

      int const separation =
          block_type_path_distance[block_type][atom_ind_1][atom_ind_2];

      Real const dist = sqrt(
          (coord1[0] - coord2[0]) * (coord1[0] - coord2[0])
          + (coord1[1] - coord2[1]) * (coord1[1] - coord2[1])
          + (coord1[2] - coord2[2]) * (coord1[2] - coord2[2]));

      LKTypeParams<Real> p1 = params1[heavy_atom_tile_ind_1];
      LKTypeParams<Real> p2 = params1[heavy_atom_tile_ind_2];

      Real lk = lk_isotropic_score<Real>::V(
          dist,
          separation,
          // type_params[atom_1_type],
          // type_params[atom_2_type],
          p1,
          p2,
          global_params_local);

      lk *= lk_weight;
      score_total += lk;
    }
    return score_total;
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

    __shared__ union {
      struct {
        Real coords1[TILE_SIZE * 3];
        Real coords2[TILE_SIZE * 3];
        Int at_ind1[TILE_SIZE];
        Int at_ind2[TILE_SIZE];
        // Int at_type1[TILE_SIZE];
        // Int at_type2[TILE_SIZE];
        LKTypeParams<Real> params1[TILE_SIZE];
        LKTypeParams<Real> params2[TILE_SIZE];
        int min_separation;
        Int n_conn1;
        Int n_conn2;
        Int conn_ats1[MAX_N_CONN];
        Int conn_ats2[MAX_N_CONN];
        Int path_dist1[MAX_N_CONN * TILE_SIZE];
        Int path_dist2[MAX_N_CONN * TILE_SIZE];
        Int conn_seps[MAX_N_CONN * MAX_N_CONN];
      } vals;
      typename reduce_t::storage_t reduce;
    } shared;

    Real *coords1 = shared.vals.coords1;
    Real *coords2 = shared.vals.coords2;
    Int *at_ind1 = shared.vals.at_ind1;
    Int *at_ind2 = shared.vals.at_ind2;
    // Int *at_type1 = shared.vals.at_type1;
    // Int *at_type2 = shared.vals.at_type2;
    LKTypeParams<Real> *params1 = shared.vals.params1;
    LKTypeParams<Real> *params2 = shared.vals.params2;

    int alt_ind = cta / max_n_neighbors;
    int neighb_ind = cta % max_n_neighbors;

    int const max_important_bond_separation = 4;
    int const alt_context = alternate_ids[alt_ind][0];
    if (alt_context == -1) {
      return;
    }

    int const alt_block_ind = alternate_ids[alt_ind][1];
    int const alt_block_type = alternate_ids[alt_ind][2];
    int const system = context_system_ids[alt_context];

    int const neighb_block_ind =
        system_neighbor_list[system][alt_block_ind][neighb_ind];
    if (neighb_block_ind == -1) {
      return;
    }

    // Let's load coordinates
    int const n_iterations = (max_n_heavy_atoms - 4 - 1) / TILE_SIZE + 1;

    Real totalE = 0;
    if (alt_block_ind != neighb_block_ind) {
      int const neighb_block_type =
          context_block_type[alt_context][neighb_block_ind];
      int const alt_n_heavy_atoms = block_type_n_heavy_atoms[alt_block_type];
      int const neighb_n_heavy_atoms =
          block_type_n_heavy_atoms[neighb_block_type];

      if (tid == 0) {
        int const min_sep =
            system_min_bond_separation[system][alt_block_ind][neighb_block_ind];
        shared.vals.min_separation = min_sep;
        int const n_conn1 = block_type_n_interblock_bonds[alt_block_type];
        int const n_conn2 = block_type_n_interblock_bonds[neighb_block_type];
        shared.vals.n_conn1 = n_conn1;
        shared.vals.n_conn2 = n_conn2;
      }
      __syncthreads();
      int const min_sep = shared.vals.min_separation;

      bool const count_pair_striking_dist =
          min_sep <= max_important_bond_separation;

      int const n_conn1 = shared.vals.n_conn1;
      int const n_conn2 = shared.vals.n_conn2;
      if (count_pair_striking_dist && tid < n_conn1) {
        shared.vals.conn_ats1[tid] =
            block_type_atoms_forming_chemical_bonds[alt_block_type][tid];
      }
      if (count_pair_striking_dist && tid < n_conn2) {
        shared.vals.conn_ats2[tid] =
            block_type_atoms_forming_chemical_bonds[neighb_block_type][tid];
      }
      if (count_pair_striking_dist && tid < n_conn1 * n_conn2) {
        int conn1 = tid / n_conn2;
        int conn2 = tid % n_conn2;
        shared.vals.conn_seps[tid] =
            system_inter_block_bondsep[system][alt_block_ind][neighb_block_ind]
                                      [conn1][conn2];
      }
      __syncthreads();

      // Tile the sets of TILE_SIZE atoms
      for (int i = 0; i < n_iterations; ++i) {
        if (i != 0) {
          // make sure calculations from previous iteration have finished before
          // we overwrite shared memory
          __syncthreads();
        }
        if (tid < TILE_SIZE && i * TILE_SIZE + tid + 4 < max_n_heavy_atoms) {
          int atid = i * TILE_SIZE + tid + 4;
          int heavy_ind = block_type_heavyatom_index[alt_block_type][atid];
          at_ind1[tid] = heavy_ind;
          if (heavy_ind >= 0) {
            for (int j = 0; j < 3; ++j) {
              coords1[3 * tid + j] = alternate_coords[alt_ind][heavy_ind][j];
            }
            int const attype = block_type_atom_types[alt_block_type][heavy_ind];
            // at_type1[tid] = attype;
            params1[tid] = type_params[attype];
            if (count_pair_striking_dist) {
              for (int j = 0; j < n_conn1; ++j) {
                int ij_path_dist =
                    block_type_path_distance[alt_block_type]
                                            [shared.vals.conn_ats1[j]]
                                            [heavy_ind];

                // path dist indexed by heavy-atom index and not atom index
                shared.vals.path_dist1[j * TILE_SIZE + tid] = ij_path_dist;
              }
            }
          }
        }

        for (int j = 0; j < n_iterations; ++j) {
          if (j != 0) {
            // make sure calculations from previous iteration have finished
            // before we overwrite the contents of shared memory
            __syncthreads();
          }
          if (tid < TILE_SIZE && j * TILE_SIZE + 4 + tid < max_n_heavy_atoms) {
            int atid = j * TILE_SIZE + tid + 4;
            int heavy_ind = block_type_heavyatom_index[neighb_block_type][atid];
            at_ind2[tid] = heavy_ind;
            if (heavy_ind >= 0) {
              for (int k = 0; k < 3; ++k) {
                coords2[3 * tid + k] =
                    context_coords[alt_context][neighb_block_ind][heavy_ind][k];
              }
              int attype = block_type_atom_types[neighb_block_type][heavy_ind];
              // at_type2[tid] = attype;
              LKTypeParams<Real> params_local = type_params[attype];
              params2[tid] = params_local;
              if (count_pair_striking_dist) {
                for (int k = 0; k < n_conn2; ++k) {
                  int jk_path_dist =
                      block_type_path_distance[neighb_block_type]
                                              [shared.vals.conn_ats2[k]]
                                              [heavy_ind];
                  // path dist indexed by heavy-atom index and not atom index
                  shared.vals.path_dist2[k * TILE_SIZE + tid] = jk_path_dist;
                }
              }
            }
          }

          // wait for shared memory to be fully loaded before we start
          // calculating energies
          __syncthreads();

          // Now we will calculate ij pairs
          // printf("cuda score inter pairs %d %d %d %d\n", tid, cta, i, j);

          totalE += score_inter_pairs(
              tid,
              i * TILE_SIZE + 4,
              j * TILE_SIZE + 4,
              coords1,
              coords2,
              at_ind1,
              at_ind2,
              // at_type1,
              // at_type2,
              params1,
              params2,
              max_important_bond_separation,
              alt_block_ind,
              neighb_block_ind,
              alt_block_type,
              neighb_block_type,
              min_sep,
              system_inter_block_bondsep[system],
              alt_n_heavy_atoms,
              neighb_n_heavy_atoms,
              n_conn1,
              n_conn2,
              shared.vals.path_dist1,
              shared.vals.path_dist2,
              shared.vals.conn_seps);
        }  // for j
      }  // for i
    } else {
      int const alt_n_heavy_atoms = block_type_n_heavy_atoms[alt_block_type];

      for (int i = 0; i < n_iterations; ++i) {
        if (i != 0) {
          // make sure the previous iteration has completed before we
          // overwrite the contents of shared memory
          __syncthreads();
        }
        if (tid < TILE_SIZE && i * TILE_SIZE + tid + 4 < max_n_heavy_atoms) {
          int atid = i * TILE_SIZE + tid + 4;
          int heavy_ind = block_type_heavyatom_index[alt_block_type][atid];
          at_ind1[tid] = heavy_ind;
          if (heavy_ind >= 0) {
            for (int j = 0; j < 3; ++j) {
              coords1[3 * tid + j] = alternate_coords[alt_ind][heavy_ind][j];
            }
            int attype = block_type_atom_types[alt_block_type][heavy_ind];
            // at_type1[tid] = attype;
            LKTypeParams<Real> params_local = type_params[attype];
            params1[tid] = params_local;
          }
        }
        for (int j = i; j < n_iterations; ++j) {
          if (j != i) {
            // make sure previous iteration has completed before we
            // overwrite the contents of shared memory
            __syncthreads();
          }
          if (j != i && tid < TILE_SIZE
              && j * TILE_SIZE + tid + 4 < max_n_heavy_atoms) {
            int atid = j * TILE_SIZE + tid + 4;
            int heavy_ind = block_type_heavyatom_index[alt_block_type][atid];
            if (heavy_ind >= 0) {
              at_ind2[tid] = heavy_ind;
              for (int k = 0; k < 3; ++k) {
                coords2[3 * tid + k] = alternate_coords[alt_ind][heavy_ind][k];
              }
              int attype = block_type_atom_types[alt_block_type][heavy_ind];
              // at_type2[tid] = attype;
              LKTypeParams<Real> params_local = type_params[attype];
              params2[tid] = params_local;
            }
          }

          // all threads must wait for shared memory to be loaded before
          // beginning energy calculations
          __syncthreads();

          totalE += score_intra_pairs(
              tid,
              i * TILE_SIZE + 4,
              j * TILE_SIZE + 4,
              coords1,
              (i == j ? coords1 : coords2),
              at_ind1,
              (i == j ? at_ind1 : at_ind2),
              // at_type1,
              //(i == j ? at_type1 : at_type2),
              params1,
              (i == j ? params1 : params2),
              max_important_bond_separation,
              alt_block_type,
              alt_n_heavy_atoms);
        }  // for j
      }  // for i
    }  // else

    // wait for all calcs to conclude before overwriting
    // shared memory in the reduction
    __syncthreads();

    Real all_reduce =
        reduce_t().reduce(tid, totalE, shared.reduce, nt, mgpu::plus_t<Real>());

    if (tid == 0 && all_reduce != 0) {
      atomicAdd(&output_tensor[alt_ind], all_reduce);
    }
  });

  // Allocate and zero the output tensors in a separate stream
  at::cuda::CUDAStream wrapped_stream = at::cuda::getStreamFromPool();
  setCurrentCUDAStream(wrapped_stream);
  //
  // TEMP mgpu::standard_context_t context(wrapped_stream.stream());
  mgpu::standard_context_t context;
  mgpu::cta_launch<launch_t>(
      eval_energies, n_alternate_blocks * max_n_neighbors, context);

  at::cuda::setCurrentCUDAStream(at::cuda::getDefaultCUDAStream());

#ifdef __CUDACC__
  float first;
  cudaMemcpy(&first, &output_tensor[0], sizeof(float), cudaMemcpyDeviceToHost);
  //
  // clock_t stop_time = clock();
  // std::chrono::duration<double> wctduration =
  // (std::chrono::system_clock::now() - wcts);
  //
  // std::cout << n_systems << " " << n_contexts << " " <<n_alternate_blocks <<
  // " "; std::cout << n_alternate_blocks * max_n_neighbors * max_n_atoms *
  // max_n_atoms << " "; std::cout << "runtime? " << ((double)stop_time -
  // start_time) / CLOCKS_PER_SEC
  //           << " wall time: " << wctduration.count() << " " << first
  //           << std::endl;
#endif
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
class LKRPECudaCalc : public pack::sim_anneal::compiled::RPECalc {
 public:
  LKRPECudaCalc(
      TView<Vec<Real, 3>, 3, D> context_coords,
      TView<Int, 2, D> context_block_type,
      TView<Vec<Real, 3>, 2, D> alternate_coords,
      TView<Vec<Int, 3>, 1, D> alternate_ids,
      TView<Int, 1, D> context_system_ids,
      TView<Int, 3, D> system_min_bond_separation,
      TView<Int, 5, D> system_inter_block_bondsep,
      TView<Int, 3, D> system_neighbor_list,
      TView<Int, 1, D> block_type_n_heavy_atoms,
      TView<Int, 2, D> block_type_heavyatom_index,
      TView<Int, 2, D> block_type_atom_types,
      TView<Int, 1, D> block_type_n_interblock_bonds,
      TView<Int, 2, D> block_type_atoms_forming_chemical_bonds,
      TView<Int, 3, D> block_type_path_distance,
      TView<LKTypeParams<Real>, 1, D> type_params,
      TView<LJGlobalParams<Real>, 1, D> global_params,
      TView<Real, 1, D> lj_lk_weights,
      TView<Real, 1, D> output)
      : context_coords_(context_coords),
        context_block_type_(context_block_type),
        alternate_coords_(alternate_coords),
        alternate_ids_(alternate_ids),
        context_system_ids_(context_system_ids),
        system_min_bond_separation_(system_min_bond_separation),
        system_inter_block_bondsep_(system_inter_block_bondsep),
        system_neighbor_list_(system_neighbor_list),
        block_type_n_heavy_atoms_(block_type_n_heavy_atoms),
        block_type_heavyatom_index_(block_type_heavyatom_index),
        block_type_atom_types_(block_type_atom_types),
        block_type_n_interblock_bonds_(block_type_n_interblock_bonds),
        block_type_atoms_forming_chemical_bonds_(
            block_type_atoms_forming_chemical_bonds),
        block_type_path_distance_(block_type_path_distance),
        type_params_(type_params),
        global_params_(global_params),
        lj_lk_weights_(lj_lk_weights),
        output_(output) {}

  void calc_energies() override {
    LKRPEDispatch<DeviceDispatch, D, Real, Int>::f(
        context_coords_,
        context_block_type_,
        alternate_coords_,
        alternate_ids_,
        context_system_ids_,
        system_min_bond_separation_,
        system_inter_block_bondsep_,
        system_neighbor_list_,
        block_type_n_heavy_atoms_,
        block_type_heavyatom_index_,
        block_type_atom_types_,
        block_type_n_interblock_bonds_,
        block_type_atoms_forming_chemical_bonds_,
        block_type_path_distance_,
        type_params_,
        global_params_,
        lj_lk_weights_,
        output_);
  }

  void finalize() override {}

 private:
  TView<Vec<Real, 3>, 3, D> context_coords_;
  TView<Int, 2, D> context_block_type_;
  TView<Vec<Real, 3>, 2, D> alternate_coords_;
  TView<Vec<Int, 3>, 1, D> alternate_ids_;
  TView<Int, 1, D> context_system_ids_;
  TView<Int, 3, D> system_min_bond_separation_;
  TView<Int, 5, D> system_inter_block_bondsep_;
  TView<Int, 3, D> system_neighbor_list_;
  TView<Int, 1, D> block_type_n_heavy_atoms_;
  TView<Int, 2, D> block_type_heavyatom_index_;
  TView<Int, 2, D> block_type_atom_types_;
  TView<Int, 1, D> block_type_n_interblock_bonds_;
  TView<Int, 2, D> block_type_atoms_forming_chemical_bonds_;
  TView<Int, 3, D> block_type_path_distance_;
  TView<LKTypeParams<Real>, 1, D> type_params_;
  TView<LJGlobalParams<Real>, 1, D> global_params_;
  TView<Real, 1, D> lj_lk_weights_;
  TView<Real, 1, D> output_;
};

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto LKRPERegistratorDispatch<DeviceDispatch, D, Real, Int>::f(
    TView<Vec<Real, 3>, 3, D> context_coords,
    TView<Int, 2, D> context_block_type,
    TView<Vec<Real, 3>, 2, D> alternate_coords,
    TView<Vec<Int, 3>, 1, D>
        alternate_ids,  // 0 == context id; 1 == block id; 2 == block type

    // which system does a given context belong to
    TView<Int, 1, D> context_system_ids,

    // dims: n-systems x max-n-blocks x max-n-blocks
    // Quick lookup: given the inds of two blocks, ask: what is the minimum
    // number of chemical bonds that separate any pair of atoms in those blocks?
    // If this minimum is greater than the crossover, then no further logic for
    // deciding whether two atoms in those blocks should have their interaction
    // energies calculated: all should. intentionally small to (possibly) fit in
    // constant cache
    TView<Int, 3, D> system_min_bond_separation,

    // dims: n-systems x max-n-blocks x max-n-blocks x
    // max-n-interblock-connections x max-n-interblock-connections
    TView<Int, 5, D> system_inter_block_bondsep,

    // dims n-systems x max-n-blocks x max-n-neighbors
    // -1 as the sentinel
    TView<Int, 3, D> system_neighbor_list,

    //////////////////////
    // Chemical properties
    // how many atoms for a given block
    // Dimsize n_block_types
    TView<Int, 1, D> block_type_n_heavy_atoms,

    // index of the ith heavy atom in a block type
    TView<Int, 2, D> block_type_heavyatom_index,

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
    TView<LKTypeParams<Real>, 1, D> type_params,
    TView<LJGlobalParams<Real>, 1, D> global_params,
    TView<Real, 1, D> lj_lk_weights,
    TView<Real, 1, D> output,
    TView<int64_t, 1, tmol::Device::CPU> annealer) -> void {
  using tmol::pack::sim_anneal::compiled::RPECalc;
  using tmol::pack::sim_anneal::compiled::SimAnnealer;

  int64_t annealer_uint = annealer[0];
  SimAnnealer *sim_annealer = reinterpret_cast<SimAnnealer *>(annealer_uint);
  std::shared_ptr<RPECalc> calc =
      std::make_shared<LKRPECudaCalc<DeviceDispatch, D, Real, Int>>(
          context_coords,
          context_block_type,
          alternate_coords,
          alternate_ids,
          context_system_ids,
          system_min_bond_separation,
          system_inter_block_bondsep,
          system_neighbor_list,
          block_type_n_heavy_atoms,
          block_type_heavyatom_index,
          block_type_atom_types,
          block_type_n_interblock_bonds,
          block_type_atoms_forming_chemical_bonds,
          block_type_path_distance,
          type_params,
          global_params,
          lj_lk_weights,
          output);

  sim_annealer->add_score_component(calc);
}

template struct LKRPEDispatch<ForallDispatch, tmol::Device::CUDA, float, int>;
template struct LKRPEDispatch<ForallDispatch, tmol::Device::CUDA, double, int>;

template struct LKRPERegistratorDispatch<
    ForallDispatch,
    tmol::Device::CUDA,
    float,
    int>;
template struct LKRPERegistratorDispatch<
    ForallDispatch,
    tmol::Device::CUDA,
    double,
    int>;

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
