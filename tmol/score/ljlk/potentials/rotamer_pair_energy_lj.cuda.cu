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

#include <tmol/score/ljlk/potentials/lj.hh>
#include <tmol/score/ljlk/potentials/lk_isotropic.hh>
#include <tmol/score/ljlk/potentials/rotamer_pair_energy_lj.hh>

#include <tmol/pack/sim_anneal/compiled/annealer.hh>

#include <chrono>

#include <tmol/score/common/forall_dispatch.cuda.impl.cuh>

//#include <moderngpu/cta_load_balance.hxx>
#include <moderngpu/cta_reduce.hxx>
//#include <moderngpu/cta_scan.hxx>
//#include <moderngpu/cta_segreduce.hxx>
//#include <moderngpu/cta_segscan.hxx>
//#include <moderngpu/memory.hxx>
//#include <moderngpu/search.hxx>
#include <moderngpu/transform.hxx>

// This file moves in more recent versions of Torch
#include <c10/cuda/CUDAStream.h>

// The maximum number of inter-residue chemical bonds
#define MAX_N_CONN 4
#define TILE_SIZE 32

int count_scoring_passes = 0;

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

cudaStream_t ljlk_stream = 0;

void clear_old_score_events(std::list<cudaEvent_t> &previously_created_events) {
  return;
  for (auto event_iter = previously_created_events.begin();
       event_iter != previously_created_events.end();
       /*no increment*/) {
    cudaEvent_t event = *event_iter;
    cudaError_t status = cudaEventQuery(event);
    auto event_iter_next = event_iter;
    ++event_iter_next;
    if (status == cudaSuccess) {
      cudaEventDestroy(event);
      previously_created_events.erase(event_iter);
    }
    event_iter = event_iter_next;
  }
}

void create_score_event(
    TView<int64_t, 1, tmol::Device::CPU> score_event,
    std::list<cudaEvent_t> previously_created_events) {
  assert(score_event.size(0) == 1);
  cudaEvent_t event;
  cudaEventCreate(&event);
  score_event[0] = reinterpret_cast<int64_t>(event);
  previously_created_events.push_back(event);
}

void wait_on_annealer_event(
    cudaStream_t stream, TView<int64_t, 1, tmol::Device::CPU> annealer_event) {
  assert(annealer_event.size(0) == 1);
  cudaEvent_t event = reinterpret_cast<cudaEvent_t>(annealer_event[0]);
  if (event) {
    // std::cout << "LJLK " << count_scoring_passes << " -- Waiting on event "
    // << event << " in stream " << stream << std::endl;
    cudaStreamWaitEvent(stream, event, 0);
  }
}

void record_scoring_event(
    cudaStream_t stream, TView<int64_t, 1, tmol::Device::CPU> score_event) {
  assert(score_event.size(0) == 1);
  cudaEvent_t event = reinterpret_cast<cudaEvent_t>(score_event[0]);
  if (event) {
    // std::cout << "LJLK " << count_scoring_passes << " -- Recording score
    // event " << event << " in stream " << stream << std::endl;
    cudaEventRecord(event, stream);
  }
}

void sync_and_destroy_old_score_events(
    std::list<cudaEvent_t> &previously_created_events) {
  for (auto event : previously_created_events) {
    cudaEventSynchronize(event);
    cudaEventDestroy(event);
  }
  previously_created_events.clear();
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto LJLKRPEDispatch<DeviceDispatch, D, Real, Int>::f(
    TView<Vec<Real, 3>, 2, D>
        context_coords,  // n-contexts x max-n-atoms-per-context
    TView<Int, 2, D>
        context_coord_offsets,  // n-contexts x max-n-blocks-per-context
    TView<Int, 2, D>
        context_block_type,  // n-contexts x max-n-blocks-per-context
    TView<Vec<Real, 3>, 1, D>
        alternate_coords,  // max-n-atoms-in-all-alt-coord-contexts
    TView<Int, 1, D> alternate_coord_offsets,  // n-alternate-blocks
    TView<Vec<Int, 3>, 1, D> alternate_ids,    // n-alternate-blocks
    // 0 == context id; 1 == block id; 2 == block type

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
    TView<LJGlobalParams<Real>, 1, D> global_params,

    TView<Real, 1, D> lj_lk_weights,
    TView<Real, 1, D> output,

    TView<int64_t, 1, tmol::Device::CPU> score_event,
    TView<int64_t, 1, tmol::Device::CPU> annealer_event

    ) -> void {
  int const n_systems = system_min_bond_separation.size(0);
  int const n_contexts = context_coords.size(0);
  // int64_t const n_alternate_blocks = alternate_coords.size(0);
  int64_t const n_alternate_blocks = alternate_coord_offsets.size(0);
  int const max_n_blocks = context_coord_offsets.size(1);
  // int64_t const max_n_atoms_per_block = context_coords.size(2);
  int64_t const max_n_atoms_per_block = block_type_atom_types.size(1);
  int const n_block_types = block_type_n_atoms.size(0);
  int const max_n_interblock_bonds =
      block_type_atoms_forming_chemical_bonds.size(1);
  int64_t const max_n_neighbors = system_neighbor_list.size(2);
  int64_t const n_atom_types = type_params.size(0);
  int64_t const max_n_tiles = block_type_n_heavy_atoms_in_tile.size(2);

  // Size assertions
  assert(context_coord_offsets.size(0) == n_contexts);
  assert(context_block_type.size(0) == n_contexts);
  assert(context_block_type.size(1) == max_n_blocks);

  // assert(alternate_coords.size(1) == max_n_atoms);
  assert(alternate_ids.size(0) == n_alternate_blocks);

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

  assert(block_type_n_heavy_atoms_in_tile.size(0) == n_block_types);
  assert(block_type_heavy_atoms_in_tile.size(0) == n_block_types);
  assert(block_type_heavy_atoms_in_tile.size(1) == TILE_SIZE * max_n_tiles);

  assert(block_type_atom_types.size(0) == n_block_types);
  assert(block_type_atom_types.size(1) == max_n_atoms);
  assert(block_type_n_interblock_bonds.size(0) == n_block_types);
  assert(block_type_atoms_forming_chemical_bonds.size(0) == n_block_types);
  assert(block_type_path_distance.size(0) == n_block_types);
  assert(block_type_path_distance.size(1) == max_n_atoms_per_block);
  assert(block_type_path_distance.size(2) == max_n_atoms_per_block);

  assert(lj_lk_weights.size(0) == 2);
  assert(max_n_interblock_bonds <= MAX_N_CONN);

  using namespace mgpu;
  typedef launch_box_t<
      arch_20_cta<32, 1>,
      arch_35_cta<32, 1>,
      arch_52_cta<32, 1>>
      launch_t;

  int const local_count_scoring_passes = ++count_scoring_passes;

  // between one alternate rotamer and its neighbors in the surrounding context
  auto score_inter_pairs_lj =
      ([=] MGPU_DEVICE(
           int tid,
           int alt_start_atom,
           int neighb_start_atom,
           Real *__restrict__ alt_coords,                     // shared
           Real *__restrict__ neighb_coords,                  // shared
           LJLKTypeParams<Real> *__restrict__ alt_params,     // shared
           LJLKTypeParams<Real> *__restrict__ neighb_params,  // shared
           int const max_important_bond_separation,
           int const min_separation,

           int const alt_n_atoms,
           int const neighb_n_atoms,
           int const alt_n_conn,
           int const neighb_n_conn,
           unsigned char const *__restrict__ alt_path_dist,     // shared
           unsigned char const *__restrict__ neighb_path_dist,  // shared
           unsigned char const *__restrict__ conn_seps) {       // shared
        Real score_total = 0;
        Real coord1[3];
        Real coord2[3];

        int const alt_remain = min(TILE_SIZE, alt_n_atoms - alt_start_atom);
        int const neighb_remain =
            min(TILE_SIZE, neighb_n_atoms - neighb_start_atom);

        int const n_pairs = alt_remain * neighb_remain;

        LJGlobalParams<Real> global_params_local = global_params[0];
        Real lj_weight = lj_lk_weights[0];
        for (int i = tid; i < n_pairs; i += blockDim.x) {
          int const alt_atom_tile_ind = i / neighb_remain;
          int const neighb_atom_tile_ind = i % neighb_remain;
          // int const alt_atom_ind = alt_atom_tile_ind + alt_start_atom;
          // int const neighb_atom_ind = neighb_atom_tile_ind +
          // neighb_start_atom;
          for (int j = 0; j < 3; ++j) {
            coord1[j] = alt_coords[3 * alt_atom_tile_ind + j];
            coord2[j] = neighb_coords[3 * neighb_atom_tile_ind + j];
          }
          Real dist2 =
              ((coord1[0] - coord2[0]) * (coord1[0] - coord2[0])
               + (coord1[1] - coord2[1]) * (coord1[1] - coord2[1])
               + (coord1[2] - coord2[2]) * (coord1[2] - coord2[2]));
          if (dist2 > 36.0) {
            // DANGER -- maximum reach of LJ potential hard coded here in a
            // second place out of range!
            continue;
          }
          Real dist = std::sqrt(dist2);
          int separation = min_separation;
          if (separation <= max_important_bond_separation) {
            separation =
                common::count_pair::CountPair<D, Int>::inter_block_separation<
                    TILE_SIZE>(
                    max_important_bond_separation,
                    alt_atom_tile_ind,
                    neighb_atom_tile_ind,
                    alt_n_conn,
                    neighb_n_conn,
                    alt_path_dist,
                    neighb_path_dist,
                    conn_seps);
          }
          Real lj = lj_score<Real>::V(
              dist,
              separation,
              alt_params[alt_atom_tile_ind].lj_params(),
              neighb_params[neighb_atom_tile_ind].lj_params(),
              global_params_local);
          score_total += lj;
        }
        return score_total * lj_weight;
      });

  auto score_inter_pairs_lk =
      ([=] MGPU_DEVICE(
           int tid,
           int alt_n_heavy,
           int neighb_n_heavy,
           Real *__restrict__ alt_coords,                        // shared
           Real *__restrict__ neighb_coords,                     // shared
           LJLKTypeParams<Real> *__restrict__ alt_params,        // shared
           LJLKTypeParams<Real> *__restrict__ neighb_params,     // shared
           unsigned char const *__restrict__ alt_heavy_inds,     // shared
           unsigned char const *__restrict__ neighb_heavy_inds,  // shared
           int const max_important_bond_separation,
           int const min_separation,
           int const alt_n_atoms,
           int const neighb_n_atoms,
           int const alt_n_conn,
           int const neighb_n_conn,
           unsigned char const *__restrict__ alt_path_dist,     // shared
           unsigned char const *__restrict__ neighb_path_dist,  // shared
           unsigned char const *__restrict__ conn_seps) {       // shared
        Real score_total = 0;
        // return score_total;

        Real coord1[3];
        Real coord2[3];

        int const n_pairs = alt_n_heavy * neighb_n_heavy;

        LJGlobalParams<Real> global_params_local = global_params[0];
        Real lk_weight = lj_lk_weights[1];
        for (int i = tid; i < n_pairs; i += blockDim.x) {
          int const alt_atom_tile_ind = i / neighb_n_heavy;
          int const neighb_atom_tile_ind = i % neighb_n_heavy;
          int const alt_atom_ind = alt_heavy_inds[alt_atom_tile_ind];
          int const neighb_atom_ind = neighb_heavy_inds[neighb_atom_tile_ind];
          if (alt_atom_ind < 0 || neighb_atom_ind < 0
              || alt_atom_ind >= TILE_SIZE || neighb_atom_ind >= TILE_SIZE) {
            printf(
                "bad atom index in lk-inter %d %d -- iteration %d\n",
                alt_atom_ind,
                neighb_atom_ind,
                local_count_scoring_passes);
            continue;
          }

          for (int j = 0; j < 3; ++j) {
            coord1[j] = alt_coords[3 * alt_atom_ind + j];
            coord2[j] = neighb_coords[3 * neighb_atom_ind + j];
          }
          Real dist2 =
              ((coord1[0] - coord2[0]) * (coord1[0] - coord2[0])
               + (coord1[1] - coord2[1]) * (coord1[1] - coord2[1])
               + (coord1[2] - coord2[2]) * (coord1[2] - coord2[2]));
          if (dist2 > 36.0) {
            // DANGER -- maximum reach of LK potential hard coded here in a
            // second place out of range!
            continue;
          }
          Real dist = std::sqrt(dist2);
          int separation = min_separation;
          if (separation <= max_important_bond_separation) {
            separation =
                common::count_pair::CountPair<D, Int>::inter_block_separation<
                    TILE_SIZE>(
                    max_important_bond_separation,
                    alt_atom_ind,
                    neighb_atom_ind,
                    alt_n_conn,
                    neighb_n_conn,
                    alt_path_dist,
                    neighb_path_dist,
                    conn_seps);
          }
          Real lk = lk_isotropic_score<Real>::V(
              dist,
              separation,
              alt_params[alt_atom_ind].lk_params(),
              neighb_params[neighb_atom_ind].lk_params(),
              global_params_local);
          score_total += lk;
        }
        return score_total * lk_weight;
      });

  // between one atoms within an alternate rotamer
  auto score_intra_pairs_lj = ([=] MGPU_DEVICE(
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
    Real coord1[3];
    Real coord2[3];

    int const remain1 = min(TILE_SIZE, n_atoms - start_atom1);
    int const remain2 = min(TILE_SIZE, n_atoms - start_atom2);

    int const n_pairs = remain1 * remain2;
    LJGlobalParams<Real> global_params_local = global_params[0];
    // Real lj_weight = lj_lk_weights[0];

    for (int i = tid; i < n_pairs; i += blockDim.x) {
      int const atom_tile_ind_1 = i / remain2;
      int const atom_tile_ind_2 = i % remain2;
      int const atom_ind_1 = atom_tile_ind_1 + start_atom1;
      int const atom_ind_2 = atom_tile_ind_2 + start_atom2;
      if (atom_ind_1 >= atom_ind_2) {
        continue;
      }

      for (int j = 0; j < 3; ++j) {
        coord1[j] = coords1[3 * atom_tile_ind_1 + j];
        coord2[j] = coords2[3 * atom_tile_ind_2 + j];
      }

      // read path distances from global memory
      int const separation =
          block_type_path_distance[block_type][atom_ind_1][atom_ind_2];

      Real const dist = sqrt(
          (coord1[0] - coord2[0]) * (coord1[0] - coord2[0])
          + (coord1[1] - coord2[1]) * (coord1[1] - coord2[1])
          + (coord1[2] - coord2[2]) * (coord1[2] - coord2[2]));

      Real lj = lj_score<Real>::V(
          dist,
          separation,
          params1[atom_tile_ind_1].lj_params(),
          params2[atom_tile_ind_2].lj_params(),
          global_params_local);

      score_total += lj;
    }
    return score_total *= lj_lk_weights[0];
    ;
  });

  // between one atoms within an alternate rotamer
  auto score_intra_pairs_lk = ([=] MGPU_DEVICE(
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
                                   int const n_atoms) {
    Real score_total = 0;
    // return score_total;
    Real coord1[3];
    Real coord2[3];

    int const n_pairs = n_heavy1 * n_heavy2;
    LJGlobalParams<Real> global_params_local = global_params[0];
    // Real lk_weight = lj_lk_weights[1];

    for (int i = tid; i < n_pairs; i += blockDim.x) {
      int const atom_heavy_tile_ind_1 = i / n_heavy2;
      int const atom_heavy_tile_ind_2 = i % n_heavy2;
      int const atom_tile_ind_1 = heavy_inds1[atom_heavy_tile_ind_1];
      int const atom_tile_ind_2 = heavy_inds2[atom_heavy_tile_ind_2];
      int const atom_ind_1 = atom_tile_ind_1 + start_atom1;
      int const atom_ind_2 = atom_tile_ind_2 + start_atom2;
      // if (atom_tile_ind_1 < 0 || atom_tile_ind_2 < 0
      //     || atom_tile_ind_1 >= TILE_SIZE || atom_tile_ind_2 >= TILE_SIZE) {
      //   printf(
      //       "bad atom index in lk-intra %d %d\n",
      //       atom_tile_ind_1,
      //       atom_tile_ind_2);
      // }

      if (atom_ind_1 >= atom_ind_2) {
        continue;
      }

      for (int j = 0; j < 3; ++j) {
        coord1[j] = coords1[3 * atom_tile_ind_1 + j];
        coord2[j] = coords2[3 * atom_tile_ind_2 + j];
      }

      // read path distances from global memory
      int const separation =
          block_type_path_distance[block_type][atom_ind_1][atom_ind_2];

      Real const dist = sqrt(
          (coord1[0] - coord2[0]) * (coord1[0] - coord2[0])
          + (coord1[1] - coord2[1]) * (coord1[1] - coord2[1])
          + (coord1[2] - coord2[2]) * (coord1[2] - coord2[2]));

      Real lk = lk_isotropic_score<Real>::V(
          dist,
          separation,
          params1[atom_tile_ind_1].lk_params(),
          params2[atom_tile_ind_2].lk_params(),
          global_params_local);

      score_total += lk;
    }
    return score_total *= lj_lk_weights[1];
    ;
  });

  auto load_alt_coords_and_params_into_shared =
      ([=] MGPU_DEVICE(
           int rot_ind,
           int rot_coord_offset,
           int n_atoms,
           int n_atoms_to_load,
           int block_type,
           int tid,
           int tile_ind,
           bool new_context_ind,
           Real *__restrict__ shared_coords,
           LJLKTypeParams<Real> *__restrict__ params,
           unsigned char *__restrict__ heavy_inds) {
        if (new_context_ind || n_atoms > TILE_SIZE) {
          mgpu::mem_to_shared<TILE_SIZE, 3>(
              reinterpret_cast<Real *>(
                  &alternate_coords[rot_coord_offset + TILE_SIZE * tile_ind]),
              tid,
              n_atoms_to_load * 3,
              shared_coords,
              false);
        }
        // if (tid < n_atoms_to_load){
        // 	for (int i = 0; i < 3; ++i) {
        // 	  coords[3*tid + i] = 1.0001 * tid;
        // 	}
        // }
        if ((new_context_ind || n_atoms > TILE_SIZE) && tid < TILE_SIZE) {
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

  auto load_alt_into_shared =
      ([=] MGPU_DEVICE(
           int rot_ind,
           int rot_coord_offset,
           int n_atoms,
           int n_atoms_to_load,
           int block_type,
           int n_conn,
           int tid,
           int tile_ind,
           bool new_context_ind,
           bool count_pair_data_loaded,
           bool count_pair_striking_dist,
           unsigned char *__restrict__ conn_ats,
           Real *__restrict__ shared_coords,
           LJLKTypeParams<Real> *__restrict__ params,
           unsigned char *__restrict__ heavy_inds,
           unsigned char *__restrict__ path_dist  // to conn
       ) {
        load_alt_coords_and_params_into_shared(
            rot_ind,
            rot_coord_offset,
            n_atoms,
            n_atoms_to_load,
            block_type,
            tid,
            tile_ind,
            new_context_ind,
            shared_coords,
            params,
            heavy_inds);
        if ((n_atoms > TILE_SIZE || !count_pair_data_loaded)
            && tid < n_atoms_to_load && count_pair_striking_dist) {
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

    __shared__ struct shared_mem_struct {
      // Memory that should remain intact between vt iterations
      Real coords_alt1[TILE_SIZE * 3];  // 786 bytes for coords
      Real coords_alt2[TILE_SIZE * 3];
      LJLKTypeParams<Real> params_alt1[TILE_SIZE];  // 1536 bytes for params
      LJLKTypeParams<Real> params_alt2[TILE_SIZE];
      unsigned char n_heavy_alt1;
      unsigned char n_heavy_alt2;
      unsigned char heavy_inds_alt1[TILE_SIZE];
      unsigned char heavy_inds_alt2[TILE_SIZE];
      unsigned char conn_ats_alt1[MAX_N_CONN];  // 8 bytes for conn ats
      unsigned char conn_ats_alt2[MAX_N_CONN];
      unsigned char
          path_dist_alt1[MAX_N_CONN * TILE_SIZE];  // 256 bytes for path dists
      unsigned char path_dist_alt2[MAX_N_CONN * TILE_SIZE];

      union {
        struct {
          Real coords_other[TILE_SIZE * 3];              // 384 bytes for coords
          LJLKTypeParams<Real> params_other[TILE_SIZE];  // 768 bytes for params
          unsigned char n_heavy_other;
          unsigned char heavy_inds_other[TILE_SIZE];
          unsigned char conn_ats_other[MAX_N_CONN];               // 4 bytes
          unsigned char path_dist_other[MAX_N_CONN * TILE_SIZE];  // 128 bypes
          unsigned char
              conn_seps[MAX_N_CONN * MAX_N_CONN];  // 64 bytes for conn/conn
        } vals;

        typename reduce_t::storage_t reduce;

      } union_vals;
    } shared;

    int n_conn_alt(-1);
    Int last_ivt_context_ind = -1;
    int last_alt_block_type1 = -1;
    int last_alt_block_type2 = -1;
    int last_rot_ind1 = -1;
    int last_rot_ind2 = -1;
    int last_alt_block_ind = -1;
    bool count_pair_data_loaded = false;

    for (int ivt = 0; ivt < vt; ++ivt) {
      Real totalE1 = 0;
      Real totalE2 = 0;

      int const ivt_context_ind = (vt * cta + ivt) / max_n_neighbors;
      int const neighb_ind = (vt * cta + ivt) % max_n_neighbors;
      // ivt_context_ind represents a pair of rotamers from the same residue
      // their indices are 2*alt_ind and 2*alt_ind+1
      if (ivt_context_ind >= n_alternate_blocks / 2) {
        break;
      }
      int const rot_ind1 = 2 * ivt_context_ind;
      int const rot_ind2 = 2 * ivt_context_ind + 1;

      int const max_important_bond_separation = 4;
      int const alt_context = alternate_ids[rot_ind1][0];
      if (alt_context == -1) {
        continue;
      }

      int const system = context_system_ids[alt_context];
      int const alt_block_ind = alternate_ids[rot_ind1][1];
      int const neighb_block_ind =
          system_neighbor_list[system][alt_block_ind][neighb_ind];
      if (neighb_block_ind == -1) {
        continue;
      }

      // only update last_ivt_context_ind if this is a legit residue pair intxn
      bool const new_context_ind = ivt_context_ind != last_ivt_context_ind;
      last_ivt_context_ind = ivt_context_ind;
      if (new_context_ind) {
        count_pair_data_loaded = false;
      }

      int const alt_block_type1 = alternate_ids[rot_ind1][2];
      int const alt_block_type2 = alternate_ids[rot_ind2][2];
      // if (!new_context_ind && alt_block_type1 != last_alt_block_type1) {
      //   printf(
      //       "%d alt_block_type1 and last_alt_block_type1 mismatch, %d vs %d,
      //       " "rotind %d vs %d \n",
      // 	    local_count_scoring_passes,
      //       alt_block_type1,
      //       last_alt_block_type1,
      //       rot_ind1,
      //       last_rot_ind1);
      // }
      // if (!new_context_ind && alt_block_type2 != last_alt_block_type2) {
      //   printf(
      //       "%d alt_block_type2 and last_alt_block_type2 mismatch, %d vs %d,
      //       " "rotind %d vs %d\n",
      // 	    local_count_scoring_passes,
      //       alt_block_type2,
      //       last_alt_block_type2,
      //       rot_ind2,
      //       last_rot_ind2);
      // }
      // if (!new_context_ind && alt_block_ind != last_alt_block_ind) {
      // 	printf(
      // 	  "%d alt_block_ind and last_alt_block_ind mismatch, %d vs %d, "
      // 	  "rotind %d vs %d\n",
      // 	  local_count_scoring_passes,
      // 	  alt_block_ind,
      // 	  last_alt_block_ind,
      // 	  rot_ind1,
      // 	  rot_ind2);
      // }
      // last_rot_ind1 = rot_ind1;
      // last_rot_ind2 = rot_ind2;
      // last_alt_block_type1 = alt_block_type1;
      // last_alt_block_type2 = alt_block_type2;
      // last_alt_block_ind = alt_block_ind;

      int const alt_n_atoms1 = block_type_n_atoms[alt_block_type1];
      int const alt_n_atoms2 = block_type_n_atoms[alt_block_type2];

      int n_conn_other(-1);

      if (alt_block_ind != neighb_block_ind) {
        // inter-residue energy evaluation

        int const neighb_block_type =
            context_block_type[alt_context][neighb_block_ind];
        int const neighb_n_atoms = block_type_n_atoms[neighb_block_type];

        if (new_context_ind) {
          // if (true) { // temp!!
          n_conn_alt = block_type_n_interblock_bonds[alt_block_type1];
        }
        int const n_conn_other =
            block_type_n_interblock_bonds[neighb_block_type];
        int const min_sep =
            system_min_bond_separation[system][alt_block_ind][neighb_block_ind];
        bool const count_pair_striking_dist =
            min_sep <= max_important_bond_separation;

        if (count_pair_striking_dist && tid < n_conn_alt) {
          int at1 =
              block_type_atoms_forming_chemical_bonds[alt_block_type1][tid];
          int at2 =
              block_type_atoms_forming_chemical_bonds[alt_block_type2][tid];
          shared.conn_ats_alt1[tid] = at1;
          shared.conn_ats_alt2[tid] = at2;
        }
        if (count_pair_striking_dist && tid < n_conn_other) {
          shared.union_vals.vals.conn_ats_other[tid] =
              block_type_atoms_forming_chemical_bonds[neighb_block_type][tid];
        }

        if (count_pair_striking_dist && tid < n_conn_alt * n_conn_other) {
          int conn1 = tid / n_conn_other;
          int conn2 = tid % n_conn_other;
          shared.union_vals.vals.conn_seps[tid] =
              system_inter_block_bondsep[system][alt_block_ind]
                                        [neighb_block_ind][conn1][conn2];
        }

        // Tile the sets of TILE_SIZE atoms
        int const alt_n_iterations =
            (max(alt_n_atoms1, alt_n_atoms2) - 1) / TILE_SIZE + 1;
        int const neighb_n_iterations = (neighb_n_atoms - 1) / TILE_SIZE + 1;
        for (int i = 0; i < alt_n_iterations; ++i) {
          // make sure all threads have completed their work
          // from the previous iteration before we overwrite
          // the contents of shared memory, and, on our first
          // iteration, make sure that the conn_ats_altX arrays
          // have been written to
          __syncthreads();

          int const i_n_atoms_to_load1 =
              max(0, min(Int(TILE_SIZE), Int((alt_n_atoms1 - TILE_SIZE * i))));

          int const i_n_atoms_to_load2 =
              max(0, min(Int(TILE_SIZE), Int((alt_n_atoms2 - TILE_SIZE * i))));

          // Let's load coordinates and Lennard-Jones parameters for
          // TILE_SIZE atoms into shared memory

          if (tid == 0) {
            shared.n_heavy_alt1 =
                block_type_n_heavy_atoms_in_tile[alt_block_type1][i];
            shared.n_heavy_alt2 =
                block_type_n_heavy_atoms_in_tile[alt_block_type2][i];
          }

          load_alt_into_shared(
              rot_ind1,
              alt_n_atoms1,
              i_n_atoms_to_load1,
              alt_block_type1,
              n_conn_alt,
              tid,
              i,
              new_context_ind,
              count_pair_data_loaded,
              count_pair_striking_dist,
              shared.conn_ats_alt1,
              shared.coords_alt1,
              shared.params_alt1,
              shared.heavy_inds_alt1,
              shared.path_dist_alt1);

          load_alt_into_shared(
              rot_ind2,
              alt_n_atoms2,
              i_n_atoms_to_load2,
              alt_block_type2,
              n_conn_alt,
              tid,
              i,
              new_context_ind,
              count_pair_data_loaded,
              count_pair_striking_dist,
              shared.conn_ats_alt2,
              shared.coords_alt2,
              shared.params_alt2,
              shared.heavy_inds_alt2,
              shared.path_dist_alt2);

          if (count_pair_striking_dist) {
            count_pair_data_loaded = true;
          }
          for (int j = 0; j < neighb_n_iterations; ++j) {
            if (j != 0) {
              // make sure that all threads have finished energy
              // calculations from the previous iteration before we
              // overwrite shared memory
              __syncthreads();
            }
            if (tid == 0) {
              shared.union_vals.vals.n_heavy_other =
                  block_type_n_heavy_atoms_in_tile[neighb_block_type][j];
              // printf("n heavy other: %d %d %d\n", alt_block_ind,
              // neighb_block_ind, shared.union_vals.vals.n_heavy_other);
            }

            int j_n_atoms_to_load =
                min(Int(TILE_SIZE), Int((neighb_n_atoms - TILE_SIZE * j)));
            mgpu::mem_to_shared<TILE_SIZE, 3>(
                reinterpret_cast<Real *>(
                    &context_coords[alt_context][neighb_block_ind]
                                   [j * TILE_SIZE]),
                tid,
                j_n_atoms_to_load * 3,
                shared.union_vals.vals.coords_other,
                false);
            if (tid < TILE_SIZE) {
              // load the Lennard-Jones parameters for these TILE_SIZE atoms
              if (tid < j_n_atoms_to_load) {
                int const atid = TILE_SIZE * j + tid;
                int const attype =
                    block_type_atom_types[neighb_block_type][atid];
                if (attype >= 0) {
                  shared.union_vals.vals.params_other[tid] =
                      type_params[attype];
                }
                shared.union_vals.vals.heavy_inds_other[tid] =
                    block_type_heavy_atoms_in_tile[neighb_block_type][atid];
                if (count_pair_striking_dist) {
                  for (int k = 0; k < n_conn_other; ++k) {
                    int jk_path_dist =
                        block_type_path_distance[neighb_block_type]
                                                [shared.union_vals.vals
                                                     .conn_ats_other[k]][atid];
                    shared.union_vals.vals
                        .path_dist_other[k * TILE_SIZE + tid] = jk_path_dist;
                  }
                }
              }
            }
            // make sure all shared memory writes have completed before we read
            // from it when calculating atom-pair energies.
            __syncthreads();
            int n_heavy_alt1 = shared.n_heavy_alt1;
            int n_heavy_alt2 = shared.n_heavy_alt2;
            int n_heavy_other = shared.union_vals.vals.n_heavy_other;

            totalE1 += score_inter_pairs_lj(
                tid,
                i * TILE_SIZE,
                j * TILE_SIZE,
                shared.coords_alt1,
                shared.union_vals.vals.coords_other,
                shared.params_alt1,
                shared.union_vals.vals.params_other,
                max_important_bond_separation,
                min_sep,
                alt_n_atoms1,
                neighb_n_atoms,
                n_conn_alt,
                n_conn_other,
                shared.path_dist_alt1,
                shared.union_vals.vals.path_dist_other,
                shared.union_vals.vals.conn_seps);

            totalE1 += score_inter_pairs_lk(
                tid,
                n_heavy_alt1,
                n_heavy_other,
                shared.coords_alt1,
                shared.union_vals.vals.coords_other,
                shared.params_alt1,
                shared.union_vals.vals.params_other,
                shared.heavy_inds_alt1,
                shared.union_vals.vals.heavy_inds_other,
                max_important_bond_separation,
                min_sep,
                alt_n_atoms1,
                neighb_n_atoms,
                n_conn_alt,
                n_conn_other,
                shared.path_dist_alt1,
                shared.union_vals.vals.path_dist_other,
                shared.union_vals.vals.conn_seps);

            totalE2 += score_inter_pairs_lj(
                tid,
                i * TILE_SIZE,
                j * TILE_SIZE,
                shared.coords_alt2,
                shared.union_vals.vals.coords_other,
                shared.params_alt2,
                shared.union_vals.vals.params_other,
                max_important_bond_separation,
                min_sep,
                alt_n_atoms2,
                neighb_n_atoms,
                n_conn_alt,
                n_conn_other,
                shared.path_dist_alt2,
                shared.union_vals.vals.path_dist_other,
                shared.union_vals.vals.conn_seps);

            totalE2 += score_inter_pairs_lk(
                tid,
                n_heavy_alt2,
                n_heavy_other,
                shared.coords_alt2,
                shared.union_vals.vals.coords_other,
                shared.params_alt2,
                shared.union_vals.vals.params_other,
                shared.heavy_inds_alt2,
                shared.union_vals.vals.heavy_inds_other,
                max_important_bond_separation,
                min_sep,
                alt_n_atoms2,
                neighb_n_atoms,
                n_conn_alt,
                n_conn_other,
                shared.path_dist_alt2,
                shared.union_vals.vals.path_dist_other,
                shared.union_vals.vals.conn_seps);

            // if (totalE1 != totalE2) {
            //   // printf("totalE discrepancy %f %f %d %d %d %d %d %d %d\n",
            //   // totalE1, totalE2, tid, alt_n_atoms1, alt_n_atoms2,
            //   i_n_atoms_to_load1, i_n_atoms_to_load2, alt_block_ind,
            //   neighb_block_ind); for (int k = 0; k < i_n_atoms_to_load1; ++k)
            //   {
            // 	if (shared.params_alt1[k].lj_radius !=
            // shared.params_alt2[k].lj_radius) { 	  printf("shared params
            // lj_radius discrepancy %d %d %d\n", tid, alt_block_ind,
            // neighb_block_ind);
            // 	}
            // 	if (shared.params_alt1[k].lj_wdepth !=
            // shared.params_alt2[k].lj_wdepth) { 	  printf("shared params
            // lj_wdepth discrepancy %d %d %d\n", tid, alt_block_ind,
            // neighb_block_ind);
            // 	}
            // 	if (shared.params_alt1[k].is_donor !=
            // shared.params_alt2[k].is_donor) { 	  printf("shared params
            // is_donor discrepancy %d %d %d\n", tid, alt_block_ind,
            // neighb_block_ind);
            // 	}
            // 	if (shared.params_alt1[k].is_hydroxyl !=
            // shared.params_alt2[k].is_hydroxyl) { 	  printf("shared params
            // is_hydroxyl discrepancy %d %d %d\n", tid, alt_block_ind,
            // neighb_block_ind);
            // 	}
            // 	if (shared.params_alt1[k].is_polarh !=
            // shared.params_alt2[k].is_polarh) { 	  printf("shared params
            // is_polarh discrepancy %d %d %d\n", tid, alt_block_ind,
            // neighb_block_ind);
            // 	}
            // 	if (shared.params_alt1[k].is_acceptor !=
            // shared.params_alt2[k].is_acceptor) { 	  printf("shared params
            // is_acceptor discrepancy %d %d %d\n", tid, alt_block_ind,
            // neighb_block_ind);
            // 	}
            //
            // 	if (shared.coords_alt1[3*k + 0] != shared.coords_alt2[3*k + 0])
            // { 	  printf("shared coords x discrepancy %d %d %d\n", tid,
            // alt_block_ind, neighb_block_ind);
            // 	}
            // 	if (shared.coords_alt1[3*k + 1] != shared.coords_alt2[3*k + 1])
            // { 	  printf("shared coords y discrepancy %d %d %d\n", tid,
            // alt_block_ind, neighb_block_ind);
            // 	}
            // 	if (shared.coords_alt1[3*k + 2] != shared.coords_alt2[3*k + 2])
            // { 	  printf("shared coords z discrepancy %d %d %d\n", tid,
            // alt_block_ind, neighb_block_ind);
            // 	}
            //
            // 	if (shared.path_dist_alt1[k] != shared.path_dist_alt2[k]) {
            // 	  printf("shared path dist discrepancy1 %f %f %d %d %d %d %d %d
            // %d\n", totalE1, totalE2, tid, alt_block_ind, neighb_block_ind, k,
            // i_n_atoms_to_load1, int(shared.path_dist_alt1[k]),
            // int(shared.path_dist_alt2[k]));
            // 	}
            // 	if (tid == 0 && shared.path_dist_alt1[32+k] !=
            // shared.path_dist_alt2[32 + k]) { 	  printf("shared path
            // dist discrepancy2 %f %f %d %d %d %d %d %d %d\n", totalE1,
            // totalE2, tid, alt_block_ind, neighb_block_ind, k,
            // i_n_atoms_to_load1, int(shared.path_dist_alt1[32+k]),
            // int(shared.path_dist_alt2[32+k]));
            // 	}
            //   }
            //
            // }
          }  // for j
        }    // for i
      } else {
        // alt_block_ind == neighb_block_ind

        int const n_iterations =
            (max(alt_n_atoms1, alt_n_atoms2) - 1) / TILE_SIZE + 1;

        for (int i = 0; i < n_iterations; ++i) {
          if (i != 0) {
            // make sure the calculations for the previous iteration
            // have completed before we overwrite the contents of
            // shared memory
            __syncthreads();
          }
          int const i_n_atoms_to_load1 =
              min(Int(TILE_SIZE), Int((alt_n_atoms1 - TILE_SIZE * i)));

          int const i_n_atoms_to_load2 =
              min(Int(TILE_SIZE), Int((alt_n_atoms2 - TILE_SIZE * i)));

          if (tid == 0) {
            shared.n_heavy_alt1 =
                block_type_n_heavy_atoms_in_tile[alt_block_type1][i];
            shared.n_heavy_alt2 =
                block_type_n_heavy_atoms_in_tile[alt_block_type2][i];
          }

          load_alt_coords_and_params_into_shared(
              rot_ind1,
              alt_n_atoms1,
              i_n_atoms_to_load1,
              alt_block_type1,
              tid,
              i,
              true,  // temp! new_context_ind,
              shared.coords_alt1,
              shared.params_alt1,
              shared.heavy_inds_alt1);
          load_alt_coords_and_params_into_shared(
              rot_ind2,
              alt_n_atoms2,
              i_n_atoms_to_load2,
              alt_block_type2,
              tid,
              i,
              true,  // temp! new_context_ind,
              shared.coords_alt2,
              shared.params_alt2,
              shared.heavy_inds_alt2);

          for (int j = i; j < n_iterations; ++j) {
            if (j != i) {
              // make sure calculations from the previous iteration have
              // completed before we overwrite the contents of shared
              // memory
              __syncthreads();
            }
            if (j != i) {
              if (tid == 0) {
                shared.union_vals.vals.n_heavy_other =
                    block_type_n_heavy_atoms_in_tile[alt_block_type1][j];
              }

              load_alt_coords_and_params_into_shared(
                  rot_ind1,
                  alt_n_atoms1,
                  i_n_atoms_to_load1,
                  alt_block_type1,
                  tid,
                  j,
                  new_context_ind,
                  shared.union_vals.vals.coords_other,
                  shared.union_vals.vals.params_other,
                  shared.union_vals.vals.heavy_inds_other);
            }
            // we are guaranteed to hit this syncthreads call; we must wait
            // here before reading from shared memory for the coordinates
            // in shared.coords_alt1 to be loaded, or if j != i, for the
            // coordinates in shared.union_vals.vals.coords_other to be loaded.
            __syncthreads();
            int const n_heavy_alt1 = shared.n_heavy_alt1;
            int const n_heavy_other1 =
                (i == j ? n_heavy_alt1 : shared.union_vals.vals.n_heavy_other);

            totalE1 += score_intra_pairs_lj(
                tid,
                i * TILE_SIZE,
                j * TILE_SIZE,
                shared.coords_alt1,
                (i == j ? shared.coords_alt1
                        : shared.union_vals.vals.coords_other),
                shared.params_alt1,
                (i == j ? shared.params_alt1
                        : shared.union_vals.vals.params_other),
                max_important_bond_separation,
                alt_block_type1,
                alt_n_atoms1);

            totalE1 += score_intra_pairs_lk(
                tid,
                i * TILE_SIZE,
                j * TILE_SIZE,
                n_heavy_alt1,
                n_heavy_other1,
                shared.coords_alt1,
                (i == j ? shared.coords_alt1
                        : shared.union_vals.vals.coords_other),
                shared.params_alt1,
                (i == j ? shared.params_alt1
                        : shared.union_vals.vals.params_other),
                shared.heavy_inds_alt1,
                (i == j ? shared.heavy_inds_alt1
                        : shared.union_vals.vals.heavy_inds_other),
                max_important_bond_separation,
                alt_block_type1,
                alt_n_atoms1);

          }  // for j

          for (int j = i; j < n_iterations; ++j) {
            if (j != i) {
              // Make sure previous calculations with rot_ind1 or
              // the previous iteration have finished
              // before we overwrite the contents of shared memory;
              // not necessary for the j == i iteration, since
              // we don't use the same shared memory arrays for that iteration
              __syncthreads();

              if (tid == 0) {
                shared.union_vals.vals.n_heavy_other =
                    block_type_n_heavy_atoms_in_tile[alt_block_type2][j];
              }

              load_alt_coords_and_params_into_shared(
                  rot_ind2,
                  alt_n_atoms2,
                  i_n_atoms_to_load2,
                  alt_block_type2,
                  tid,
                  j,
                  new_context_ind,
                  shared.union_vals.vals.coords_other,
                  shared.union_vals.vals.params_other,
                  shared.union_vals.vals.heavy_inds_other);
            }

            // make sure that all writes to shared memory have completed
            __syncthreads();
            int const n_heavy_alt2 = shared.n_heavy_alt2;
            int const n_heavy_other2 =
                (i == j ? n_heavy_alt2 : shared.union_vals.vals.n_heavy_other);

            totalE2 += score_intra_pairs_lj(
                tid,
                i * TILE_SIZE,
                j * TILE_SIZE,
                shared.coords_alt2,
                (i == j ? shared.coords_alt2
                        : shared.union_vals.vals.coords_other),
                shared.params_alt2,
                (i == j ? shared.params_alt2
                        : shared.union_vals.vals.params_other),
                max_important_bond_separation,
                alt_block_type2,
                alt_n_atoms2);

            totalE2 += score_intra_pairs_lk(
                tid,
                i * TILE_SIZE,
                j * TILE_SIZE,
                n_heavy_alt2,
                n_heavy_other2,
                shared.coords_alt2,
                (i == j ? shared.coords_alt2
                        : shared.union_vals.vals.coords_other),
                shared.params_alt2,
                (i == j ? shared.params_alt2
                        : shared.union_vals.vals.params_other),
                shared.heavy_inds_alt2,
                (i == j ? shared.heavy_inds_alt2
                        : shared.union_vals.vals.heavy_inds_other),
                max_important_bond_separation,
                alt_block_type2,
                alt_n_atoms2);

          }  // for j
        }    // for i
      }      // else

      // Make sure all energy calculations are complete before we overwrite
      // the neighbor-residue data in the shared memory union
      __syncthreads();

      Real const cta_totalE1 = reduce_t().reduce(
          tid, totalE1, shared.union_vals.reduce, nt, mgpu::plus_t<Real>());

      Real const cta_totalE2 = reduce_t().reduce(
          tid, totalE2, shared.union_vals.reduce, nt, mgpu::plus_t<Real>());

      if (tid == 0) {
        atomicAdd(&output[rot_ind1], cta_totalE1);
        atomicAdd(&output[rot_ind2], cta_totalE2);
      }

    }  // for ivt
  });

  // at::cuda::CUDAStream wrapped_stream = at::cuda::getDefaultCUDAStream();
  at::cuda::CUDAStream wrapped_stream = at::cuda::getStreamFromPool();
  // if (ljlk_stream == 0) {
  //   // cudaStreamCreate(&ljlk_stream);
  //   ljlk_stream = at::cuda::getStreamFromPool().stream();
  // }
  mgpu::standard_context_t context(wrapped_stream.stream());
  // mgpu::standard_context_t context(ljlk_stream);

  wait_on_annealer_event(wrapped_stream.stream(), annealer_event);

  int const n_ctas =
      (n_alternate_blocks * max_n_neighbors / 2 - 1) / launch_t::sm_ptx::vt + 1;

  // static bool first(true);
  // if (first) {
  //  std::cout << "nt " << launch_t::sm_ptx::nt << " vt " <<
  //  launch_t::sm_ptx::vt
  //            << std::endl;
  //  first = false;
  //}
  mgpu::cta_launch<launch_t>(eval_energies, n_ctas, context);
  record_scoring_event(wrapped_stream.stream(), score_event);

  // strictly unnecessary --
  // at::cuda::setCurrentCUDAStream(at::cuda::getDefaultCUDAStream());
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
class LJLKRPECudaCalc : public pack::sim_anneal::compiled::RPECalc {
 public:
  LJLKRPECudaCalc(
      TView<Vec<Real, 3>, 3, D> context_coords,
      TView<Int, 2, D> context_block_type,
      TView<Vec<Real, 3>, 2, D> alternate_coords,
      TView<Vec<Int, 3>, 1, D>
          alternate_ids,  // 0 == context id; 1 == block id; 2 == block type

      // which system does a given context belong to
      TView<Int, 1, D> context_system_ids,

      // dims: n-systems x max-n-blocks x max-n-blocks
      // Quick lookup: given the inds of two blocks, ask: what is the minimum
      // number of chemical bonds that separate any pair of atoms in those
      // blocks? If this minimum is greater than the crossover, then no further
      // logic for deciding whether two atoms in those blocks should have their
      // interaction energies calculated: all should. intentionally small to
      // (possibly) fit in constant cache
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
      TView<Int, 1, D> block_type_n_atoms,

      TView<Int, 2, D> block_type_n_heavy_atoms_in_tile,
      TView<Int, 2, D> block_type_heavy_atoms_in_tile,

      // what are the atom types for these atoms
      // Dimsize: n_block_types x max_n_atoms_per_block
      TView<Int, 2, D> block_type_atom_types,

      // how many inter-block chemical bonds are there
      // Dimsize: n_block_types
      TView<Int, 1, D> block_type_n_interblock_bonds,

      // what atoms form the inter-block chemical bonds
      // Dimsize: n_block_types x max_n_interblock_bonds
      TView<Int, 2, D> block_type_atoms_forming_chemical_bonds,

      // what is the path distance between pairs of atoms in the block
      // Dimsize: n_block_types x max_n_atoms_per_block x max_n_atoms_per_block
      TView<Int, 3, D> block_type_path_distance,
      //////////////////////

      // LJ parameters
      TView<LJLKTypeParams<Real>, 1, D> type_params,
      TView<LJGlobalParams<Real>, 1, D> global_params,
      TView<Real, 1, D> lj_lk_weights,
      TView<Real, 1, D> output,
      TView<int64_t, 1, tmol::Device::CPU> score_event,
      TView<int64_t, 1, tmol::Device::CPU> annealer_event)
      : context_coords_(context_coords),
        context_block_type_(context_block_type),
        alternate_coords_(alternate_coords),
        alternate_ids_(alternate_ids),
        context_system_ids_(context_system_ids),
        system_min_bond_separation_(system_min_bond_separation),
        system_inter_block_bondsep_(system_inter_block_bondsep),
        system_neighbor_list_(system_neighbor_list),
        block_type_n_atoms_(block_type_n_atoms),
        block_type_n_heavy_atoms_in_tile_(block_type_n_heavy_atoms_in_tile),
        block_type_heavy_atoms_in_tile_(block_type_heavy_atoms_in_tile),
        block_type_atom_types_(block_type_atom_types),
        block_type_n_interblock_bonds_(block_type_n_interblock_bonds),
        block_type_atoms_forming_chemical_bonds_(
            block_type_atoms_forming_chemical_bonds),
        block_type_path_distance_(block_type_path_distance),
        type_params_(type_params),
        global_params_(global_params),
        lj_lk_weights_(lj_lk_weights),
        output_(output),
        score_event_(score_event),
        annealer_event_(annealer_event) {}

  void calc_energies() override {
    clear_old_score_events(previously_created_events_);
    create_score_event(score_event_, previously_created_events_);

    LJLKRPEDispatch<DeviceDispatch, D, Real, Int>::f(
        context_coords_,
        context_block_type_,
        alternate_coords_,
        alternate_ids_,
        context_system_ids_,
        system_min_bond_separation_,
        system_inter_block_bondsep_,
        system_neighbor_list_,
        block_type_n_atoms_,
        block_type_n_heavy_atoms_in_tile_,
        block_type_heavy_atoms_in_tile_,
        block_type_atom_types_,
        block_type_n_interblock_bonds_,
        block_type_atoms_forming_chemical_bonds_,
        block_type_path_distance_,
        type_params_,
        global_params_,
        lj_lk_weights_,
        output_,
        score_event_,
        annealer_event_);
  }

  void finalize() override {
    // wait on all outstanding events and then delete them all
    sync_and_destroy_old_score_events(previously_created_events_);
  }

 private:
  TView<Vec<Real, 3>, 3, D> context_coords_;
  TView<Int, 2, D> context_block_type_;
  TView<Vec<Real, 3>, 2, D> alternate_coords_;
  TView<Vec<Int, 3>, 1, D> alternate_ids_;

  TView<Int, 1, D> context_system_ids_;
  TView<Int, 3, D> system_min_bond_separation_;

  TView<Int, 5, D> system_inter_block_bondsep_;

  TView<Int, 3, D> system_neighbor_list_;

  TView<Int, 1, D> block_type_n_atoms_;
  TView<Int, 2, D> block_type_n_heavy_atoms_in_tile_;
  TView<Int, 2, D> block_type_heavy_atoms_in_tile_;

  TView<Int, 2, D> block_type_atom_types_;

  TView<Int, 1, D> block_type_n_interblock_bonds_;

  TView<Int, 2, D> block_type_atoms_forming_chemical_bonds_;

  TView<Int, 3, D> block_type_path_distance_;

  // LJ parameters
  TView<LJLKTypeParams<Real>, 1, D> type_params_;
  TView<LJGlobalParams<Real>, 1, D> global_params_;
  TView<Real, 1, D> lj_lk_weights_;

  TView<Real, 1, D> output_;

  TView<int64_t, 1, tmol::Device::CPU> score_event_;
  TView<int64_t, 1, tmol::Device::CPU> annealer_event_;

  std::list<cudaEvent_t> previously_created_events_;
};

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto LJLKRPERegistratorDispatch<DeviceDispatch, D, Real, Int>::f(
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
    TView<Int, 1, D> block_type_n_atoms,

    TView<Int, 2, D> block_type_n_heavy_atoms_in_tile,
    TView<Int, 2, D> block_type_heavy_atoms_in_tile,

    // what are the atom types for these atoms
    // Dimsize: n_block_types x max_n_atoms_per_block
    TView<Int, 2, D> block_type_atom_types,

    // how many inter-block chemical bonds are there
    // Dimsize: n_block_types
    TView<Int, 1, D> block_type_n_interblock_bonds,

    // what atoms form the inter-block chemical bonds
    // Dimsize: n_block_types x max_n_interblock_bonds
    TView<Int, 2, D> block_type_atoms_forming_chemical_bonds,

    // what is the path distance between pairs of atoms in the block
    // Dimsize: n_block_types x max_n_atoms_per_block x max_n_atoms_per_block
    TView<Int, 3, D> block_type_path_distance,
    //////////////////////

    // LJ parameters
    TView<LJLKTypeParams<Real>, 1, D> type_params,
    TView<LJGlobalParams<Real>, 1, D> global_params,
    TView<Real, 1, D> lj_lk_weights,
    TView<Real, 1, D> output,

    TView<int64_t, 1, tmol::Device::CPU> score_event,
    TView<int64_t, 1, tmol::Device::CPU> annealer_event,

    TView<int64_t, 1, tmol::Device::CPU> annealer) -> void {
  using tmol::pack::sim_anneal::compiled::RPECalc;
  using tmol::pack::sim_anneal::compiled::SimAnnealer;

  int64_t annealer_uint = annealer[0];
  SimAnnealer *sim_annealer = reinterpret_cast<SimAnnealer *>(annealer_uint);
  std::shared_ptr<RPECalc> calc =
      std::make_shared<LJLKRPECudaCalc<DeviceDispatch, D, Real, Int>>(
          context_coords,
          context_block_type,
          alternate_coords,
          alternate_ids,
          context_system_ids,
          system_min_bond_separation,
          system_inter_block_bondsep,
          system_neighbor_list,
          block_type_n_atoms,
          block_type_n_heavy_atoms_in_tile,
          block_type_heavy_atoms_in_tile,
          block_type_atom_types,
          block_type_n_interblock_bonds,
          block_type_atoms_forming_chemical_bonds,
          block_type_path_distance,
          type_params,
          global_params,
          lj_lk_weights,
          output,
          score_event,
          annealer_event);

  sim_annealer->add_score_component(calc);
}

template struct LJLKRPEDispatch<ForallDispatch, tmol::Device::CUDA, float, int>;
template struct LJLKRPEDispatch<
    ForallDispatch,
    tmol::Device::CUDA,
    double,
    int>;
template struct LJLKRPERegistratorDispatch<
    ForallDispatch,
    tmol::Device::CUDA,
    float,
    int>;
template struct LJLKRPERegistratorDispatch<
    ForallDispatch,
    tmol::Device::CUDA,
    double,
    int>;

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
