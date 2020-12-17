#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/count_pair.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/tuple.hh>

#include <tmol/score/ljlk/potentials/lj.hh>
#include <tmol/score/ljlk/potentials/rotamer_pair_energy_lj.hh>

#include <chrono>

#include <tmol/score/common/forall_dispatch.cuda.impl.cuh>

#include <moderngpu/cta_load_balance.hxx>
#include <moderngpu/cta_reduce.hxx>
#include <moderngpu/cta_scan.hxx>
#include <moderngpu/cta_segreduce.hxx>
#include <moderngpu/cta_segscan.hxx>
#include <moderngpu/memory.hxx>
#include <moderngpu/search.hxx>
#include <moderngpu/transform.hxx>

// #include <tmol/score/ljlk/potentials/rotamer_pair_energy_lj.impl.hh>

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
auto LJRPEDispatch<DeviceDispatch, D, Real, Int>::f(
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
    TView<LJTypeParams<Real>, 1, D> type_params,
    TView<LJGlobalParams<Real>, 1, D> global_params,
    TView<Real, 1, D> lj_lk_weights) -> TPack<Real, 1, D> {
  int const n_systems = system_min_bond_separation.size(0);
  int const n_contexts = context_coords.size(0);
  int64_t const n_alternate_blocks = alternate_coords.size(0);
  int const max_n_blocks = context_coords.size(1);
  int64_t const max_n_atoms = context_coords.size(2);
  int const n_block_types = block_type_n_atoms.size(0);
  int const max_n_interblock_bonds =
      block_type_atoms_forming_chemical_bonds.size(1);
  int64_t const max_n_neighbors = system_neighbor_list.size(2);

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

  // auto wcts = std::chrono::system_clock::now();
  // clock_t start_time = clock();

  auto output_t = TPack<Real, 1, D>::zeros({n_alternate_blocks});
  auto output = output_t.view;
  auto count_t = TPack<int, 1, D>::zeros({1});
  auto count = count_t.view;

  using namespace mgpu;
  typedef launch_box_t<
      arch_20_cta<128, 4>,
      arch_35_cta<128, 4>,
      arch_52_cta<128, 4>>
      launch_t;

  // between one alternate rotamer and its neighbors in the surrounding context
  auto score_inter_pairs = ([=] MGPU_DEVICE(
                                int tid,
                                int alt_start_atom,
                                int neighb_start_atom,
                                Real *alt_coords,
                                Real *neighb_coords,
                                Int *alt_atom_type,
                                Int *neighb_atom_type,
                                int const max_important_bond_separation,
                                int const alt_block_ind,
                                int const neighb_block_ind,
                                int const alt_block_type,
                                int const neighb_block_type,

                                TensorAccessor<Int, 2, D> min_bond_separation,
                                TensorAccessor<Int, 4, D> inter_block_bondsep,

                                int const alt_n_atoms,
                                int const neighb_n_atoms) {
    Real score_total = 0;
    Real coord1[3];
    Real coord2[3];

    int const alt_remain = min(32, alt_n_atoms - alt_start_atom);
    int const neighb_remain = min(32, neighb_n_atoms - neighb_start_atom);

    int const n_pairs = alt_remain * neighb_remain;

    for (int i = alt_start_atom + tid; i < n_pairs; i += blockDim.x) {
      int const alt_atom_ind_local = i / neighb_remain;
      int const neighb_atom_ind_local = i % neighb_remain;
      int const alt_atom_ind = alt_atom_ind_local + alt_start_atom;
      int const neighb_atom_ind = neighb_atom_ind_local + neighb_start_atom;
      for (int j = 0; j < 3; ++j) {
        coord1[j] = alt_coords[3 * alt_atom_ind_local + j];
        coord2[j] = neighb_coords[3 * neighb_atom_ind_local + j];
      }
      int const atom_1_type = alt_atom_type[alt_atom_ind_local];
      int const atom_2_type = neighb_atom_type[neighb_atom_ind_local];

      int const separation =
          common::count_pair::CountPair<D, Int>::inter_block_separation(
              max_important_bond_separation,
              alt_block_ind,
              neighb_block_ind,
              alt_block_type,
              neighb_block_type,
              alt_atom_ind,
              neighb_atom_ind,
              min_bond_separation,
              inter_block_bondsep,
              block_type_n_interblock_bonds,
              block_type_atoms_forming_chemical_bonds,
              block_type_path_distance);

      Real dist = sqrt(
          (coord1[0] - coord2[0]) * (coord1[0] - coord2[0])
          + (coord1[1] - coord2[1]) * (coord1[1] - coord2[1])
          + (coord1[2] - coord2[2]) * (coord1[2] - coord2[2]));

      Real lj = lj_score<Real>::V(
          dist,
          separation,
          type_params[atom_1_type],
          type_params[atom_2_type],
          global_params[0]);
      lj *= lj_lk_weights[0];
      // if ( lj != 0 ) {
      //   printf("cuda %d %d %6.3f %6.3f %6.3f vs %6.3f %6.3f %6.3f e=
      //   %8.4f\n",
      //     alt_atom_ind, neighb_atom_ind,
      //     coord1[0], coord1[1], coord1[2],
      //     coord2[0], coord2[1], coord2[2],
      //     lj
      //   );
      // }

      score_total += lj;
    }
    return score_total;
  });

  // between one atoms within an alternate rotamer
  auto score_intra_pairs = ([=] MGPU_DEVICE(
                                int tid,
                                int start_atom1,
                                int start_atom2,
                                Real *coords1,
                                Real *coords2,
                                Int *atom_type1,
                                Int *atom_type2,
                                int const max_important_bond_separation,
                                int const block_type,
                                int const n_atoms) {
    Real score_total = 0;
    Real coord1[3];
    Real coord2[3];

    int const remain1 = min(32, n_atoms - start_atom1);
    int const remain2 = min(32, n_atoms - start_atom2);

    int const n_pairs = remain1 * remain2;

    for (int i = start_atom1 + tid; i < n_pairs; i += blockDim.x) {
      int const atom_ind_1_local = i / remain2;
      int const atom_ind_2_local = i % remain2;
      int const atom_ind_1 = atom_ind_1_local + start_atom1;
      int const atom_ind_2 = atom_ind_2_local + start_atom2;
      if (atom_ind_1 >= atom_ind_2) {
        continue;
      }

      for (int j = 0; j < 3; ++j) {
        coord1[j] = coords1[3 * atom_ind_1_local + j];
        coord2[j] = coords2[3 * atom_ind_2_local + j];
      }
      int const atom_1_type = atom_type1[atom_ind_1_local];
      int const atom_2_type = atom_type2[atom_ind_2_local];

      int const separation =
          block_type_path_distance[block_type][atom_ind_1][atom_ind_2];

      Real const dist = sqrt(
          (coord1[0] - coord2[0]) * (coord1[0] - coord2[0])
          + (coord1[1] - coord2[1]) * (coord1[1] - coord2[1])
          + (coord1[2] - coord2[2]) * (coord1[2] - coord2[2]));

      Real lj = lj_score<Real>::V(
          dist,
          separation,
          type_params[atom_1_type],
          type_params[atom_2_type],
          global_params[0]);
      lj *= lj_lk_weights[0];
      score_total += lj;
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
        Real coords1[32 * 3];
        Real coords2[32 * 3];
        Int atypes1[322];
        Int atypes2[322];
      } vals;
      typename reduce_t::storage_t reduce;
    } shared;

    Real *coords1 = shared.vals.coords1;
    Real *coords2 = shared.vals.coords2;
    Int *atom_type1 = shared.vals.atypes1;
    Int *atom_type2 =
        shared.vals
            .atypes2;  // reinterpret_cast<Int *>(&shared.coords[64 * 3 + 32]);

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
    int const n_iterations = (max_n_atoms - 1) / 32 + 1;

    Real totalE = 0;
    if (alt_block_ind != neighb_block_ind) {
      int const neighb_block_type =
          context_block_type[alt_context][neighb_block_ind];
      int const alt_n_atoms = block_type_n_atoms[alt_block_type];
      int const neighb_n_atoms = block_type_n_atoms[neighb_block_type];

      // Tile the sets of 32 atoms
      for (int i = 0; i < n_iterations; ++i) {
        __syncthreads();
        if (tid < 32 && i * 32 + tid < max_n_atoms) {
          int atid = i * 32 + tid;
          for (int j = 0; j < 3; ++j) {
            coords1[3 * tid + j] = alternate_coords[alt_ind][atid][j];
          }
          atom_type1[tid] = block_type_atom_types[alt_block_type][atid];
        }

        for (int j = 0; j < n_iterations; ++j) {
          __syncthreads();
          if (tid < 32 && j * 32 + tid < max_n_atoms) {
            int atid = j * 32 + tid;
            for (int k = 0; k < 3; ++k) {
              coords2[3 * tid + k] =
                  context_coords[alt_context][neighb_block_ind][atid][k];
            }
            atom_type2[tid] = block_type_atom_types[neighb_block_type][atid];
          }

          __syncthreads();

          // Now we will calculate ij pairs
          // printf("cuda score inter pairs %d %d %d %d\n", tid, cta, i, j);

          totalE += score_inter_pairs(
              tid,
              i * 32,
              j * 32,
              coords1,
              coords2,
              atom_type1,
              atom_type2,
              max_important_bond_separation,
              alt_block_ind,
              neighb_block_ind,
              alt_block_type,
              neighb_block_type,
              system_min_bond_separation[system],
              system_inter_block_bondsep[system],
              alt_n_atoms,
              neighb_n_atoms);
        }  // for j
      }    // for i
    } else {
      int const alt_n_atoms = block_type_n_atoms[alt_block_type];

      for (int i = 0; i < n_iterations; ++i) {
        __syncthreads();
        if (tid < 32 && i * 32 + tid < max_n_atoms) {
          int atid = i * 32 + tid;
          for (int j = 0; j < 3; ++j) {
            coords1[3 * tid + j] = alternate_coords[alt_ind][atid][j];
          }
          atom_type1[tid] = block_type_atom_types[alt_block_type][atid];
        }
        for (int j = i; j < n_iterations; ++j) {
          __syncthreads();
          if (j != i && tid < 32 && j * 32 + tid < max_n_atoms) {
            int atid = j * 32 + tid;
            for (int k = 0; k < 3; ++k) {
              coords2[3 * tid + k] = alternate_coords[alt_ind][atid][k];
            }
            atom_type2[tid] = block_type_atom_types[alt_block_type][atid];
          }
          __syncthreads();
          totalE += score_intra_pairs(
              tid,
              i * 32,
              j * 32,
              coords1,
              (i == j ? coords1 : coords2),
              atom_type1,
              (i == j ? atom_type1 : atom_type2),
              max_important_bond_separation,
              alt_block_type,
              alt_n_atoms);
        }  // for j
      }    // for i
    }      // else

    __syncthreads();

    Real all_reduce =
        reduce_t().reduce(tid, totalE, shared.reduce, nt, mgpu::plus_t<Real>());

    if (tid == 0 && all_reduce != 0) {
      atomicAdd(&output[alt_ind], all_reduce);
    }

    // accumulate<D, Real>::add_one_dst(output, alt_ind, totalE);
  });

  mgpu::standard_context_t context;
  mgpu::cta_launch<launch_t>(
      eval_energies, n_alternate_blocks * max_n_neighbors, context);

#ifdef __CUDACC__
  // float first;
  // cudaMemcpy(&first, &output[0], sizeof(float), cudaMemcpyDeviceToHost);
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
  return output_t;
}

template struct LJRPEDispatch<ForallDispatch, tmol::Device::CUDA, float, int>;
template struct LJRPEDispatch<ForallDispatch, tmol::Device::CUDA, double, int>;

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
