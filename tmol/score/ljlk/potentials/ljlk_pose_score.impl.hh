#pragma once

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
#include <tmol/score/common/warp_segreduce.hh>
#include <tmol/score/common/warp_stride_reduce.hh>

#include <tmol/score/ljlk/potentials/lj.hh>
#include <tmol/score/ljlk/potentials/lk_isotropic.hh>
#include <tmol/score/ljlk/potentials/ljlk_pose_score.hh>

#include <tmol/score/common/sphere_overlap.impl.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>

// Operator definitions; safe for CPU compilation
#include <moderngpu/operators.hxx>

#include <chrono>

// The maximum number of inter-residue chemical bonds
#define MAX_N_CONN 4
#define TILE_SIZE 32

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <typename Real>
class LJLKScoringData {
 public:
  TMOL_DEVICE_FUNC
  LJLKScoringData()
      : pose_ind(-1),
        block_type1(-1),
        block_type2(-1),
        block_ind1(-1),
        block_ind2(-1),
        block_coord_offset1(-1),
        block_coord_offset2(-1),
        max_important_bond_separation(-1),
        min_separation(-1),
        in_count_pair_striking_dist(0),
        coords1(0),
        coords2(0),
        params1(0),
        params2(0),
        heavy_inds1(0),  // shared
        heavy_inds2(0),  // shared
        n_atoms1(-1),
        n_atoms2(-1),
        n_heavy1(-1),
        n_heavy2(-1),
        n_conn1(-1),
        n_conn2(-1),
        path_dist1(0),  // shared
        path_dist2(0),  // shared
        conn_seps(0),
        global_params(),
        total_lj(0),
        total_lk(0) {}

  int pose_ind;
  int block_type1;
  int block_type2;
  int block_ind1;
  int block_ind2;
  int block_coord_offset1;
  int block_coord_offset2;
  int max_important_bond_separation;
  int min_separation;
  bool in_count_pair_striking_dist;
  Real *coords1;
  Real *coords2;
  LJLKTypeParams<Real> *params1;
  LJLKTypeParams<Real> *params2;
  unsigned char *heavy_inds1;  // shared
  unsigned char *heavy_inds2;  // shared
  int n_atoms1;
  int n_atoms2;
  int n_heavy1;
  int n_heavy2;
  int n_conn1;
  int n_conn2;
  unsigned char *path_dist1;  // shared
  unsigned char *path_dist2;  // shared
  unsigned char *conn_seps;
  LJGlobalParams<Real> global_params;
  Real total_lj;
  Real total_lk;
};

template <template <typename T> typename InterPairData, typename T>
class AllAtomPairSelector {
 public:
  static EIGEN_DEVICE_FUNC int n_atoms1(InterPairData<T> const &inter_dat) {
    return inter_dat.n_atoms1;
  }
  static EIGEN_DEVICE_FUNC int n_atoms2(InterPairData<T> const &inter_data) {
    return inter_data.n_atoms2;
  }
};

template <template <typename T> typename InterPairData, typename T>
class HeavyAtomPairSelector {
 public:
  static EIGEN_DEVICE_FUNC int n_atoms1(InterPairData<T> const &inter_dat) {
    return inter_dat.n_heavy1;
  }
  static EIGEN_DEVICE_FUNC int n_atoms2(InterPairData<T> const &inter_data) {
    return inter_data.n_heavy2;
  }
};

template <
    template <typename>
    typename InterEnergyData,
    template <template <typename> typename, typename>
    typename PairSelector,
    tmol::Device D,
    int TILE,
    int nt,
    typename Real,
    typename Int>
class InterResBlockEvaluation {
 public:
  template <typename AtomPairFunc>
  static TMOL_DEVICE_FUNC Real eval_interres_atom_pair(
      int tid,
      int start_atom1,
      int start_atom2,
      AtomPairFunc f,
      InterEnergyData<Real> const &inter_dat) {
    Real score_total = 0;
    int const n_remain1 = min(
        TILE,
        PairSelector<InterEnergyData, Real>::n_atoms1(inter_dat) - start_atom1);
    int const n_remain2 = min(
        TILE,
        PairSelector<InterEnergyData, Real>::n_atoms2(inter_dat) - start_atom2);
    int const n_pairs = n_remain1 * n_remain2;
    for (int i = tid; i < n_pairs; i += nt) {
      int const atom_tile_ind1 = i / n_remain2;
      int const atom_tile_ind2 = i % n_remain2;
      score_total += f(
          start_atom1, start_atom2, atom_tile_ind1, atom_tile_ind2, inter_dat);
    }
    return score_total;
  }
};

template <
    template <typename>
    typename IntraEnergyData,
    template <template <typename> typename, typename>
    typename PairSelector,
    tmol::Device D,
    int TILE,
    int nt,
    typename Real,
    typename Int>
class IntraResBlockEvaluation {
 public:
  template <typename AtomPairFunc>
  static TMOL_DEVICE_FUNC Real eval_intrares_atom_pairs(
      int tid,
      int start_atom1,
      int start_atom2,
      AtomPairFunc f,
      IntraEnergyData<Real> const &intra_dat) {
    Real score_total = 0;
    int const n_remain1 = min(
        TILE,
        PairSelector<IntraEnergyData, Real>::n_atoms1(intra_dat) - start_atom1);
    int const n_remain2 = min(
        TILE,
        PairSelector<IntraEnergyData, Real>::n_atoms2(intra_dat) - start_atom2);
    int const n_pairs = n_remain1 * n_remain2;
    for (int i = tid; i < n_pairs; i += nt) {
      int const atom_tile_ind1 = i / n_remain2;
      int const atom_tile_ind2 = i % n_remain2;
      int const atom_ind1 = atom_tile_ind1 + start_atom1;
      int const atom_ind2 = atom_tile_ind2 + start_atom2;

      // avoid calculating atom_ind1/atom_ind2 interaction twice
      if (atom_ind1 >= atom_ind2) {
        continue;
      }
      score_total += f(
          start_atom1, start_atom2, atom_tile_ind1, atom_tile_ind2, intra_dat);
    }
    return score_total;
  }
};

template <typename Real>
EIGEN_DEVICE_FUNC Vec<Real, 3> coord_from_shared(
    Real *coord_array, int atom_ind) {
  Vec<Real, 3> local_coord;
  for (int i = 0; i < 3; ++i) {
    local_coord[i] = coord_array[3 * atom_ind + i];
  }
  return local_coord;
}

template <int TILE, template <typename> typename InterEnergyData, typename Real>
EIGEN_DEVICE_FUNC int interres_count_pair_separation(
    InterEnergyData<Real> const &inter_dat,
    int atom_tile_ind1,
    int atom_tile_ind2) {
  int separation = inter_dat.min_separation;
  if (separation <= inter_dat.max_important_bond_separation) {
    separation = common::count_pair::shared_mem_inter_block_separation<TILE>(
        inter_dat.max_important_bond_separation,
        atom_tile_ind1,
        atom_tile_ind2,
        inter_dat.n_conn1,
        inter_dat.n_conn2,
        inter_dat.path_dist1,
        inter_dat.path_dist2,
        inter_dat.conn_seps);
  }
  return separation;
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    template <typename T>
    typename InterResScoringData,
    template <typename T>
    typename IntraResScoringData,
    typename Real,
    int TILE,
    typename SharedMemData,
    typename LoadConstInterFunc,
    typename LoadConstIntraFunc,
    typename LoadInterDatFunc1,
    typename LoadInterDatFunc2,
    typename LoadIntraDatFunc1,
    typename LoadIntraDatFunc2,
    typename LoadInterSharedDatFunc,
    typename LoadIntraSharedDatFunc,
    typename CalcInterFunc,
    typename CalcIntraFunc,
    typename StoreEnergyFunc>
TMOL_DEVICE_FUNC void eval_block_pair(
    SharedMemData &shared_data,
    int pose_ind,
    int block_ind1,
    int block_ind2,
    int block_type1,
    int block_type2,
    int n_atoms1,
    int n_atoms2,
    LoadConstInterFunc load_constant_interres_data,
    LoadInterDatFunc1 load_interres1_tile_data_to_shared,
    LoadInterDatFunc2 load_interres2_tile_data_to_shared,
    LoadInterSharedDatFunc load_interres_data_from_shared,
    CalcInterFunc eval_interres_atom_pair_scores,
    StoreEnergyFunc store_calculated_interres_energies,
    LoadConstIntraFunc load_constant_intrares_data,
    LoadIntraDatFunc1 load_intrares1_tile_data_to_shared,
    LoadIntraDatFunc2 load_intrares2_tile_data_to_shared,
    LoadIntraSharedDatFunc load_intrares_data_from_shared,
    CalcIntraFunc eval_intrares_atom_pair_scores,
    StoreEnergyFunc store_calculated_intrares_energies) {
  // printf("starting %d %d\n", block_ind1, block_ind2);
  if (block_ind1 != block_ind2) {
    // Step 1: load any data that is consistent across all tile pairs
    InterResScoringData<Real> interres_data;
    // printf("calling load_constant_interres_data\n");
    load_constant_interres_data(
        pose_ind,
        block_ind1,
        block_ind2,
        block_type1,
        block_type2,
        n_atoms1,
        n_atoms2,
        interres_data,
        shared_data);

    // Step 2: Tile data loading
    int const n_iterations1 = (n_atoms1 - 1) / TILE + 1;
    int const n_iterations2 = (n_atoms2 - 1) / TILE + 1;

    for (int i = 0; i < n_iterations1; ++i) {
      // Make sure the constant inter-res data has been loaded
      // if i is 0 before loading the tile data in, and make
      // sure that the calculations from the previous iteration
      // have completed before overwriting the data in shared
      // memory if i > 0
      DeviceDispatch<D>::synchronize_workgroup();

      int const i_n_atoms_to_load1 =
          max(0, min(int(TILE), int((n_atoms1 - TILE * i))));
      // printf("calling load_interres1_tile_data_to_shared\n");
      load_interres1_tile_data_to_shared(
          i, TILE * i, i_n_atoms_to_load1, interres_data, shared_data);
      for (int j = 0; j < n_iterations2; ++j) {
        if (j != 0) {
          // We can safely move into the loading of tile data for j == 0
          // because we synchronized at the top of the "for i" loop above
          // but for j > 0, we have to wait for the calculations from the
          // previous iteration to complete  before overwriting the data
          // in shared memory
          DeviceDispatch<D>::synchronize_workgroup();
        }
        int j_n_atoms_to_load2 = min(int(TILE), int((n_atoms2 - TILE * j)));
        // printf("calling load_interres2_tile_data_to_shared\n");
        load_interres2_tile_data_to_shared(
            j, TILE * j, j_n_atoms_to_load2, interres_data, shared_data);

        // Wait for all loading to complete before moving on to any
        // energy calculations;
        DeviceDispatch<D>::synchronize_workgroup();

        // Step 3: initialize combo shared/
        // printf("calling load_interres_data_from_shared\n");
        load_interres_data_from_shared(i, j, shared_data, interres_data);

        // printf("calling eval_interres_atom_pair_scores\n");
        eval_interres_atom_pair_scores(interres_data, i * TILE, j * TILE);
      }
    }
    DeviceDispatch<D>::synchronize_workgroup();
    store_calculated_interres_energies(interres_data, shared_data);

  } else {
    // Step 1: load any data that is consistent across all tile pairs
    IntraResScoringData<Real> intrares_data;
    // printf("calling load_constant_intrares_data\n");
    load_constant_intrares_data(
        pose_ind,
        block_ind1,
        block_type1,
        n_atoms1,
        intrares_data,
        shared_data);

    // Step 2: Tile data loading
    int const n_iterations = (n_atoms1 - 1) / TILE + 1;
    for (int i = 0; i < n_iterations; ++i) {
      // make sure the calculatixons for the previous iteration
      // or from the tile-independent load have completed before
      // we overwrite the contents of shared memory
      DeviceDispatch<D>::synchronize_workgroup();
      int const i_n_atoms_to_load1 = min(int(TILE), int((n_atoms1 - TILE * i)));
      // printf("calling load_intrares1_tile_data_to_shared\n");
      load_intrares1_tile_data_to_shared(
          i, TILE * i, i_n_atoms_to_load1, intrares_data, shared_data);
      for (int j = i; j < n_iterations; ++j) {
        int const j_n_atoms_to_load2 =
            min(int(TILE), int((n_atoms1 - TILE * j)));

        if (j != i) {
          // make sure calculations from the previous iteration have
          // completed before we overwrite the contents of shared
          // memory
          DeviceDispatch<D>::synchronize_workgroup();
          // printf("calling load_intrares2_tile_data_to_shared\n");
          load_intrares2_tile_data_to_shared(
              j, TILE * j, j_n_atoms_to_load2, intrares_data, shared_data);
        }
        // Make sure that all the data has been loaded into shared memory
        // before we start any calculations
        DeviceDispatch<D>::synchronize_workgroup();
        // printf("calling load_intrares_data_from_shared\n");
        load_intrares_data_from_shared(i, j, shared_data, intrares_data);
        // printf("calling eval_intrares_atom_pair_scores\n");
        eval_intrares_atom_pair_scores(intrares_data, i * TILE, j * TILE);
      }
    }
    DeviceDispatch<D>::synchronize_workgroup();
    store_calculated_intrares_energies(intrares_data, shared_data);
  }
};

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

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;

  auto lj_atom_energy_and_derivs = ([=] TMOL_DEVICE_FUNC(
                                        int atom_tile_ind1,
                                        int atom_tile_ind2,
                                        int start_atom1,
                                        int start_atom2,
                                        LJLKScoringData<Real> const &score_dat,
                                        int cp_separation) {
    Real3 coord1 = coord_from_shared(score_dat.coords1, atom_tile_ind1);
    Real3 coord2 = coord_from_shared(score_dat.coords2, atom_tile_ind2);

    auto dist_r = distance<Real>::V_dV(coord1, coord2);
    auto &dist = dist_r.V;
    auto &ddist_dat1 = dist_r.dV_dA;
    auto &ddist_dat2 = dist_r.dV_dB;
    auto lj = lj_score<Real>::V_dV(
        dist,
        cp_separation,
        score_dat.params1[atom_tile_ind1].lj_params(),
        score_dat.params2[atom_tile_ind2].lj_params(),
        score_dat.global_params);

    // all threads accumulate derivatives for atom 1 to global memory
    Vec<Real, 3> lj_dxyz_at1 = lj.dV_ddist * ddist_dat1;
    for (int j = 0; j < 3; ++j) {
      if (lj_dxyz_at1[j] != 0) {
        accumulate<D, Real>::add(
            dV_dcoords[0][score_dat.pose_ind]
                      [score_dat.block_coord_offset1 + atom_tile_ind1
                       + start_atom1][j],
            lj_dxyz_at1[j]);
      }
    }

    // all threads accumulate derivatives for atom 2 to global memory
    Vec<Real, 3> lj_dxyz_at2 = lj.dV_ddist * ddist_dat2;
    for (int j = 0; j < 3; ++j) {
      if (lj_dxyz_at2[j] != 0) {
        accumulate<D, Real>::add(
            dV_dcoords[0][score_dat.pose_ind]
                      [score_dat.block_coord_offset2 + atom_tile_ind2
                       + start_atom2][j],
            lj_dxyz_at2[j]);
      }
    }
    return lj.V;
  });

  auto lk_atom_energy_and_derivs = ([=] TMOL_DEVICE_FUNC(
                                        int atom_tile_ind1,
                                        int atom_tile_ind2,
                                        int start_atom1,
                                        int start_atom2,
                                        LJLKScoringData<Real> const &score_dat,
                                        int cp_separation) {
    Real3 coord1 = coord_from_shared(score_dat.coords1, atom_tile_ind1);
    Real3 coord2 = coord_from_shared(score_dat.coords2, atom_tile_ind2);

    auto dist_r = distance<Real>::V_dV(coord1, coord2);
    auto &dist = dist_r.V;
    auto &ddist_dat1 = dist_r.dV_dA;
    auto &ddist_dat2 = dist_r.dV_dB;
    auto lk = lk_isotropic_score<Real>::V_dV(
        dist,
        cp_separation,
        score_dat.params1[atom_tile_ind1].lk_params(),
        score_dat.params2[atom_tile_ind2].lk_params(),
        score_dat.global_params);

    Vec<Real, 3> lk_dxyz_at1 = lk.dV_ddist * ddist_dat1;
    for (int j = 0; j < 3; ++j) {
      if (lk_dxyz_at1[j] != 0) {
        accumulate<D, Real>::add(
            dV_dcoords[1][score_dat.pose_ind]
                      [score_dat.block_coord_offset1 + atom_tile_ind1
                       + start_atom1][j],
            lk_dxyz_at1[j]);
      }
    }

    Vec<Real, 3> lk_dxyz_at2 = lk.dV_ddist * ddist_dat2;
    for (int j = 0; j < 3; ++j) {
      if (lk_dxyz_at2[j] != 0) {
        accumulate<D, Real>::add(
            dV_dcoords[1][score_dat.pose_ind]
                      [score_dat.block_coord_offset2 + atom_tile_ind2
                       + start_atom2][j],
            lk_dxyz_at2[j]);
      }
    }
    return lk.V;
  });

  auto score_inter_lj_atom_pair = ([=] TMOL_DEVICE_FUNC(
                                       int start_atom1,
                                       int start_atom2,
                                       int atom_tile_ind1,
                                       int atom_tile_ind2,
                                       LJLKScoringData<Real> const &inter_dat) {
    int separation = interres_count_pair_separation<TILE_SIZE>(
        inter_dat, atom_tile_ind1, atom_tile_ind2);
    return lj_atom_energy_and_derivs(
        atom_tile_ind1,
        atom_tile_ind2,
        start_atom1,
        start_atom2,
        inter_dat,
        separation);
  });

  auto score_intra_lj_atom_pair = ([=] TMOL_DEVICE_FUNC(
                                       int start_atom1,
                                       int start_atom2,
                                       int atom_tile_ind1,
                                       int atom_tile_ind2,
                                       LJLKScoringData<Real> const &intra_dat) {
    int const atom_ind1 = start_atom1 + atom_tile_ind1;
    int const atom_ind2 = start_atom2 + atom_tile_ind2;

    int const separation =
        block_type_path_distance[intra_dat.block_type1][atom_ind1][atom_ind2];
    return lj_atom_energy_and_derivs(
        atom_tile_ind1,
        atom_tile_ind2,
        start_atom1,
        start_atom2,
        intra_dat,
        separation);
  });

  auto score_inter_lk_atom_pair = ([=] TMOL_DEVICE_FUNC(
                                       int start_atom1,
                                       int start_atom2,
                                       int atom_heavy_tile_ind1,
                                       int atom_heavy_tile_ind2,
                                       LJLKScoringData<Real> const &inter_dat) {
    int const atom_tile_ind1 = inter_dat.heavy_inds1[atom_heavy_tile_ind1];
    int const atom_tile_ind2 = inter_dat.heavy_inds2[atom_heavy_tile_ind2];

    int separation = interres_count_pair_separation<TILE_SIZE>(
        inter_dat, atom_tile_ind1, atom_tile_ind2);

    return lk_atom_energy_and_derivs(
        atom_tile_ind1,
        atom_tile_ind2,
        start_atom1,
        start_atom2,
        inter_dat,
        separation);
  });

  auto score_intra_lk_atom_pair = ([=] TMOL_DEVICE_FUNC(
                                       int start_atom1,
                                       int start_atom2,
                                       int atom_heavy_tile_ind1,
                                       int atom_heavy_tile_ind2,
                                       LJLKScoringData<Real> const &intra_dat) {
    int const atom_tile_ind1 = intra_dat.heavy_inds1[atom_heavy_tile_ind1];
    int const atom_tile_ind2 = intra_dat.heavy_inds2[atom_heavy_tile_ind2];
    int const atom_ind1 = start_atom1 + atom_tile_ind1;
    int const atom_ind2 = start_atom2 + atom_tile_ind2;

    int const separation =
        block_type_path_distance[intra_dat.block_type1][atom_ind1][atom_ind2];
    return lk_atom_energy_and_derivs(
        atom_tile_ind1,
        atom_tile_ind2,
        start_atom1,
        start_atom2,
        intra_dat,
        separation);
  });

  auto eval_energies = ([=] TMOL_DEVICE_FUNC(int cta) {
    // Define nt and reduce_t
    CTA_REAL_REDUCE_T_TYPEDEF;

    auto load_block_coords_and_params_into_shared =
        ([=] TMOL_DEVICE_FUNC(
             int pose_ind,
             int block_coord_offset,
             int n_atoms_to_load,
             int block_type,
             int tile_ind,
             Real *__restrict__ shared_coords,
             LJLKTypeParams<Real> *__restrict__ params,
             unsigned char *__restrict__ heavy_inds) {
          DeviceDispatch<D>::template copy_contiguous_data<nt, 3>(
              shared_coords,
              reinterpret_cast<Real *>(
                  &coords[pose_ind][block_coord_offset + TILE_SIZE * tile_ind]),
              n_atoms_to_load * 3);
          auto copy_atom_types = ([=](int tid) {
            if (tid < TILE_SIZE) {
              if (tid < n_atoms_to_load) {
                int const atid = TILE_SIZE * tile_ind + tid;
                int const attype = block_type_atom_types[block_type][atid];
                if (attype >= 0) {
                  params[tid] = type_params[attype];
                }
                heavy_inds[tid] =
                    block_type_heavy_atoms_in_tile[block_type][atid];
              }
            }
          });
          DeviceDispatch<D>::template for_each_in_workgroup<nt>(
              copy_atom_types);
        });

    auto load_block_into_shared =
        ([=] TMOL_DEVICE_FUNC(
             int pose_ind,
             int block_coord_offset,
             int n_atoms,
             int n_atoms_to_load,
             int block_type,
             int n_conn,
             int tile_ind,
             bool count_pair_striking_dist,
             unsigned char *__restrict__ conn_ats,
             Real *__restrict__ shared_coords,
             LJLKTypeParams<Real> *__restrict__ params,
             unsigned char *__restrict__ heavy_inds,
             unsigned char *__restrict__ path_dist  // to conn
         ) {
          load_block_coords_and_params_into_shared(
              pose_ind,
              block_coord_offset,
              n_atoms_to_load,
              block_type,
              tile_ind,
              shared_coords,
              params,
              heavy_inds);

          auto copy_path_dists = ([=](int tid) {
            if (tid < n_atoms_to_load && count_pair_striking_dist) {
              int const atid = TILE_SIZE * tile_ind + tid;
              for (int j = 0; j < n_conn; ++j) {
                unsigned char ij_path_dist =
                    block_type_path_distance[block_type][conn_ats[j]][atid];
                path_dist[j * TILE_SIZE + tid] = ij_path_dist;
              }
            }
          });
          DeviceDispatch<D>::template for_each_in_workgroup<nt>(
              copy_path_dists);
        });

    SHARED_MEMORY union shared_mem_union {
      shared_mem_union() {}
      struct {
        Real coords1[TILE_SIZE * 3];  // 786 bytes for coords
        Real coords2[TILE_SIZE * 3];
        LJLKTypeParams<Real> params1[TILE_SIZE];  // 1536 bytes for params
        LJLKTypeParams<Real> params2[TILE_SIZE];
        // unsigned char n_heavy1;
        // unsigned char n_heavy2;
        unsigned char heavy_inds1[TILE_SIZE];
        unsigned char heavy_inds2[TILE_SIZE];
        unsigned char conn_ats1[MAX_N_CONN];  // 8 bytes
        unsigned char conn_ats2[MAX_N_CONN];
        unsigned char path_dist1[MAX_N_CONN * TILE_SIZE];  // 256 bytes
        unsigned char path_dist2[MAX_N_CONN * TILE_SIZE];
        unsigned char conn_seps[MAX_N_CONN * MAX_N_CONN];  // 64 bytes

      } m;

      CTA_REAL_REDUCE_T_VARIABLE;

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
    // int const block_coord_offset1 =
    //     pose_stack_block_coord_offset[pose_ind][block_ind1];
    // int const block_coord_offset2 =
    //     pose_stack_block_coord_offset[pose_ind][block_ind2];

    auto load_constant_interres_data = ([=](int pose_ind,
                                            int block_ind1,
                                            int block_ind2,
                                            int block_type1,
                                            int block_type2,
                                            int n_atoms1,
                                            int n_atoms2,
                                            LJLKScoringData<Real> &inter_dat,
                                            shared_mem_union &shared) {
      inter_dat.pose_ind = pose_ind;
      inter_dat.block_type1 = block_type1;
      inter_dat.block_type2 = block_type2;
      inter_dat.block_ind1 = block_ind1;
      inter_dat.block_ind2 = block_ind2;
      inter_dat.block_coord_offset1 =
          pose_stack_block_coord_offset[pose_ind][block_ind1];
      inter_dat.block_coord_offset2 =
          pose_stack_block_coord_offset[pose_ind][block_ind2];
      inter_dat.max_important_bond_separation = max_important_bond_separation;
      inter_dat.min_separation =
          pose_stack_min_bond_separation[pose_ind][block_ind1][block_ind2];
      inter_dat.in_count_pair_striking_dist =
          inter_dat.min_separation <= max_important_bond_separation;
      inter_dat.n_atoms1 = n_atoms1;
      inter_dat.n_atoms2 = n_atoms2;
      inter_dat.n_conn1 = block_type_n_interblock_bonds[block_type1];
      inter_dat.n_conn2 = block_type_n_interblock_bonds[block_type2];

      // set the pointers in inter_dat to point at the shared-memory arrays
      inter_dat.coords1 = shared.m.coords1;
      inter_dat.coords2 = shared.m.coords2;
      inter_dat.params1 = shared.m.params1;
      inter_dat.params2 = shared.m.params2;
      inter_dat.heavy_inds1 = shared.m.heavy_inds1;
      inter_dat.heavy_inds2 = shared.m.heavy_inds2;
      // inter_dat.conn_ats1 = shared.m.conn_ats1;
      // inter_dat.conn_ats2 = shared.m.conn_ats2;
      inter_dat.path_dist1 = shared.m.path_dist1;
      inter_dat.path_dist2 = shared.m.path_dist2;
      inter_dat.conn_seps = shared.m.conn_seps;

      // Count pair setup that does not depend on which tile we are operating on
      if (inter_dat.in_count_pair_striking_dist) {
        // Load data into shared arrays
        auto load_count_pair_conn_at_data = ([&](int tid) {
          if (tid < inter_dat.n_conn1) {
            shared.m.conn_ats1[tid] =
                block_type_atoms_forming_chemical_bonds[block_type1][tid];
          }
          if (tid < inter_dat.n_conn2) {
            shared.m.conn_ats2[tid] =
                block_type_atoms_forming_chemical_bonds[block_type2][tid];
          }

          // NOTE MAX_N_CONN ^ 2 <= 32; limit MAX_N_CONN = 5 before this code
          // would need to be adjusted
          if (tid < inter_dat.n_conn1 * inter_dat.n_conn2) {
            int conn1 = tid / inter_dat.n_conn2;
            int conn2 = tid % inter_dat.n_conn2;
            shared.m.conn_seps[tid] =
                pose_stack_inter_block_bondsep[pose_ind][block_ind1][block_ind2]
                                              [conn1][conn2];
          }
        });
        // On CPU: a for loop executed once; on GPU threads within the workgroup
        // working in parallel will just continue to work in parallel
        DeviceDispatch<D>::template for_each_in_workgroup<nt>(
            load_count_pair_conn_at_data);
      }

      // Final data members
      inter_dat.global_params = global_params[0];
      inter_dat.total_lj = 0;
      inter_dat.total_lk = 0;
    });

    auto load_interres1_tile_data_to_shared =
        ([=](int tile_ind,
             int start_atom1,
             int n_atoms_to_load1,
             LJLKScoringData<Real> &inter_dat,
             shared_mem_union &shared) {
          inter_dat.n_heavy1 =
              block_type_n_heavy_atoms_in_tile[inter_dat.block_type1][tile_ind];

          load_block_into_shared(
              inter_dat.pose_ind,
              inter_dat.block_coord_offset1,
              inter_dat.n_atoms1,
              n_atoms_to_load1,
              inter_dat.block_type1,
              inter_dat.n_conn1,
              tile_ind,
              inter_dat.in_count_pair_striking_dist,
              shared.m.conn_ats1,
              shared.m.coords1,
              shared.m.params1,
              shared.m.heavy_inds1,
              shared.m.path_dist1);
        });

    auto load_interres2_tile_data_to_shared =
        ([=](int tile_ind,
             int start_atom2,
             int n_atoms_to_load2,
             LJLKScoringData<Real> &inter_dat,
             shared_mem_union &shared) {
          inter_dat.n_heavy2 =
              block_type_n_heavy_atoms_in_tile[inter_dat.block_type2][tile_ind];

          load_block_into_shared(
              inter_dat.pose_ind,
              inter_dat.block_coord_offset2,
              inter_dat.n_atoms2,
              n_atoms_to_load2,
              inter_dat.block_type2,
              inter_dat.n_conn2,
              tile_ind,
              inter_dat.in_count_pair_striking_dist,
              shared.m.conn_ats2,
              shared.m.coords2,
              shared.m.params2,
              shared.m.heavy_inds2,
              shared.m.path_dist2);
        });

    auto load_interres_data_from_shared =
        ([=](int, int, shared_mem_union &, LJLKScoringData<Real> &) {
          // ?? no op ??
        });

    // Evaluate both the LJ and LK scores in separate dispatches
    // over all atoms in the tile and the subset of heavy atoms in
    // the tile
    auto eval_interres_atom_pair_scores = ([=](LJLKScoringData<Real> &inter_dat,
                                               int start_atom1,
                                               int start_atom2) {
      assert(inter_dat.pose_ind != -1);
      assert(inter_dat.block_type1 != -1);
      assert(inter_dat.block_type2 != -1);
      assert(inter_dat.block_ind1 != -1);
      assert(inter_dat.block_ind2 != -1);
      assert(inter_dat.block_coord_offset1 != -1);
      assert(inter_dat.block_coord_offset2 != -1);
      assert(inter_dat.coords1 != 0);
      assert(inter_dat.coords2 != 0);
      assert(inter_dat.params1 != 0);
      assert(inter_dat.params2 != 0);
      assert(inter_dat.heavy_inds1 != 0);
      assert(inter_dat.heavy_inds2 != 0);
      assert(inter_dat.n_atoms1 != -1);
      assert(inter_dat.n_atoms2 != -1);
      assert(inter_dat.n_heavy1 != -1);
      assert(inter_dat.n_heavy2 != -1);
      assert(inter_dat.n_conn1 != -1);
      assert(inter_dat.n_conn2 != -1);

      auto eval_scores_for_atom_pairs = ([&](int tid) {
        inter_dat.total_lj += InterResBlockEvaluation<
            LJLKScoringData,
            AllAtomPairSelector,
            D,
            TILE_SIZE,
            nt,
            Real,
            Int>::
            eval_interres_atom_pair(
                tid,
                start_atom1,
                start_atom2,
                score_inter_lj_atom_pair,
                inter_dat);

        inter_dat.total_lk += InterResBlockEvaluation<
            LJLKScoringData,
            HeavyAtomPairSelector,
            D,
            TILE_SIZE,
            nt,
            Real,
            Int>::
            eval_interres_atom_pair(
                tid,
                start_atom1,
                start_atom2,
                score_inter_lk_atom_pair,
                inter_dat);
      });

      // The work: On GPU threads work independently, on CPU, this will be a
      // for loop
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(
          eval_scores_for_atom_pairs);
    });

    auto store_calculated_energies = ([=](LJLKScoringData<Real> &score_dat,
                                          shared_mem_union &shared) {
      auto reduce_energies = ([&](int tid) {
        Real const cta_total_lj =
            DeviceDispatch<D>::template reduce_in_workgroup<TILE_SIZE>(
                score_dat.total_lj, shared, mgpu::plus_t<Real>());
        Real const cta_total_lk =
            DeviceDispatch<D>::template reduce_in_workgroup<TILE_SIZE>(
                score_dat.total_lk, shared, mgpu::plus_t<Real>());

        if (tid == 0) {
          // printf("Storing energy %d %d %f %f\n", score_dat.block_ind1,
          // score_dat.block_ind2, cta_total_lj, cta_total_lk);
          accumulate<D, Real>::add(output[0][score_dat.pose_ind], cta_total_lj);
          accumulate<D, Real>::add(output[1][score_dat.pose_ind], cta_total_lk);
        }
      });
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(reduce_energies);
    });

    auto load_constant_intrares_data = ([=](int pose_ind,
                                            int block_ind1,
                                            int block_type1,
                                            int n_atoms1,
                                            LJLKScoringData<Real> &intra_dat,
                                            shared_mem_union &shared) {
      intra_dat.pose_ind = pose_ind;
      intra_dat.block_type1 = block_type1;
      intra_dat.block_type2 = block_type1;
      intra_dat.block_ind1 = block_ind1;
      intra_dat.block_ind2 = block_ind1;
      intra_dat.block_coord_offset1 =
          pose_stack_block_coord_offset[pose_ind][block_ind1];
      intra_dat.block_coord_offset2 = intra_dat.block_coord_offset1;
      intra_dat.max_important_bond_separation = max_important_bond_separation;
      intra_dat.min_separation =
          0;  // Intra-residue guarantees the need for count-pair calcs
      intra_dat.in_count_pair_striking_dist = true;

      intra_dat.n_atoms1 = n_atoms1;
      intra_dat.n_atoms2 = n_atoms1;
      intra_dat.n_conn1 = block_type_n_interblock_bonds[block_type1];
      intra_dat.n_conn2 = intra_dat.n_conn1;

      // depends on tile pair! // set the pointers in intra_dat to point at the
      // shared-memory arrays depends on tile pair! intra_dat.coords1 =
      // shared.m.coords1; depends on tile pair! intra_dat.coords2 =
      // shared.m.coords2; depends on tile pair! intra_dat.params1 =
      // shared.m.params1; depends on tile pair! intra_dat.params2 =
      // shared.m.params1; depends on tile pair! intra_dat.heavy_inds1 =
      // shared.m.heavy_inds1; depends on tile pair! intra_dat.heavy_inds2 =
      // shared.m.heavy_inds2;

      // these count pair arrays are not going to be used
      intra_dat.path_dist1 = 0;  // shared.m.path_dist1;
      intra_dat.path_dist2 = 0;  // shared.m.path_dist2;
      intra_dat.conn_seps = 0;   // shared.m.conn_seps;

      // Final data members
      intra_dat.global_params = global_params[0];
      intra_dat.total_lj = 0;
      intra_dat.total_lk = 0;
    });

    auto load_intrares1_tile_data_to_shared =
        ([=](int tile_ind,
             int start_atom1,
             int n_atoms_to_load1,
             LJLKScoringData<Real> &intra_dat,
             shared_mem_union &shared) {
          intra_dat.n_heavy1 =
              block_type_n_heavy_atoms_in_tile[intra_dat.block_type1][tile_ind];

          load_block_into_shared(
              intra_dat.pose_ind,
              intra_dat.block_coord_offset1,
              intra_dat.n_atoms1,
              n_atoms_to_load1,
              intra_dat.block_type1,
              intra_dat.n_conn1,
              tile_ind,
              intra_dat.in_count_pair_striking_dist,
              shared.m.conn_ats1,
              shared.m.coords1,
              shared.m.params1,
              shared.m.heavy_inds1,
              shared.m.path_dist1);
        });

    auto load_intrares2_tile_data_to_shared =
        ([=](int tile_ind,
             int start_atom2,
             int n_atoms_to_load2,
             LJLKScoringData<Real> &intra_dat,
             shared_mem_union &shared) {
          intra_dat.n_heavy2 =
              block_type_n_heavy_atoms_in_tile[intra_dat.block_type2][tile_ind];

          load_block_into_shared(
              intra_dat.pose_ind,
              intra_dat.block_coord_offset2,
              intra_dat.n_atoms2,
              n_atoms_to_load2,
              intra_dat.block_type2,
              intra_dat.n_conn2,
              tile_ind,
              intra_dat.in_count_pair_striking_dist,
              shared.m.conn_ats2,
              shared.m.coords2,
              shared.m.params2,
              shared.m.heavy_inds2,
              shared.m.path_dist2);
        });

    auto load_intrares_data_from_shared =
        ([=](int tile_ind1,
             int tile_ind2,
             shared_mem_union &shared,
             LJLKScoringData<Real> &intra_dat) {
          // set the pointers in intra_dat to point at the shared-memory arrays
          // If we are evaluating the energies between atoms in the same tile
          // then only the "1" shared-memory arrays will be loaded with data;
          // we will point the "2" memory pointers at the "1" arrays
          bool same_tile = tile_ind1 == tile_ind2;
          intra_dat.n_heavy2 =
              (same_tile ? intra_dat.n_heavy1 : intra_dat.n_heavy2);
          intra_dat.coords1 = shared.m.coords1;
          intra_dat.coords2 = (same_tile ? shared.m.coords1 : shared.m.coords2);
          intra_dat.params1 = shared.m.params1;
          intra_dat.params2 = (same_tile ? shared.m.params1 : shared.m.params2);
          intra_dat.heavy_inds1 = shared.m.heavy_inds1;
          intra_dat.heavy_inds2 =
              (same_tile ? shared.m.heavy_inds1 : shared.m.heavy_inds2);
        });

    // Evaluate both the LJ and LK scores in separate dispatches
    // over all atoms in the tile and the subset of heavy atoms in
    // the tile
    auto eval_intrares_atom_pair_scores = ([=](LJLKScoringData<Real> &intra_dat,
                                               int start_atom1,
                                               int start_atom2) {
      auto eval_scores_for_atom_pairs = ([&](int tid) {
        intra_dat.total_lj += IntraResBlockEvaluation<
            LJLKScoringData,
            AllAtomPairSelector,
            D,
            TILE_SIZE,
            nt,
            Real,
            Int>::
            eval_intrares_atom_pairs(
                tid,
                start_atom1,
                start_atom2,
                score_intra_lj_atom_pair,
                intra_dat);
        intra_dat.total_lk += IntraResBlockEvaluation<
            LJLKScoringData,
            HeavyAtomPairSelector,
            D,
            TILE_SIZE,
            nt,
            Real,
            Int>::
            eval_intrares_atom_pairs(
                tid,
                start_atom1,
                start_atom2,
                score_intra_lk_atom_pair,
                intra_dat);
      });

      // The work: On GPU threads work independently, on CPU, this will be a
      // for loop
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(
          eval_scores_for_atom_pairs);
    });

    eval_block_pair<
        DeviceDispatch,
        D,
        LJLKScoringData,
        LJLKScoringData,
        Real,
        TILE_SIZE>(
        shared,
        pose_ind,
        block_ind1,
        block_ind2,
        block_type1,
        block_type2,
        n_atoms1,
        n_atoms2,
        load_constant_interres_data,
        load_interres1_tile_data_to_shared,
        load_interres2_tile_data_to_shared,
        load_interres_data_from_shared,
        eval_interres_atom_pair_scores,
        store_calculated_energies,
        load_constant_intrares_data,
        load_intrares1_tile_data_to_shared,
        load_intrares2_tile_data_to_shared,
        load_intrares_data_from_shared,
        eval_intrares_atom_pair_scores,
        store_calculated_energies);

    // if (block_ind1 != block_ind2) {
    //   // inter-residue energy evaluation
    //
    //   int const n_conn1 = block_type_n_interblock_bonds[block_type1];
    //   int const n_conn2 = block_type_n_interblock_bonds[block_type2];
    //   int const min_sep =
    //       pose_stack_min_bond_separation[pose_ind][block_ind1][block_ind2];
    //   bool const count_pair_striking_dist =
    //       min_sep <= max_important_bond_separation;
    //
    //   if (count_pair_striking_dist) {
    //     // Load data into shared arrays
    //     auto load_count_pair_conn_at_data = ([&](int tid) {
    //       if (tid < n_conn1) {
    //         shared.m.conn_ats1[tid] =
    //             block_type_atoms_forming_chemical_bonds[block_type1][tid];
    //       }
    //       if (tid < n_conn2) {
    //         shared.m.conn_ats2[tid] =
    //             block_type_atoms_forming_chemical_bonds[block_type2][tid];
    //       }
    //
    //       // NOTE MAX_N_CONN ^ 2 <= 32; limit MAX_N_CONN = 4 before this code
    //       // would need to be adjusted
    //       if (tid < n_conn1 * n_conn2) {
    //         int conn1 = tid / n_conn2;
    //         int conn2 = tid % n_conn2;
    //         shared.m.conn_seps[tid] =
    //             pose_stack_inter_block_bondsep[pose_ind][block_ind1][block_ind2]
    //                                           [conn1][conn2];
    //       }
    //     });
    //     // On CPU: a for loop executed once; on GPU threads within the
    //     workgroup
    //     // working in parallel will just continue to work in parallel
    //     DeviceDispatch<D>::template for_each_in_workgroup<nt>(
    //         load_count_pair_conn_at_data);
    //   }
    //
    //   // Tile the sets of TILE_SIZE atoms
    //   int const n_iterations1 = (n_atoms1 - 1) / TILE_SIZE + 1;
    //   int const n_iterations2 = (n_atoms2 - 1) / TILE_SIZE + 1;
    //   for (int i = 0; i < n_iterations1; ++i) {
    //     // make sure all threads have completed their work
    //     // from the previous iteration before we overwrite
    //     // the contents of shared memory, and, on our first
    //     // iteration, make sure that the conn_ats arrays
    //     // have been written to
    //
    //     // __syncthreads();
    //     DeviceDispatch<D>::synchronize_workgroup();
    //
    //     int const i_n_atoms_to_load1 =
    //         max(0, min(Int(TILE_SIZE), Int((n_atoms1 - TILE_SIZE * i))));
    //
    //     // Let's load coordinates and Lennard-Jones parameters for
    //     // TILE_SIZE atoms into shared memory
    //
    //     auto store_n_heavy1 = ([&](int tid) {
    //       if (tid == 0) {
    //         shared.m.n_heavy1 =
    //             block_type_n_heavy_atoms_in_tile[block_type1][i];
    //       }
    //     });
    //     DeviceDispatch<D>::template
    //     for_each_in_workgroup<nt>(store_n_heavy1);
    //
    //     load_block_into_shared(
    //         pose_ind,
    //         block_coord_offset1,
    //         n_atoms1,
    //         i_n_atoms_to_load1,
    //         block_type1,
    //         n_conn1,
    //         i,
    //         count_pair_striking_dist,
    //         shared.m.conn_ats1,
    //         shared.m.coords1,
    //         shared.m.params1,
    //         shared.m.heavy_inds1,
    //         shared.m.path_dist1);
    //
    //     for (int j = 0; j < n_iterations2; ++j) {
    //       if (j != 0) {
    //         // make sure that all threads have finished energy
    //         // calculations from the previous iteration before we
    //         // overwrite shared memory
    //         DeviceDispatch<D>::synchronize_workgroup();
    //       }
    //       auto store_n_heavy2 = ([&](int tid) {
    //         if (tid == 0) {
    //           shared.m.n_heavy2 =
    //               block_type_n_heavy_atoms_in_tile[block_type2][j];
    //           // printf("n heavy other: %d %d %d\n", alt_block_ind,
    //           // neighb_block_ind, shared.m.union_vals.vals.n_heavy_other);
    //         }
    //       });
    //       DeviceDispatch<D>::template
    //       for_each_in_workgroup<nt>(store_n_heavy2);
    //
    //       int j_n_atoms_to_load2 =
    //           min(Int(TILE_SIZE), Int((n_atoms2 - TILE_SIZE * j)));
    //       load_block_into_shared(
    //           pose_ind,
    //           block_coord_offset2,
    //           n_atoms2,
    //           j_n_atoms_to_load2,
    //           block_type2,
    //           n_conn2,
    //           j,
    //           count_pair_striking_dist,
    //           shared.m.conn_ats2,
    //           shared.m.coords2,
    //           shared.m.params2,
    //           shared.m.heavy_inds2,
    //           shared.m.path_dist2);
    //
    //       // make sure all shared memory writes have completed before we read
    //       // from it when calculating atom-pair energies.
    //       // __syncthreads();
    //       DeviceDispatch<D>::synchronize_workgroup();
    //       int n_heavy1 = shared.m.n_heavy1;
    //       int n_heavy2 = shared.m.n_heavy2;
    //
    //       LJLKScoringData<Real> inter_dat{pose_ind,
    //                                       block_type1,
    // 					  block_type2,
    //                                       block_ind1,
    //                                       block_ind2,
    //                                       block_coord_offset1,
    //                                       block_coord_offset2,
    //                                       shared.m.coords1,
    //                                       shared.m.coords2,
    //                                       shared.m.params1,
    //                                       shared.m.params2,
    //                                       shared.m.heavy_inds1,
    //                                       shared.m.heavy_inds2,
    //                                       max_important_bond_separation,
    //                                       min_sep,
    //                                       n_atoms1,
    //                                       n_atoms2,
    //                                       n_heavy1,
    //                                       n_heavy2,
    //                                       n_conn1,
    //                                       n_conn2,
    //                                       shared.m.path_dist1,
    //                                       shared.m.path_dist2,
    //                                       shared.m.conn_seps,
    //                                       global_params[0]};
    //       auto eval_scores_for_atom_pairs = ([&](int tid) {
    //         total_lj += InterResBlockEvaluation<
    //             LJLKScoringData,
    //             AllAtomPairSelector,
    //             D,
    //             TILE_SIZE,
    //             nt,
    //             Real,
    //             Int>::
    //             eval_interres_atom_pair(
    //                 tid,
    //                 i * TILE_SIZE,
    //                 j * TILE_SIZE,
    //                 score_inter_lj_atom_pair,
    //                 inter_dat);
    //
    //         total_lk += InterResBlockEvaluation<
    //             LJLKScoringData,
    //             HeavyAtomPairSelector,
    //             D,
    //             TILE_SIZE,
    //             nt,
    //             Real,
    //             Int>::
    //             eval_interres_atom_pair(
    //                 tid,
    //                 i * TILE_SIZE,
    //                 j * TILE_SIZE,
    //                 score_inter_lk_atom_pair,
    //                 inter_dat);
    //       });
    //
    //       // The work: On GPU threads work independently, on CPU, this will
    //       be a
    //       // for loop
    //       DeviceDispatch<D>::template for_each_in_workgroup<nt>(
    //           eval_scores_for_atom_pairs);
    //
    //     }  // for j
    //   }    // for i
    // } else {
    //   // alt_block_ind == neighb_block_ind; intra-residue energy evaluation
    //
    //   int const n_iterations = (n_atoms1 - 1) / TILE_SIZE + 1;
    //
    //   for (int i = 0; i < n_iterations; ++i) {
    //     if (i != 0) {
    //       // make sure the calculations for the previous iteration
    //       // have completed before we overwrite the contents of
    //       // shared memory
    //       // __syncthreads();
    //       DeviceDispatch<D>::synchronize_workgroup();
    //     }
    //     int const i_n_atoms_to_load1 =
    //         min(Int(TILE_SIZE), Int((n_atoms1 - TILE_SIZE * i)));
    //
    //     auto set_n_heavy1 = ([&](int tid) {
    //       if (tid == 0) {
    //         shared.m.n_heavy1 =
    //             block_type_n_heavy_atoms_in_tile[block_type1][i];
    //       }
    //     });
    //     DeviceDispatch<D>::template for_each_in_workgroup<nt>(set_n_heavy1);
    //
    //     load_block_coords_and_params_into_shared(
    //         pose_ind,
    //         block_coord_offset1,
    //         i_n_atoms_to_load1,
    //         block_type1,
    //         i,
    //         shared.m.coords1,
    //         shared.m.params1,
    //         shared.m.heavy_inds1);
    //
    //     for (int j = i; j < n_iterations; ++j) {
    //       int const j_n_atoms_to_load2 =
    //           min(Int(TILE_SIZE), Int((n_atoms1 - TILE_SIZE * j)));
    //
    //       if (j != i) {
    //         // make sure calculations from the previous iteration have
    //         // completed before we overwrite the contents of shared
    //         // memory
    //         // __syncthreads();
    //         DeviceDispatch<D>::synchronize_workgroup();
    //       }
    //       if (j != i) {
    //         auto set_n_heavy2 = ([&](int tid) {
    //           if (tid == 0) {
    //             shared.m.n_heavy2 =
    //                 block_type_n_heavy_atoms_in_tile[block_type1][j];
    //           }
    //         });
    //         // Load integer into shared memory
    //         DeviceDispatch<D>::template
    //         for_each_in_workgroup<nt>(set_n_heavy2);
    //
    //         load_block_coords_and_params_into_shared(
    //             pose_ind,
    //             block_coord_offset2,
    //             j_n_atoms_to_load2,
    //             block_type1,
    //             j,
    //             shared.m.coords2,
    //             shared.m.params2,
    //             shared.m.heavy_inds2);
    //       }
    //
    //       // we are guaranteed to hit this syncthreads call; we must wait
    //       // here before reading from shared memory for the coordinates
    //       // in shared.coords_alt1 to be loaded, or if j != i, for the
    //       // coordinates in shared..coords2 to be loaded.
    //       // __syncthreads();
    //       DeviceDispatch<D>::synchronize_workgroup();
    //       int const n_heavy1 = shared.m.n_heavy1;
    //       int const n_heavy2 = (i == j ? n_heavy1 : shared.m.n_heavy2);
    //
    //       LJLKScoringData<Real> intra_dat{
    //           pose_ind,
    //           block_type1,
    //           block_ind1,
    //           block_ind1,
    //           block_coord_offset1,
    //           block_coord_offset1,
    //           shared.m.coords1,
    //           (i == j ? shared.m.coords1 : shared.m.coords2),
    //           shared.m.params1,
    //           (i == j ? shared.m.params1 : shared.m.params2),
    //           shared.m.heavy_inds1,
    //           (i == j ? shared.m.heavy_inds1 : shared.m.heavy_inds2),
    //           max_important_bond_separation,
    //           0,         // min_sep, unused
    //           n_atoms1,  // total number of atoms
    //           n_atoms1,  // total number of atoms
    //           n_heavy1,  // number of heavy atoms within this tile
    //           n_heavy2,  // number of heavy atoms within this tile
    //           0,         // n_conn1, unused in this func
    //           0,         // n_conn2, unused in this func
    //           nullptr,   // shared.m.path_dist1, // unused in this func
    //           nullptr,   // shared.m.path_dist2, // unused in this func
    //           nullptr,   // shared.m.conn_seps, // unused in this func
    //           global_params[0]};
    //
    //       auto eval_scores_for_atom_pairs = ([&](int tid) {
    //         total_lj += IntraResBlockEvaluation<
    //             LJLKScoringData,
    //             AllAtomPairSelector,
    //             D,
    //             TILE_SIZE,
    //             nt,
    //             Real,
    //             Int>::
    //             eval_intrares_atom_pairs(
    //                 tid,
    //                 i * TILE_SIZE,
    //                 j * TILE_SIZE,
    //                 score_intra_lj_atom_pair,
    //                 intra_dat);
    //         total_lk += IntraResBlockEvaluation<
    //             LJLKScoringData,
    //             HeavyAtomPairSelector,
    //             D,
    //             TILE_SIZE,
    //             nt,
    //             Real,
    //             Int>::
    //             eval_intrares_atom_pairs(
    //                 tid,
    //                 i * TILE_SIZE,
    //                 j * TILE_SIZE,
    //                 score_intra_lk_atom_pair,
    //                 intra_dat);
    //       });
    //       // The work: On GPU threads work independently, on CPU, this will
    //       be a
    //       // for loop
    //       DeviceDispatch<D>::template for_each_in_workgroup<nt>(
    //           eval_scores_for_atom_pairs);
    //
    //     }  // for j
    //   }    // for i
    // }      // else
    //
    // // Make sure all energy calculations are complete before we overwrite
    // // the neighbor-residue data in the shared memory union
    // // __syncthreads();
    // DeviceDispatch<D>::synchronize_workgroup();
    //
    // // Real cta_total_lj(0), cta_total_lk(0);
    //
    // auto reduce_energies = ([&](int tid) {
    //   Real const cta_total_lj =
    //       DeviceDispatch<D>::template reduce_in_workgroup<TILE_SIZE>(
    //           total_lj, shared, mgpu::plus_t<Real>());
    //   Real const cta_total_lk =
    //       DeviceDispatch<D>::template reduce_in_workgroup<TILE_SIZE>(
    //           total_lk, shared, mgpu::plus_t<Real>());
    //
    //   if (tid == 0) {
    //     accumulate<D, Real>::add(output[0][pose_ind], cta_total_lj);
    //     accumulate<D, Real>::add(output[1][pose_ind], cta_total_lk);
    //   }
    // });
    // DeviceDispatch<D>::template for_each_in_workgroup<nt>(reduce_energies);
  });

  ///////////////////////////////////////////////////////////////////////

  // Three steps
  // 0: setup
  // 1: launch a kernel to find a small bounding sphere surrounding the blocks
  // 2: launch a kernel to look for spheres that are within striking distance of
  // each other
  // 3: launch a kernel to evaluate lj/lk between pairs of blocks
  // within striking distance

  // 0
  // TO DO: let DeviceDispatch hold a cuda stream (??)
  // at::cuda::CUDAStream wrapped_stream = at::cuda::getDefaultCUDAStream();
  // mgpu::standard_context_t context(wrapped_stream.stream());
  int const n_block_pairs = n_poses * max_n_blocks * max_n_blocks;

  score::common::sphere_overlap::
      compute_block_spheres<DeviceDispatch, D, Real, Int>::f(
          coords,
          pose_stack_block_coord_offset,
          pose_stack_block_type,
          block_type_n_atoms,
          scratch_block_spheres);

  score::common::sphere_overlap::
      detect_block_neighbors<DeviceDispatch, D, Real, Int>::f(
          coords,
          pose_stack_block_coord_offset,
          pose_stack_block_type,
          block_type_n_atoms,
          scratch_block_spheres,
          scratch_block_neighbors,
          Real(6.0));  // 6A hard coded here. Please fix! TEMP!

  // 3
  DeviceDispatch<D>::template foreach_workgroup<launch_t>(
      n_block_pairs, eval_energies);

  return {output_t, dV_dcoords_t};
}

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
