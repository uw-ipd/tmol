#pragma once

#include <Eigen/Core>
#include <tmol/score/common/diamond_macros.hh>

namespace tmol {
namespace score {
namespace common {

template <template <typename T> typename InterPairData, typename T>
class AllAtomPairSelector {
 public:
  static EIGEN_DEVICE_FUNC int n_atoms1(InterPairData<T> const &inter_data) {
    return inter_data.r1.invar_dat.n_atoms;
  }
  static EIGEN_DEVICE_FUNC int n_atoms2(InterPairData<T> const &inter_data) {
    return inter_data.r2.invar_dat.n_atoms;
  }
};

template <template <typename T> typename InterPairData, typename T>
class HeavyAtomPairSelector {
 public:
  static EIGEN_DEVICE_FUNC int n_atoms1(InterPairData<T> const &inter_data) {
    return inter_data.r1.tile_dat.n_heavy;
  }
  static EIGEN_DEVICE_FUNC int n_atoms2(InterPairData<T> const &inter_data) {
    return inter_data.r2.tile_dat.n_heavy;
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
TMOL_DEVICE_FUNC void tile_evaluate_block_pair(
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

}  // namespace common
}  // namespace score
}  // namespace tmol
