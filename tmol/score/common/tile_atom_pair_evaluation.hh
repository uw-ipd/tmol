#pragma once

#include <Eigen/Core>
#include <tmol/score/common/diamond_macros.hh>

namespace tmol {
namespace score {
namespace common {

// Implements n_atoms1 and n_atoms2 to load data out of a templated class
// "InterPairData" that should have data members "r1" and "r2" both
// of which have a data member named "n_atoms."
template <template <typename T> typename InterPairData, typename T>
class AllAtomPairSelector {
 public:
  static EIGEN_DEVICE_FUNC int n_atoms1(InterPairData<T> const &inter_data) {
    return inter_data.r1.n_atoms;
  }
  static EIGEN_DEVICE_FUNC int n_atoms2(InterPairData<T> const &inter_data) {
    return inter_data.r2.n_atoms;
  }
};

// Implements n_atoms1 and n_atoms2 to load data out of a templated class
// "InterPairData" that should have data members "r1" and "r2" both
// of which have a data member named "n_heavy."
template <template <typename T> typename InterPairData, typename T>
class HeavyAtomPairSelector {
 public:
  static EIGEN_DEVICE_FUNC int n_atoms1(InterPairData<T> const &inter_data) {
    return inter_data.r1.n_heavy;
  }
  static EIGEN_DEVICE_FUNC int n_atoms2(InterPairData<T> const &inter_data) {
    return inter_data.r2.n_heavy;
  }
};

// helper code for summing over std::arrays at compile time
// https://stackoverflow.com/a/47563100
template <std::size_t N>
struct num {
  static const constexpr auto value = N;
};

template <class F, std::size_t... Is>
TMOL_DEVICE_FUNC void for_(F func, std::index_sequence<Is...>) {
  using expander = int[];
  (void)expander{0, ((void)func(num<Is>{}), 0)...};
}

template <std::size_t N, typename F>
TMOL_DEVICE_FUNC void for_(F func) {
  for_(func, std::make_index_sequence<N>());
}

// Templated function for the evaluation of inter-block atom-pair energy
// evaluations where each worker (identified by tid) steps across the available
// atom pairs with a stride of "nt." The templated PairSelector class should
// determine how many atoms there are in the tile; e.g. if iterating across all
// atoms, a different number of atom-pairs will be evaluated than when iterating
// across only the heavy-atom pairs. The AllAtomPairSelector or
// HeavyAtomPairSelectors defined above can be used for this purpose.
template <
    template <typename>
    typename InterEnergyData,
    template <template <typename> typename, typename>
    typename PairSelector,
    tmol::Device D,
    int TILE,
    int nt,
    int NTERMS,
    typename Real,
    typename Int>
class InterResBlockEvaluation {
 public:
  template <typename AtomPairFunc>
  static TMOL_DEVICE_FUNC std::array<Real, NTERMS> eval_interres_atom_pair(
      int tid,
      int start_atom1,
      int start_atom2,
      AtomPairFunc f,
      InterEnergyData<Real> const &inter_dat) {
    std::array<Real, NTERMS> score_total = {};
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
      std::array<Real, NTERMS> score_i = f(
          start_atom1, start_atom2, atom_tile_ind1, atom_tile_ind2, inter_dat);
      for_<NTERMS>([&](auto i) {
        std::get<i.value>(score_total) += std::get<i.value>(score_i);
      });
    }
    return score_total;
  }
};

// Templated function for iterating across a tile of intra-block work where
// each worker (identified by tid) steps across the the available atom
// pairs with a stride of "nt." Atom pairs are evaluated only a single time
// so if the atom1 index is >= the atom2 index, the work is skipped.
// TO DO: replace with upper-triangle indexing to reduce idle threads
template <
    template <typename>
    typename IntraEnergyData,
    template <template <typename> typename, typename>
    typename PairSelector,
    tmol::Device D,
    int TILE,
    int nt,
    int NTERMS,
    typename Real,
    typename Int>
class IntraResBlockEvaluation {
 public:
  template <typename AtomPairFunc>
  static TMOL_DEVICE_FUNC std::array<Real, NTERMS> eval_intrares_atom_pairs(
      int tid,
      int start_atom1,
      int start_atom2,
      AtomPairFunc f,
      IntraEnergyData<Real> const &intra_dat) {
    std::array<Real, NTERMS> score_total = {};
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
      std::array<Real, NTERMS> score_i = f(
          start_atom1, start_atom2, atom_tile_ind1, atom_tile_ind2, intra_dat);
      for_<NTERMS>([&](auto i) {
        std::get<i.value>(score_total) += std::get<i.value>(score_i);
      });
    }
    return score_total;
  }
};

// Templated method for tiling atom-pair energy evaluations which lets
// energy functions define lambda functions representing the five steps
// of tiled energy evaluation. In "tiled" evaluation of energies,
// each block is partitioned into subsets up to a maximum fixed number
// of atoms, (e.g. 32 atoms from block 1 and 32 atoms from block 2)
// so that the work of evaluating all pairs of atom interactions
// is divided into "tiles" (e.g. 32 x 32 atom pair interactions). This
// amortizes the expense of loading O(N) data into memory over
// O(N^2) calculations (for N < 32), maximizing the memory-bandwidth-
// used-per floating-point operation.
//
// There are five steps that the user is expected to define for
// both inter- and intra-block energy evaluations.
//
//   1. initializing a local-memory data structure for holding the data
//   needed to evaluate the interaction between the atoms in a tile
//   2. loading the data needed for the tile atoms of block 1 into
//   shared memory
//   3. loading the data needed for the tile atoms of block 2 into
//   shared memory
//   4. finalizing any relationship between the shared-memory data
//   structure and the local-memory data structure, and lastly
//   5. evaluating the energy between the atoms in this tile
//
// The structure for inter-residue calculations and intra-residue
// calculations is remarkably similar with one notable difference:
// The local-memory data used for inter-block evaluations should likely
// have one set of pointers to shared-memory arrays for block 1 and
// another set of pointers to shared-memory arrays for block 2. The
// local-memory data structure for intra-residue calculations
// should be similar in that when atoms from one tile are evaluated
// against atoms in a different tile, the atom data will need to be
// loaded into separate locations in shared memory. (Indeed there is
// no reason that the user could not use the same data structure
// for InterResScoringData as for IntraResScoringData.) If the user
// plans to load data for the atom1 from the block1 shared-mem
// pointers and atom2 from the block2 shared-mem pointers as a rule,
// then, TAKE HEED, an important point of caution must be noted: the
// lambda function "LoadIntraDataFunc2" is only invoked for inter-tile
// calculations. Any pointer to block2 data should be set to point at
// "block1" data in the "LoadInterDataFunc1" or in the
// "LoadIntraSharedDatFunc."
template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename InterResScoringData,
    typename IntraResScoringData,
    typename Real,
    int TILE,
    typename SharedMemData,
    typename LoadInvarInterFunc,
    typename LoadInvarIntraFunc,
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
    LoadInvarInterFunc load_tile_invariant_interres_data,
    LoadInterDatFunc1 load_interres1_tile_data_to_shared,
    LoadInterDatFunc2 load_interres2_tile_data_to_shared,
    LoadInterSharedDatFunc load_interres_data_from_shared,
    CalcInterFunc eval_interres_atom_pair_scores,
    StoreEnergyFunc store_calculated_interres_energies,
    LoadInvarIntraFunc load_tile_invariant_intrares_data,
    LoadIntraDatFunc1 load_intrares1_tile_data_to_shared,
    LoadIntraDatFunc2 load_intrares2_tile_data_to_shared,
    LoadIntraSharedDatFunc load_intrares_data_from_shared,
    CalcIntraFunc eval_intrares_atom_pair_scores,
    StoreEnergyFunc store_calculated_intrares_energies) {
  // printf("starting %d %d\n", block_ind1, block_ind2);
  if (block_ind1 != block_ind2) {
    // Step 1: load any data that is consistent across all tile pairs
    InterResScoringData interres_data;
    // printf("calling load_tile_invariant_interres_data\n");
    load_tile_invariant_interres_data(
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
      // Make sure the tile-invariant inter-res data has been loaded
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
    IntraResScoringData intrares_data;
    // printf("calling load_tile_invariant_intrares_data\n");
    load_tile_invariant_intrares_data(
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
