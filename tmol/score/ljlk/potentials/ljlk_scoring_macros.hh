//// kernel macros
//    these define functions that are used in multiple lambda captures
//    variables that are expected to be captured for each macro are specified

#define ATOM_PAIR_LJ_SCORE_W_GRADIENT_EVAL                           \
  TMOL_DEVICE_FUNC(                                                  \
      int atom_tile_ind1,                                            \
      int atom_tile_ind2,                                            \
      int start_atom1,                                               \
      int start_atom2,                                               \
      LJLKScoringData<Real> const& score_dat,                        \
      int cp_separation) {                                           \
    if (require_gradient) {                                          \
      return lj_atom_energy_and_derivs_full(                         \
          atom_tile_ind1,                                            \
          atom_tile_ind2,                                            \
          start_atom1,                                               \
          start_atom2,                                               \
          score_dat,                                                 \
          cp_separation,                                             \
          dV_dcoords);                                               \
    } else {                                                         \
      return lj_atom_energy(                                         \
          atom_tile_ind1, atom_tile_ind2, score_dat, cp_separation); \
    }                                                                \
  }

#define ATOM_PAIR_LK_SCORE_W_GRADIENT_EVAL                           \
  TMOL_DEVICE_FUNC(                                                  \
      int atom_tile_ind1,                                            \
      int atom_tile_ind2,                                            \
      int start_atom1,                                               \
      int start_atom2,                                               \
      LJLKScoringData<Real> const& score_dat,                        \
      int cp_separation) {                                           \
    if (require_gradient) {                                          \
      return lk_atom_energy_and_derivs_full(                         \
          atom_tile_ind1,                                            \
          atom_tile_ind2,                                            \
          start_atom1,                                               \
          start_atom2,                                               \
          score_dat,                                                 \
          cp_separation,                                             \
          dV_dcoords);                                               \
    } else {                                                         \
      return lk_atom_energy(                                         \
          atom_tile_ind1, atom_tile_ind2, score_dat, cp_separation); \
    }                                                                \
  }

#define ATOM_PAIR_LJ_BLOCK_SCORING_DERIVS_EVAL            \
  TMOL_DEVICE_FUNC(                                       \
      int atom_tile_ind1,                                 \
      int atom_tile_ind2,                                 \
      int start_atom1,                                    \
      int start_atom2,                                    \
      LJLKScoringData<Real> const& score_dat,             \
      int cp_separation)                                  \
      ->std::array<Real, 2> {                             \
    lj_atom_derivs(                                       \
        atom_tile_ind1,                                   \
        atom_tile_ind2,                                   \
        start_atom1,                                      \
        start_atom2,                                      \
        score_dat,                                        \
        cp_separation,                                    \
        dTdV[0][score_dat.pose_ind][score_dat.block_ind1] \
            [score_dat.block_ind2],                       \
        dTdV[1][score_dat.pose_ind][score_dat.block_ind1] \
            [score_dat.block_ind2],                       \
        dV_dcoords);                                      \
    return {0.0, 0.0};                                    \
  }

#define ATOM_PAIR_LK_BLOCK_SCORING_DERIVS_EVAL            \
  TMOL_DEVICE_FUNC(                                       \
      int atom_tile_ind1,                                 \
      int atom_tile_ind2,                                 \
      int start_atom1,                                    \
      int start_atom2,                                    \
      LJLKScoringData<Real> const& score_dat,             \
      int cp_separation)                                  \
      ->Real {                                            \
    lk_atom_derivs(                                       \
        atom_tile_ind1,                                   \
        atom_tile_ind2,                                   \
        start_atom1,                                      \
        start_atom2,                                      \
        score_dat,                                        \
        cp_separation,                                    \
        dTdV[2][score_dat.pose_ind][score_dat.block_ind1] \
            [score_dat.block_ind2],                       \
        dV_dcoords);                                      \
    return 0.0;                                           \
  }

#define ATOM_PAIR_LJ_SCORE_WO_GRADIENT_EVAL                        \
  TMOL_DEVICE_FUNC(                                                \
      int atom_tile_ind1,                                          \
      int atom_tile_ind2,                                          \
      int,                                                         \
      int,                                                         \
      LJLKScoringData<Real> const& score_dat,                      \
      int cp_separation) {                                         \
    return lj_atom_energy(                                         \
        atom_tile_ind1, atom_tile_ind2, score_dat, cp_separation); \
  }

#define ATOM_PAIR_LK_SCORE_WO_GRADIENT_EVAL                        \
  TMOL_DEVICE_FUNC(                                                \
      int atom_tile_ind1,                                          \
      int atom_tile_ind2,                                          \
      int,                                                         \
      int,                                                         \
      LJLKScoringData<Real> const& score_dat,                      \
      int cp_separation) {                                         \
    return lk_atom_energy(                                         \
        atom_tile_ind1, atom_tile_ind2, score_dat, cp_separation); \
  }

#define ATOM_PAIR_LJ_SPARSE_IND_SCORING_DERIVS_EVAL \
  TMOL_DEVICE_FUNC(                                 \
      int atom_tile_ind1,                           \
      int atom_tile_ind2,                           \
      int start_atom1,                              \
      int start_atom2,                              \
      LJLKScoringData<Real> const& score_dat,       \
      int cp_separation)                            \
      ->std::array<Real, 2> {                       \
    lj_atom_derivs(                                 \
        atom_tile_ind1,                             \
        atom_tile_ind2,                             \
        start_atom1,                                \
        start_atom2,                                \
        score_dat,                                  \
        cp_separation,                              \
        dTdV[0][cta],                               \
        dTdV[1][cta],                               \
        dV_dcoords);                                \
    return {0.0, 0.0};                              \
  }

#define ATOM_PAIR_LK_SPARSE_IND_SCORING_DERIVS_EVAL \
  TMOL_DEVICE_FUNC(                                 \
      int atom_tile_ind1,                           \
      int atom_tile_ind2,                           \
      int start_atom1,                              \
      int start_atom2,                              \
      LJLKScoringData<Real> const& score_dat,       \
      int cp_separation)                            \
      ->Real {                                      \
    lk_atom_derivs(                                 \
        atom_tile_ind1,                             \
        atom_tile_ind2,                             \
        start_atom1,                                \
        start_atom2,                                \
        score_dat,                                  \
        cp_separation,                              \
        dTdV[2][cta],                               \
        dV_dcoords);                                \
    return 0.0;                                     \
  }

// SCORE_INTER_LJ_ATOM_PAIR
// input argument:  a function with signature (
//     int atom_tile_idx1
//     int atom_tile_idx2
//     int start_atom1
//     int start_atom2
//     LJLKScoringData<Real> const &score_dat
//     int cp_separation)
//   ->std::array<Real, 2>
#define SCORE_INTER_LJ_ATOM_PAIR(atom_pair_func)                \
  TMOL_DEVICE_FUNC(                                             \
      int start_atom1,                                          \
      int start_atom2,                                          \
      int atom_tile_ind1,                                       \
      int atom_tile_ind2,                                       \
      LJLKScoringData<Real> const& inter_dat) {                 \
    int separation = interres_count_pair_separation<TILE_SIZE>( \
        inter_dat, atom_tile_ind1, atom_tile_ind2);             \
    return atom_pair_func(                                      \
        atom_tile_ind1,                                         \
        atom_tile_ind2,                                         \
        start_atom1,                                            \
        start_atom2,                                            \
        inter_dat,                                              \
        separation);                                            \
  }

// SCORE_INTRA_LJ_ATOM_PAIR
// input argument:  a function with signature (
//     int atom_tile_idx1
//     int atom_tile_idx2
//     int start_atom1
//     int start_atom2
//     LJLKScoringData<Real> const &score_dat
//     int cp_separation)
//   ->std::array<Real, 2>
// captures:
//    block_type_path_distance
#define SCORE_INTRA_LJ_ATOM_PAIR(atom_pair_func)                             \
  TMOL_DEVICE_FUNC(                                                          \
      int start_atom1,                                                       \
      int start_atom2,                                                       \
      int atom_tile_ind1,                                                    \
      int atom_tile_ind2,                                                    \
      LJLKScoringData<Real> const& intra_dat)                                \
      ->std::array<Real, 2> {                                                \
    int const atom_ind1 = start_atom1 + atom_tile_ind1;                      \
    int const atom_ind2 = start_atom2 + atom_tile_ind2;                      \
    int const separation = block_type_path_distance[intra_dat.r1.block_type] \
                                                   [atom_ind1][atom_ind2];   \
    return atom_pair_func(                                                   \
        atom_tile_ind1,                                                      \
        atom_tile_ind2,                                                      \
        start_atom1,                                                         \
        start_atom2,                                                         \
        intra_dat,                                                           \
        separation);                                                         \
  }

// SCORE_INTER_LK_ATOM_PAIR
// input argument:  a function with signature (
//     int atom_tile_idx1
//     int atom_tile_idx2
//     int start_atom1
//     int start_atom2
//     LJLKScoringData<Real> const &score_dat
//     int cp_separation)
//   ->Real
// captures:
//    None
#define SCORE_INTER_LK_ATOM_PAIR(atom_pair_func)                              \
  TMOL_DEVICE_FUNC(                                                           \
      int start_atom1,                                                        \
      int start_atom2,                                                        \
      int atom_heavy_tile_ind1,                                               \
      int atom_heavy_tile_ind2,                                               \
      LJLKScoringData<Real> const& inter_dat)                                 \
      ->std::array<Real, 1> {                                                 \
    int const atom_tile_ind1 = inter_dat.r1.heavy_inds[atom_heavy_tile_ind1]; \
    int const atom_tile_ind2 = inter_dat.r2.heavy_inds[atom_heavy_tile_ind2]; \
    int separation = interres_count_pair_separation<TILE_SIZE>(               \
        inter_dat, atom_tile_ind1, atom_tile_ind2);                           \
    Real lk = atom_pair_func(                                                 \
        atom_tile_ind1,                                                       \
        atom_tile_ind2,                                                       \
        start_atom1,                                                          \
        start_atom2,                                                          \
        inter_dat,                                                            \
        separation);                                                          \
    return {lk};                                                              \
  }

// SCORE_INTRA_LK_ATOM_PAIR
// input argument:  a function with signature (
//     int atom_tile_idx1
//     int atom_tile_idx2
//     int start_atom1
//     int start_atom2
//     LJLKScoringData<Real> const &score_dat
//     int cp_separation)
//   ->Real
// captures:
//    block_type_path_distance
#define SCORE_INTRA_LK_ATOM_PAIR(atom_pair_func)                              \
  TMOL_DEVICE_FUNC(                                                           \
      int start_atom1,                                                        \
      int start_atom2,                                                        \
      int atom_heavy_tile_ind1,                                               \
      int atom_heavy_tile_ind2,                                               \
      LJLKScoringData<Real> const& intra_dat)                                 \
      ->std::array<Real, 1> {                                                 \
    int const atom_tile_ind1 = intra_dat.r1.heavy_inds[atom_heavy_tile_ind1]; \
    int const atom_tile_ind2 = intra_dat.r2.heavy_inds[atom_heavy_tile_ind2]; \
    int const atom_ind1 = start_atom1 + atom_tile_ind1;                       \
    int const atom_ind2 = start_atom2 + atom_tile_ind2;                       \
    int const separation = block_type_path_distance[intra_dat.r1.block_type]  \
                                                   [atom_ind1][atom_ind2];    \
    Real lk = atom_pair_func(                                                 \
        atom_tile_ind1,                                                       \
        atom_tile_ind2,                                                       \
        start_atom1,                                                          \
        start_atom2,                                                          \
        intra_dat,                                                            \
        separation);                                                          \
    return {lk};                                                              \
  }

// SCORE_INTRA_LK_ATOM_PAIR
// captures:
//    coords (TView<Vec<Real, 3>, 2, D>)
//    block_type_atom_types (TView<Int, 2, D>)
//    type_params (TView<LJLKTypeParams<Real>, 1, D>)
//    block_type_heavy_atoms_in_tile (TView<Int, 2, D>)
#define LOAD_BLOCK_COORDS_AND_PARAMS_INTO_SHARED                            \
  TMOL_DEVICE_FUNC(                                                         \
      int pose_ind,                                                         \
      LJLKSingleResData<Real>& r_dat,                                       \
      int n_atoms_to_load,                                                  \
      int start_atom) {                                                     \
    ljlk_load_block_coords_and_params_into_shared<DeviceOperations, D, nt>( \
        rot_coords,                                                         \
        block_type_atom_types,                                              \
        type_params,                                                        \
        block_type_heavy_atoms_in_tile,                                     \
        pose_ind,                                                           \
        r_dat,                                                              \
        n_atoms_to_load,                                                    \
        start_atom);                                                        \
  }

// LOAD_BLOCK_INTO_SHARED
// captures:
//    coords (TView<Vec<Real, 3>, 2, D>)
//    block_type_atom_types (TView<Int, 2, D>)
//    type_params (TView<LJLKTypeParams<Real>, 1, D>)
//    block_type_heavy_atoms_in_tile (TView<Int, 2, D>)
//    block_type_path_distance (TView<Int, 3, D>)
#define LOAD_BLOCK_INTO_SHARED                                       \
  TMOL_DEVICE_FUNC(                                                  \
      int pose_ind,                                                  \
      LJLKSingleResData<Real>& r_dat,                                \
      int n_atoms_to_load,                                           \
      int start_atom,                                                \
      bool count_pair_striking_dist,                                 \
      unsigned char* __restrict__ conn_ats) {                        \
    ljlk_load_block_into_shared<DeviceOperations, D, nt, TILE_SIZE>( \
        rot_coords,                                                  \
        block_type_atom_types,                                       \
        type_params,                                                 \
        block_type_heavy_atoms_in_tile,                              \
        block_type_path_distance,                                    \
        pose_ind,                                                    \
        r_dat,                                                       \
        n_atoms_to_load,                                             \
        start_atom,                                                  \
        count_pair_striking_dist,                                    \
        conn_ats);                                                   \
  }

// LOAD_TILE_INVARIANT_INTERRES_DATA
// captures:
//    pose_stack_block_coord_offset (TView<Vec<Real, 3>, 2, D>)
//    pose_stack_min_bond_separation (TView<Int, 3, D>)
//    block_type_n_interblock_bonds (TView<Int, 1, D>)
//    block_type_atoms_forming_chemical_bonds (TView<Int, 2, D>)
//    pose_stack_inter_block_bondsep (TView<Int, 5, D>)
//    global_params (TView<LJGlobalParams<Real>, 1, D>)
//    max_important_bond_separation (int)
#define LOAD_TILE_INVARIANT_INTERRES_DATA                            \
  TMOL_DEVICE_FUNC(                                                  \
      int pose_ind,                                                  \
      int rot_ind1,                                                  \
      int rot_ind2,                                                  \
      int block_ind1,                                                \
      int block_ind2,                                                \
      int block_type1,                                               \
      int block_type2,                                               \
      int n_atoms1,                                                  \
      int n_atoms2,                                                  \
      LJLKScoringData<Real>& inter_dat,                              \
      shared_mem_union& shared) {                                    \
    ljlk_load_tile_invariant_interres_data<DeviceOperations, D, nt>( \
        rot_coord_offset,                                            \
        pose_stack_min_bond_separation,                              \
        block_type_n_interblock_bonds,                               \
        block_type_atoms_forming_chemical_bonds,                     \
        pose_stack_inter_block_bondsep,                              \
        global_params,                                               \
        max_important_bond_separation,                               \
        pose_ind,                                                    \
        rot_ind1,                                                    \
        rot_ind2,                                                    \
        block_ind1,                                                  \
        block_ind2,                                                  \
        block_type1,                                                 \
        block_type2,                                                 \
        n_atoms1,                                                    \
        n_atoms2,                                                    \
        inter_dat,                                                   \
        shared.m);                                                   \
  }

// LOAD_INTERRES1_TILE_DATA_TO_SHARED
// captures:
//    coords (TView<Vec<Real, 3>, 2, D>)
//    block_type_atom_types (TView<Int, 2, D>)
//    type_params (TView<LJLKTypeParams<Real>, 1, D>)
//    block_type_heavy_atoms_in_tile (TView<Int, 2, D>)
//    block_type_path_distance (TView<Int, 3, D>)
//    block_type_n_heavy_atoms_in_tile (TView<Int, 2, D>)
#define LOAD_INTERRES1_TILE_DATA_TO_SHARED                            \
  TMOL_DEVICE_FUNC(                                                   \
      int tile_ind,                                                   \
      int start_atom1,                                                \
      int n_atoms_to_load1,                                           \
      LJLKScoringData<Real>& inter_dat,                               \
      shared_mem_union& shared) {                                     \
    ljlk_load_interres1_tile_data_to_shared<DeviceOperations, D, nt>( \
        rot_coords,                                                   \
        block_type_atom_types,                                        \
        type_params,                                                  \
        block_type_heavy_atoms_in_tile,                               \
        block_type_path_distance,                                     \
        block_type_n_heavy_atoms_in_tile,                             \
        tile_ind,                                                     \
        start_atom1,                                                  \
        n_atoms_to_load1,                                             \
        inter_dat,                                                    \
        shared.m);                                                    \
  }

// LOAD_INTERRES2_TILE_DATA_TO_SHARED
//   same as LOAD_INTERRES1_TILE_DATA_TO_SHARED but saves to inter_dat.r2
// captures:
//    coords (TView<Vec<Real, 3>, 2, D>)
//    block_type_atom_types (TView<Int, 2, D>)
//    type_params (TView<LJLKTypeParams<Real>, 1, D>)
//    block_type_heavy_atoms_in_tile (TView<Int, 2, D>)
//    block_type_path_distance (TView<Int, 3, D>)
//    block_type_n_heavy_atoms_in_tile (TView<Int, 2, D>)
#define LOAD_INTERRES2_TILE_DATA_TO_SHARED                            \
  TMOL_DEVICE_FUNC(                                                   \
      int tile_ind,                                                   \
      int start_atom2,                                                \
      int n_atoms_to_load2,                                           \
      LJLKScoringData<Real>& inter_dat,                               \
      shared_mem_union& shared) {                                     \
    ljlk_load_interres2_tile_data_to_shared<DeviceOperations, D, nt>( \
        rot_coords,                                                   \
        block_type_atom_types,                                        \
        type_params,                                                  \
        block_type_heavy_atoms_in_tile,                               \
        block_type_path_distance,                                     \
        block_type_n_heavy_atoms_in_tile,                             \
        tile_ind,                                                     \
        start_atom2,                                                  \
        n_atoms_to_load2,                                             \
        inter_dat,                                                    \
        shared.m);                                                    \
  }

// LOAD_INTERRES_DATA_FROM_SHARED
// captures:
//    nothing
#define LOAD_INTERRES_DATA_FROM_SHARED                                        \
  TMOL_DEVICE_FUNC(                                                           \
      int, int, shared_mem_union& shared, LJLKScoringData<Real>& inter_dat) { \
    ljlk_load_interres_data_from_shared(shared.m, inter_dat);                 \
  }

// EVAL_INTERRES_ATOM_PAIR_SCORES
// captures:
//    score_inter_lj_atom_pair (lambda)
//    score_inter_lk_atom_pair (lambda)
#define EVAL_INTERRES_ATOM_PAIR_SCORES                                      \
  TMOL_DEVICE_FUNC(                                                         \
      LJLKScoringData<Real>& inter_dat, int start_atom1, int start_atom2) { \
    auto eval_scores_for_atom_pairs = ([&](int tid) {                       \
      auto LJ = tmol::score::common::InterResBlockEvaluation<               \
          LJLKScoringData,                                                  \
          AllAtomPairSelector,                                              \
          D,                                                                \
          TILE_SIZE,                                                        \
          nt,                                                               \
          2,                                                                \
          Real,                                                             \
          Int>::                                                            \
          eval_interres_atom_pair(                                          \
              tid,                                                          \
              start_atom1,                                                  \
              start_atom2,                                                  \
              score_inter_lj_atom_pair,                                     \
              inter_dat);                                                   \
                                                                            \
      inter_dat.total_ljatr += std::get<0>(LJ);                             \
      inter_dat.total_ljrep += std::get<1>(LJ);                             \
                                                                            \
      auto LK = tmol::score::common::InterResBlockEvaluation<               \
          LJLKScoringData,                                                  \
          HeavyAtomPairSelector,                                            \
          D,                                                                \
          TILE_SIZE,                                                        \
          nt,                                                               \
          1,                                                                \
          Real,                                                             \
          Int>::                                                            \
          eval_interres_atom_pair(                                          \
              tid,                                                          \
              start_atom1,                                                  \
              start_atom2,                                                  \
              score_inter_lk_atom_pair,                                     \
              inter_dat);                                                   \
      inter_dat.total_lk += std::get<0>(LK);                                \
    });                                                                     \
    DeviceOperations<D>::template for_each_in_workgroup<nt>(                \
        eval_scores_for_atom_pairs);                                        \
  }

// STORE_CALCULATED_ENERGIES
//    store energies if we are NOT computing per-blockpair
// captures:
//    output_block_pair_energies (bool)
//    output (TView<Real, 4, D>)
#define STORE_POSE_CALCULATED_ENERGIES                                        \
  TMOL_DEVICE_FUNC(                                                           \
      LJLKScoringData<Real>& score_dat, shared_mem_union& shared) {           \
    auto reduce_energies = ([&](int tid) {                                    \
      Real const cta_total_ljatr =                                            \
          DeviceOperations<D>::template reduce_in_workgroup<nt>(              \
              score_dat.total_ljatr, shared, mgpu::plus_t<Real>());           \
      Real const cta_total_ljrep =                                            \
          DeviceOperations<D>::template reduce_in_workgroup<nt>(              \
              score_dat.total_ljrep, shared, mgpu::plus_t<Real>());           \
      Real const cta_total_lk =                                               \
          DeviceOperations<D>::template reduce_in_workgroup<nt>(              \
              score_dat.total_lk, shared, mgpu::plus_t<Real>());              \
                                                                              \
      if (tid == 0) {                                                         \
        int const p = score_dat.pose_ind;                                     \
        if (output_block_pair_energies) {                                     \
          int const b1 = score_dat.block_ind1;                                \
          int const b2 = score_dat.block_ind2;                                \
          output[0][p][b1][b2] = cta_total_ljatr;                             \
          output[1][p][b1][b2] = cta_total_ljrep;                             \
          output[2][p][b1][b2] = cta_total_lk;                                \
        } else {                                                              \
          accumulate<D, Real>::add(output[0][p][0][0], cta_total_ljatr);      \
          accumulate<D, Real>::add(output[1][p][0][0], cta_total_ljrep);      \
          accumulate<D, Real>::add(output[2][p][0][0], cta_total_lk);         \
        }                                                                     \
      }                                                                       \
    });                                                                       \
    DeviceOperations<D>::template for_each_in_workgroup<nt>(reduce_energies); \
  }

// STORE_ROTAMER_CALCULATED_ENERGIES
//    store energies if we ARE computing rotamer pair energies
// captures:
//    cta (int)
//    output (TView<Real, 4, D>)
#define STORE_ROTAMER_CALCULATED_ENERGIES                                     \
  TMOL_DEVICE_FUNC(                                                           \
      LJLKScoringData<Real>& score_dat, shared_mem_union& shared) {           \
    auto reduce_energies = ([&](int tid) {                                    \
      Real const cta_total_ljatr =                                            \
          DeviceOperations<D>::template reduce_in_workgroup<nt>(              \
              score_dat.total_ljatr, shared, mgpu::plus_t<Real>());           \
      Real const cta_total_ljrep =                                            \
          DeviceOperations<D>::template reduce_in_workgroup<nt>(              \
              score_dat.total_ljrep, shared, mgpu::plus_t<Real>());           \
      Real const cta_total_lk =                                               \
          DeviceOperations<D>::template reduce_in_workgroup<nt>(              \
              score_dat.total_lk, shared, mgpu::plus_t<Real>());              \
      if (tid == 0) {                                                         \
        output[0][cta] = cta_total_ljatr;                                     \
        output[1][cta] = cta_total_ljrep;                                     \
        output[2][cta] = cta_total_lk;                                        \
      }                                                                       \
    });                                                                       \
    DeviceOperations<D>::template for_each_in_workgroup<nt>(reduce_energies); \
  }

// LOAD_TILE_INVARIANT_INTRARES_DATA
// captures:
//    pose_stack_block_coord_offset (TView<Int, 2, D>)
//    global_params (TView<LJGlobalParams<Real>, 1, D>)
//    max_important_bond_separation (int)
#define LOAD_TILE_INVARIANT_INTRARES_DATA                            \
  TMOL_DEVICE_FUNC(                                                  \
      int pose_ind,                                                  \
      int rot_ind1,                                                  \
      int block_ind1,                                                \
      int block_type1,                                               \
      int n_atoms1,                                                  \
      LJLKScoringData<Real>& intra_dat,                              \
      shared_mem_union& shared) {                                    \
    ljlk_load_tile_invariant_intrares_data<DeviceOperations, D, nt>( \
        rot_coord_offset,                                            \
        global_params,                                               \
        max_important_bond_separation,                               \
        pose_ind,                                                    \
        rot_ind1,                                                    \
        block_ind1,                                                  \
        block_type1,                                                 \
        n_atoms1,                                                    \
        intra_dat,                                                   \
        shared.m);                                                   \
  }

// LOAD_INTRARES1_TILE_DATA_TO_SHARED
// captures:
//    coords (TView<Vec<Real, 3>, 2, D>)
//    block_type_atom_types (TView<Int, 2, D>)
//    type_params (TView<LJLKTypeParams<Real>, 1, D>)
//    block_type_n_heavy_atoms_in_tile (TView<Int, 2, D>)
//    block_type_heavy_atoms_in_tile (TView<Int, 2, D>)
#define LOAD_INTRARES1_TILE_DATA_TO_SHARED                            \
  TMOL_DEVICE_FUNC(                                                   \
      int tile_ind,                                                   \
      int start_atom1,                                                \
      int n_atoms_to_load1,                                           \
      LJLKScoringData<Real>& intra_dat,                               \
      shared_mem_union& shared) {                                     \
    ljlk_load_intrares1_tile_data_to_shared<DeviceOperations, D, nt>( \
        rot_coords,                                                   \
        block_type_atom_types,                                        \
        type_params,                                                  \
        block_type_n_heavy_atoms_in_tile,                             \
        block_type_heavy_atoms_in_tile,                               \
        tile_ind,                                                     \
        start_atom1,                                                  \
        n_atoms_to_load1,                                             \
        intra_dat,                                                    \
        shared.m);                                                    \
  }

// LOAD_INTRARES2_TILE_DATA_TO_SHARED
//    same as LOAD_INTRARES1_TILE_DATA_TO_SHARED but assign to intra_dat.r2
// captures:
//    coords (TView<Vec<Real, 3>, 2, D>)
//    block_type_atom_types (TView<Int, 2, D>)
//    type_params (TView<LJLKTypeParams<Real>, 1, D>)
//    block_type_n_heavy_atoms_in_tile (TView<Int, 2, D>)
//    block_type_heavy_atoms_in_tile (TView<Int, 2, D>)
#define LOAD_INTRARES2_TILE_DATA_TO_SHARED                            \
  TMOL_DEVICE_FUNC(                                                   \
      int tile_ind,                                                   \
      int start_atom2,                                                \
      int n_atoms_to_load2,                                           \
      LJLKScoringData<Real>& intra_dat,                               \
      shared_mem_union& shared) {                                     \
    ljlk_load_intrares2_tile_data_to_shared<DeviceOperations, D, nt>( \
        rot_coords,                                                   \
        block_type_atom_types,                                        \
        type_params,                                                  \
        block_type_n_heavy_atoms_in_tile,                             \
        block_type_heavy_atoms_in_tile,                               \
        tile_ind,                                                     \
        start_atom2,                                                  \
        n_atoms_to_load2,                                             \
        intra_dat,                                                    \
        shared.m);                                                    \
  }

// LOAD_INTRARES_DATA_FROM_SHARED
// captures:
//     nothing
#define LOAD_INTRARES_DATA_FROM_SHARED              \
  TMOL_DEVICE_FUNC(                                 \
      int tile_ind1,                                \
      int tile_ind2,                                \
      shared_mem_union& shared,                     \
      LJLKScoringData<Real>& intra_dat) {           \
    ljlk_load_intrares_data_from_shared(            \
        tile_ind1, tile_ind2, shared.m, intra_dat); \
  }

// EVAL_INTRARES_ATOM_PAIR_SCORES
// captures:
//    score_intra_lj_atom_pair (lambda)
//    score_intra_lk_atom_pair (lambda)
#define EVAL_INTRARES_ATOM_PAIR_SCORES                                      \
  TMOL_DEVICE_FUNC(                                                         \
      LJLKScoringData<Real>& intra_dat, int start_atom1, int start_atom2) { \
    auto eval_scores_for_atom_pairs = ([&](int tid) {                       \
      auto LJ = tmol::score::common::IntraResBlockEvaluation<               \
          LJLKScoringData,                                                  \
          AllAtomPairSelector,                                              \
          D,                                                                \
          TILE_SIZE,                                                        \
          nt,                                                               \
          2,                                                                \
          Real,                                                             \
          Int>::                                                            \
          eval_intrares_atom_pairs(                                         \
              tid,                                                          \
              start_atom1,                                                  \
              start_atom2,                                                  \
              score_intra_lj_atom_pair,                                     \
              intra_dat);                                                   \
                                                                            \
      intra_dat.total_ljatr += std::get<0>(LJ);                             \
      intra_dat.total_ljrep += std::get<1>(LJ);                             \
                                                                            \
      auto LK = tmol::score::common::IntraResBlockEvaluation<               \
          LJLKScoringData,                                                  \
          HeavyAtomPairSelector,                                            \
          D,                                                                \
          TILE_SIZE,                                                        \
          nt,                                                               \
          1,                                                                \
          Real,                                                             \
          Int>::                                                            \
          eval_intrares_atom_pairs(                                         \
              tid,                                                          \
              start_atom1,                                                  \
              start_atom2,                                                  \
              score_intra_lk_atom_pair,                                     \
              intra_dat);                                                   \
                                                                            \
      intra_dat.total_lk += std::get<0>(LK);                                \
    });                                                                     \
    DeviceOperations<D>::template for_each_in_workgroup<nt>(                \
        eval_scores_for_atom_pairs);                                        \
  }
// end of macro definitions