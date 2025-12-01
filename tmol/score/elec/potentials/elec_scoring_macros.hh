//// kernel macros
//    these define functions that are used in multiple lambda captures
//    variables that are expected to be captured for each macro are specified

#define SCORE_INTER_ELEC_ATOM_PAIR                              \
  TMOL_DEVICE_FUNC(                                             \
      int start_atom1,                                          \
      int start_atom2,                                          \
      int atom_tile_ind1,                                       \
      int atom_tile_ind2,                                       \
      ElecScoringData<Real> const& inter_dat)                   \
      ->std::array<Real, 1> {                                   \
    int separation = interres_count_pair_separation<TILE_SIZE>( \
        inter_dat, atom_tile_ind1, atom_tile_ind2);             \
    Real elec = elec_atom_energy_and_derivs(                    \
        atom_tile_ind1,                                         \
        atom_tile_ind2,                                         \
        start_atom1,                                            \
        start_atom2,                                            \
        inter_dat,                                              \
        separation);                                            \
    return {elec};                                              \
  }

#define SCORE_INTRA_ELEC_ATOM_PAIR                                   \
  TMOL_DEVICE_FUNC(                                                  \
      int start_atom1,                                               \
      int start_atom2,                                               \
      int atom_tile_ind1,                                            \
      int atom_tile_ind2,                                            \
      ElecScoringData<Real> const& intra_dat)                        \
      ->std::array<Real, 1> {                                        \
    int const atom_ind1 = start_atom1 + atom_tile_ind1;              \
    int const atom_ind2 = start_atom2 + atom_tile_ind2;              \
    int const separation =                                           \
        block_type_intra_repr_path_distance[intra_dat.r1.block_type] \
                                           [atom_ind1][atom_ind2];   \
    Real elec = elec_atom_energy_and_derivs(                         \
        atom_tile_ind1,                                              \
        atom_tile_ind2,                                              \
        start_atom1,                                                 \
        start_atom2,                                                 \
        intra_dat,                                                   \
        separation);                                                 \
    return {elec};                                                   \
  }

#define LOAD_BLOCK_COORDS_AND_PARAMS_INTO_SHARED                             \
  TMOL_DEVICE_FUNC(                                                          \
      int pose_ind,                                                          \
      ElecSingleResData<Real>& r_dat,                                        \
      int n_atoms_to_load,                                                   \
      int start_atom) {                                                      \
    elec_load_block_coords_and_charges_into_shared<DeviceOperations, D, nt>( \
        rot_coords,                                                          \
        block_type_partial_charge,                                           \
        pose_ind,                                                            \
        r_dat,                                                               \
        n_atoms_to_load,                                                     \
        start_atom);                                                         \
  }

#define LOAD_BLOCK_INTO_SHARED                                       \
  TMOL_DEVICE_FUNC(                                                  \
      int pose_ind,                                                  \
      ElecSingleResData<Real>& r_dat,                                \
      int n_atoms_to_load,                                           \
      int start_atom,                                                \
      bool count_pair_striking_dist,                                 \
      unsigned char* __restrict__ conn_ats) {                        \
    elec_load_block_into_shared<DeviceOperations, D, nt, TILE_SIZE>( \
        rot_coords,                                                  \
        block_type_partial_charge,                                   \
        block_type_inter_repr_path_distance,                         \
        pose_ind,                                                    \
        r_dat,                                                       \
        n_atoms_to_load,                                             \
        start_atom,                                                  \
        count_pair_striking_dist,                                    \
        conn_ats);                                                   \
  }

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
      ElecScoringData<Real>& inter_dat,                              \
      shared_mem_union& shared) {                                    \
    elec_load_tile_invariant_interres_data<DeviceOperations, D, nt>( \
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

#define LOAD_INTERRES1_TILE_DATA_TO_SHARED                            \
  TMOL_DEVICE_FUNC(                                                   \
      int tile_ind,                                                   \
      int start_atom1,                                                \
      int n_atoms_to_load1,                                           \
      ElecScoringData<Real>& inter_dat,                               \
      shared_mem_union& shared) {                                     \
    elec_load_interres1_tile_data_to_shared<DeviceOperations, D, nt>( \
        rot_coords,                                                   \
        block_type_partial_charge,                                    \
        block_type_inter_repr_path_distance,                          \
        tile_ind,                                                     \
        start_atom1,                                                  \
        n_atoms_to_load1,                                             \
        inter_dat,                                                    \
        shared.m);                                                    \
  }

#define LOAD_INTERRES2_TILE_DATA_TO_SHARED                            \
  TMOL_DEVICE_FUNC(                                                   \
      int tile_ind,                                                   \
      int start_atom2,                                                \
      int n_atoms_to_load2,                                           \
      ElecScoringData<Real>& inter_dat,                               \
      shared_mem_union& shared) {                                     \
    elec_load_interres2_tile_data_to_shared<DeviceOperations, D, nt>( \
        rot_coords,                                                   \
        block_type_partial_charge,                                    \
        block_type_inter_repr_path_distance,                          \
        tile_ind,                                                     \
        start_atom2,                                                  \
        n_atoms_to_load2,                                             \
        inter_dat,                                                    \
        shared.m);                                                    \
  }

#define LOAD_INTERRES_DATA_FROM_SHARED \
  TMOL_DEVICE_FUNC(int, int, shared_mem_union&, ElecScoringData<Real>&) {}

#define EVAL_INTERRES_ATOM_PAIR_SCORES                                      \
  TMOL_DEVICE_FUNC(                                                         \
      ElecScoringData<Real>& inter_dat, int start_atom1, int start_atom2) { \
    auto eval_scores_for_atom_pairs = ([&](int tid) {                       \
      auto elecE = tmol::score::common::InterResBlockEvaluation<            \
          ElecScoringData,                                                  \
          AllAtomPairSelector,                                              \
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
              score_inter_elec_atom_pair,                                   \
              inter_dat);                                                   \
      inter_dat.total_elec += std::get<0>(elecE);                           \
    });                                                                     \
    DeviceOperations<D>::template for_each_in_workgroup<nt>(                \
        eval_scores_for_atom_pairs);                                        \
  }

// In block-pair scoring mode, we are safe to simply write the calculated
// energies to the output tensor, since each block-pair energy is assigned
// to its own CTA: no need for an atomic-add call.
#define STORE_CALCULATED_POSE_ENERGIES                                        \
  TMOL_DEVICE_FUNC(                                                           \
      ElecScoringData<Real>& score_dat, shared_mem_union& shared) {           \
    auto reduce_energies = ([&](int tid) {                                    \
      Real const cta_total_elec =                                             \
          DeviceOperations<D>::template reduce_in_workgroup<nt>(              \
              score_dat.total_elec, shared, mgpu::plus_t<Real>());            \
      if (tid == 0) {                                                         \
        if (!output_block_pair_energies) {                                    \
          accumulate<D, Real>::add(                                           \
              output[0][score_dat.pose_ind][0][0], cta_total_elec);           \
        } else {                                                              \
          int const p = score_dat.pose_ind;                                   \
          int const b1 = score_dat.block_ind1;                                \
          int const b2 = score_dat.block_ind2;                                \
          output[0][p][b1][b2] = cta_total_elec;                              \
        }                                                                     \
      }                                                                       \
    });                                                                       \
    DeviceOperations<D>::template for_each_in_workgroup<nt>(reduce_energies); \
  }

#define STORE_CALCULATED_ROTAMER_ENERGIES                                     \
  TMOL_DEVICE_FUNC(                                                           \
      ElecScoringData<Real>& score_dat, shared_mem_union& shared) {           \
    auto reduce_energies = ([&](int tid) {                                    \
      Real const cta_total_elec =                                             \
          DeviceOperations<D>::template reduce_in_workgroup<nt>(              \
              score_dat.total_elec, shared, mgpu::plus_t<Real>());            \
      if (tid == 0) {                                                         \
        output[0][cta] = cta_total_elec;                                      \
      }                                                                       \
    });                                                                       \
    DeviceOperations<D>::template for_each_in_workgroup<nt>(reduce_energies); \
  }

#define LOAD_TILE_INVARIANT_INTRARES_DATA                            \
  TMOL_DEVICE_FUNC(                                                  \
      int pose_ind,                                                  \
      int rot_ind1,                                                  \
      int block_ind1,                                                \
      int block_type1,                                               \
      int n_atoms1,                                                  \
      ElecScoringData<Real>& intra_dat,                              \
      shared_mem_union& shared) {                                    \
    elec_load_tile_invariant_intrares_data<DeviceOperations, D, nt>( \
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

#define LOAD_INTRARES1_TILE_DATA_TO_SHARED                            \
  TMOL_DEVICE_FUNC(                                                   \
      int tile_ind,                                                   \
      int start_atom1,                                                \
      int n_atoms_to_load1,                                           \
      ElecScoringData<Real>& intra_dat,                               \
      shared_mem_union& shared) {                                     \
    elec_load_intrares1_tile_data_to_shared<DeviceOperations, D, nt>( \
        rot_coords,                                                   \
        block_type_partial_charge,                                    \
        tile_ind,                                                     \
        start_atom1,                                                  \
        n_atoms_to_load1,                                             \
        intra_dat,                                                    \
        shared.m);                                                    \
  }

#define LOAD_INTRARES2_TILE_DATA_TO_SHARED                            \
  TMOL_DEVICE_FUNC(                                                   \
      int tile_ind,                                                   \
      int start_atom2,                                                \
      int n_atoms_to_load2,                                           \
      ElecScoringData<Real>& intra_dat,                               \
      shared_mem_union& shared) {                                     \
    elec_load_intrares2_tile_data_to_shared<DeviceOperations, D, nt>( \
        rot_coords,                                                   \
        block_type_partial_charge,                                    \
        tile_ind,                                                     \
        start_atom2,                                                  \
        n_atoms_to_load2,                                             \
        intra_dat,                                                    \
        shared.m);                                                    \
  }

#define LOAD_INTRARES_DATA_FROM_SHARED              \
  TMOL_DEVICE_FUNC(                                 \
      int tile_ind1,                                \
      int tile_ind2,                                \
      shared_mem_union& shared,                     \
      ElecScoringData<Real>& intra_dat) {           \
    elec_load_intrares_data_from_shared(            \
        tile_ind1, tile_ind2, shared.m, intra_dat); \
  }

#define EVAL_INTRARES_ATOM_PAIR_SCORES                                      \
  TMOL_DEVICE_FUNC(                                                         \
      ElecScoringData<Real>& intra_dat, int start_atom1, int start_atom2) { \
    auto eval_scores_for_atom_pairs = ([&](int tid) {                       \
      auto elecE = tmol::score::common::IntraResBlockEvaluation<            \
          ElecScoringData,                                                  \
          AllAtomPairSelector,                                              \
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
              score_intra_elec_atom_pair,                                   \
              intra_dat);                                                   \
      intra_dat.total_elec += std::get<0>(elecE);                           \
    });                                                                     \
    DeviceOperations<D>::template for_each_in_workgroup<nt>(                \
        eval_scores_for_atom_pairs);                                        \
  }