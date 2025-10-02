#pragma once

namespace tmol {
namespace pack {
namespace compiled {

#ifdef __CUDACC__
__device__
#endif
    inline bool
    pass_metropolis(
        float kT,
        float uniform_random,
        float deltaE,
        float prevE,
        bool quench) {
  if (deltaE < 0) return true;
  if (quench) return false;

  // Increase the acceptance probability if the
  // original rotamer is bad
  deltaE = prevE > 1 ? deltaE / prevE : deltaE;

  float prob_pass = std::exp(-1 * deltaE / kT);
  return uniform_random < prob_pass;
}

// soon template <tmol::Device D>
// soon #ifdef __CUDACC__
// soon __device__
// soon #endif
// soon inline
// soon float
// soon setup_temperature(
// soon   int outer_loop_iteration,
// soon   int n_outer_loops,
// soon   TensorAccessor<float, 1, D> round_energies,
// soon   float high_temp,
// soon   float low_temp,
// soon   int * since_last_jump_count,
// soon   bool * quench
// soon )
// soon {
// soon   int const i = outer_loop_iteration;
// soon   if ( i == n_outer_loops - 1 ) {
// soon     *quench = true;
// soon     return 1e-20; // quench temperature
// soon   } else {
// soon     if (*since_last_jump_count >= 3) {
// soon       float avgE =
// (round_energies[i-4]+round_energies[i-3]+round_energies[i-2]) / 3; soon if
// (round_energies[i-1] - avgE > -1 ) { soon 	// energies have plateaued --
// jump them up! soon 	*since_last_jump_count = 0; soon 	return
// high_temp; soon       } soon     } soon   } soon   // then we will
// geometrically cool toward lowtemp soon   *since_last_jump_count++; soon
// return (high_temp - low_temp) * exp( -1 * (*since_last_jump_count) ) +
// low_temp; soon soon }

template <tmol::Device D>
inline
#ifdef __CUDACC__
    __device__
#endif
    float
    total_energy_for_assignment(
        TView<int, 1, D> nrotamers_for_res,
        TView<int, 1, D> oneb_offsets,
        TView<int, 1, D> res_for_rot,
        TView<int, 2, D> respair_nenergies,
        TView<int, 1, D> chunk_size_t,
        TView<int, 2, D> chunk_offset_offsets,
        TView<int64_t, 2, D> twob_offsets,
        TView<int, 1, D> fine_chunk_offsets,
        TView<float, 1, D> energy1b,
        TView<float, 1, D> energy2b,
        TView<int, 2, D> rotamer_assignment,
        TView<float, 3, D> pair_energies,
        int rotassign_dim0  // i.e. thread_id
    ) {
  float totalE = 0;
  int const nres = nrotamers_for_res.size(0);
  int const chunk_size = chunk_size_t[0];
  for (int i = 0; i < nres; ++i) {
    int const irot_local = rotamer_assignment[rotassign_dim0][i];
    int const irot_global = irot_local + oneb_offsets[i];
    int const ires_nrots = nrotamers_for_res[i];
    int const ires_nchunks = (ires_nrots - 1) / chunk_size + 1;
    int const irot_chunk = irot_local / chunk_size;
    int const irot_in_chunk = irot_local - chunk_size * irot_chunk;
    int const irot_chunk_size =
        std::min(chunk_size, ires_nrots - chunk_size * irot_chunk);

    totalE += energy1b[irot_global];
    for (int j = i + 1; j < nres; ++j) {
      int const jrot_local = rotamer_assignment[rotassign_dim0][j];
      if (respair_nenergies[i][j] == 0) {
        pair_energies[rotassign_dim0][i][j] = 0;
        pair_energies[rotassign_dim0][j][i] = 0;
        continue;
      }
      int const jres_nrots = nrotamers_for_res[j];
      int const jres_nchunks = (jres_nrots - 1) / chunk_size + 1;
      int const jrot_chunk = jrot_local / chunk_size;
      int const jrot_in_chunk = jrot_local - chunk_size * jrot_chunk;
      int const jrot_chunk_size =
          std::min(chunk_size, jres_nrots - chunk_size * jrot_chunk);

      int const ij_chunk_offset_offset = chunk_offset_offsets[i][j];
      int const ij_chunk_offset = fine_chunk_offsets
          [ij_chunk_offset_offset + irot_chunk * jres_nchunks + jrot_chunk];
      if (ij_chunk_offset < 0) {
        pair_energies[rotassign_dim0][i][j] = 0;
        pair_energies[rotassign_dim0][j][i] = 0;
      }

      float ij_energy = energy2b
          [twob_offsets[i][j] + ij_chunk_offset
           + irot_in_chunk * jrot_chunk_size + jrot_in_chunk];
      totalE += ij_energy;
      pair_energies[rotassign_dim0][i][j] = ij_energy;
      pair_energies[rotassign_dim0][j][i] = ij_energy;
    }
  }
  return totalE;
}

template <tmol::Device D>
inline
#ifdef __CUDACC__
    __device__
#endif
    float
    total_energy_for_assignment(
        TView<int, 1, D> nrotamers_for_res,
        TView<int, 1, D> oneb_offsets,
        TView<int, 1, D> res_for_rot,
        TView<int, 2, D> respair_nenergies,
        TView<int, 1, D> chunk_size_t,
        TView<int, 2, D> chunk_offset_offsets,
        TView<int64_t, 2, D> twob_offsets,
        TView<int, 1, D> fine_chunk_offsets,
        TView<float, 1, D> energy1b,
        TView<float, 1, D> energy2b,
        TView<int, 2, D> rotamer_assignment,
        int rotassign_dim0  // i.e. thread_id
    ) {
  float totalE = 0;
  int const nres = nrotamers_for_res.size(0);
  int const chunk_size = chunk_size_t[0];
  for (int i = 0; i < nres; ++i) {
    int const irot_local = rotamer_assignment[rotassign_dim0][i];
    int const irot_global = irot_local + oneb_offsets[i];
    int const ires_nrots = nrotamers_for_res[i];
    int const ires_nchunks = (ires_nrots - 1) / chunk_size + 1;
    int const irot_chunk = irot_local / chunk_size;
    int const irot_in_chunk = irot_local - chunk_size * irot_chunk;
    int const irot_chunk_size =
        std::min(chunk_size, ires_nrots - chunk_size * irot_chunk);

    totalE += energy1b[irot_global];
    for (int j = i + 1; j < nres; ++j) {
      int const jrot_local = rotamer_assignment[rotassign_dim0][j];
      if (respair_nenergies[i][j] == 0) {
        continue;
      }
      int const jres_nrots = nrotamers_for_res[j];
      int const jres_nchunks = (jres_nrots - 1) / chunk_size + 1;
      int const jrot_chunk = jrot_local / chunk_size;
      int const jrot_in_chunk = jrot_local - chunk_size * jrot_chunk;
      int const jrot_chunk_size =
          std::min(chunk_size, jres_nrots - chunk_size * jrot_chunk);

      int const ij_chunk_offset_offset = chunk_offset_offsets[i][j];
      int const ij_chunk_offset = fine_chunk_offsets
          [ij_chunk_offset_offset + irot_chunk * jres_nchunks + jrot_chunk];
      if (ij_chunk_offset < 0) {
        continue;
      }

      int64_t index = twob_offsets[i][j] + ij_chunk_offset
                      + irot_in_chunk * jrot_chunk_size + jrot_in_chunk;

      // std::cout << "twob index " << index << std::endl;
      float ij_energy = energy2b[index];
      totalE += ij_energy;
    }
  }
  return totalE;
}

}  // namespace compiled
}  // namespace pack
}  // namespace tmol
