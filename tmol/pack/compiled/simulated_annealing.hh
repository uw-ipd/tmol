#pragma once

#include <algorithm>  // std::min

namespace tmol {
namespace pack {
namespace compiled {

#ifdef __CUDACC__
__device__
#endif
    inline bool pass_metropolis(
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

template <tmol::Device D>
inline
#ifdef __CUDACC__
    __device__
#endif
    float total_energy_for_assignment(
        TensorAccessor<int, 1, D> n_rotamers_for_res,  // max-n-res
        TensorAccessor<int, 1, D> oneb_offsets,        // max-n-res
        int32_t const chunk_size,
        TensorAccessor<int64_t, 2, D>
            chunk_offset_offsets,            // max-n-res x max-n-res
        TView<int64_t, 1, D> chunk_offsets,  // n-interacting-chunk-pairs
        TView<float, 1, D> energy1b,         // n-rotamers-total
        TView<float, 1, D> energy2b,         // n-interacting-rotamer-pairs
        TensorAccessor<int, 1, D>
            rotamer_assignment,  // local rotamer indices; max-n-res
        TensorAccessor<float, 2, D>
            current_pair_energies  // max-n-res x max-n-res
    ) {
#ifndef __CUDACC__
  using std::min;
#endif

  // Read the energies from energ1b and energy2b for the given
  // rotamer_assignment (represented as the local indices for
  // each rotamer on each block) and record each energy in the
  // current_pair_energies table.
  float totalE = 0;
  int const n_res = n_rotamers_for_res.size(0);
  for (int i = 0; i < n_res; ++i) {
    int const irot_local = rotamer_assignment[i];

    if (irot_local == -1) {
      // unassigned rotamer or residue off the end for the Pose
      for (int j = 0; j < n_res; ++j) {
        current_pair_energies[i][j] = 0;
        current_pair_energies[j][i] = 0;
      }
      continue;
    }
    int const irot_global = irot_local + oneb_offsets[i];
    int const ires_n_rots = n_rotamers_for_res[i];
    int const ires_n_chunks = (ires_n_rots - 1) / chunk_size + 1;
    int const irot_chunk = irot_local / chunk_size;
    int const irot_in_chunk = irot_local - chunk_size * irot_chunk;
    int const irot_chunk_size =
        min(chunk_size, ires_n_rots - chunk_size * irot_chunk);

    totalE += energy1b[irot_global];
    for (int j = i + 1; j < n_res; ++j) {
      int const jrot_local = rotamer_assignment[j];
      if (jrot_local == -1) {
        // no need to zero out current_pair_energies here; that will occur on
        // the i == this-j iteration
        continue;
      }
      int64_t const ij_chunk_offset_offset = chunk_offset_offsets[i][j];
      if (ij_chunk_offset_offset == -1) {
        // Then this pair of residues do not interact
        current_pair_energies[i][j] = 0;
        current_pair_energies[j][i] = 0;
        continue;
      }
      int const jres_n_rots = n_rotamers_for_res[j];
      int const jres_n_chunks = (jres_n_rots - 1) / chunk_size + 1;
      int const jrot_chunk = jrot_local / chunk_size;
      int const jrot_in_chunk = jrot_local - chunk_size * jrot_chunk;
      int const jrot_chunk_size =
          min(chunk_size, jres_n_rots - chunk_size * jrot_chunk);

      int64_t const ij_chunk_offset =
          (chunk_offsets
               [ij_chunk_offset_offset + irot_chunk * jres_n_chunks
                + jrot_chunk]);
      if (ij_chunk_offset == -1) {
        current_pair_energies[i][j] = 0;
        current_pair_energies[j][i] = 0;
        continue;
      }

      float const ij_energy =
          (energy2b
               [ij_chunk_offset + irot_in_chunk * jrot_chunk_size
                + jrot_in_chunk]);
      totalE += ij_energy;
      current_pair_energies[i][j] = ij_energy;
      current_pair_energies[j][i] = ij_energy;
    }
  }
  return totalE;
}

template <tmol::Device D>
inline
#ifdef __CUDACC__
    __device__
#endif
    float total_energy_for_assignment(
        TensorAccessor<int, 1, D> n_rotamers_for_res,  // max-n-res
        TensorAccessor<int, 1, D> oneb_offsets,        // max-n-res
        int32_t const chunk_size,
        TensorAccessor<int64_t, 2, D>
            chunk_offset_offsets,            // max-n-res x max-n-res
        TView<int64_t, 1, D> chunk_offsets,  // n-interacting-chunk-pairs
        TView<float, 1, D> energy1b,         // n-rotamers-total
        TView<float, 1, D> energy2b,         // n-interacting-rotamer-pairs
        TensorAccessor<int, 1, D>
            rotamer_assignment  // local rotamer indices; max-n-res
    ) {
  // Read the energies from energ1b and energy2b for the given
  // rotamer_assignment (represented as the local indices for
  // each rotamer on each block)

#ifndef __CUDACC__
  using std::min;
#endif
  int const n_res = n_rotamers_for_res.size(0);

  int count_out = 0;
  float totalE = 0;
  for (int i = 0; i < n_res; ++i) {
    int const irot_local = rotamer_assignment[i];

    if (irot_local == -1) {
      // unassigned rotamer or residue off the end for the Pose
      continue;
    }
    int const irot_global = irot_local + oneb_offsets[i];
    int const ires_n_rots = n_rotamers_for_res[i];
    int const ires_n_chunks = (ires_n_rots - 1) / chunk_size + 1;
    int const irot_chunk = irot_local / chunk_size;
    int const irot_in_chunk = irot_local - chunk_size * irot_chunk;
    int const irot_chunk_size =
        min(chunk_size, ires_n_rots - chunk_size * irot_chunk);

    totalE += energy1b[irot_global];
    for (int j = i + 1; j < n_res; ++j) {
      int const jrot_local = rotamer_assignment[j];
      if (jrot_local == -1) {
        continue;
      }
      int64_t const ij_chunk_offset_offset = chunk_offset_offsets[i][j];
      if (ij_chunk_offset_offset == -1) {
        // Then this pair of residues do not interact
        continue;
      }
      int const jres_n_rots = n_rotamers_for_res[j];
      int const jres_n_chunks = (jres_n_rots - 1) / chunk_size + 1;
      int const jrot_chunk = jrot_local / chunk_size;
      int const jrot_in_chunk = jrot_local - chunk_size * jrot_chunk;
      int const jrot_chunk_size =
          min(chunk_size, jres_n_rots - chunk_size * jrot_chunk);

      int64_t const ij_chunk_offset =
          (chunk_offsets
               [ij_chunk_offset_offset + irot_chunk * jres_n_chunks
                + jrot_chunk]);
      if (ij_chunk_offset == -1) {
        continue;
      }

      float const ij_energy =
          (energy2b
               [ij_chunk_offset + irot_in_chunk * jrot_chunk_size
                + jrot_in_chunk]);
      totalE += ij_energy;
    }
  }
  return totalE;
}

}  // namespace compiled
}  // namespace pack
}  // namespace tmol
