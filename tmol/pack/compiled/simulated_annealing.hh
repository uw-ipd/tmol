#pragma once

namespace tmol {
namespace pack {
namespace compiled {

#ifdef __CUDACC__
__device__
#endif
inline
bool
pass_metropolis(
  float const kT,
  float const uniform_random,
  float const deltaE,
  bool const quench
)
{
  if ( deltaE < 0 ) return true;
  if (quench) return false;

  float prob_pass = std::exp( -1 * deltaE / kT );
  return uniform_random < prob_pass;
}

template<
  tmol::Device D,
  typename Real,
  typename Int
>
inline
#ifdef __CUDACC__
__device__
#endif
Real
total_energy_for_assignment(
  TView<Int, 1, D> nrotamers_for_res,
  TView<Int, 1, D> oneb_offsets,
  TView<Int, 1, D> res_for_rot,
  TView<Int, 2, D> nenergies,
  TView<Int, 2, D> twob_offsets,
  TView<Real, 1, D> energy1b,
  TView<Real, 1, D> energy2b,
  TView<Int, 2, D> rotamer_assignment, 
  TView<float, 3, D> pair_energies,
  int rotassign_dim0 // i.e. thread_id
)
{
  Real totalE = 0;
  int const nres = nrotamers_for_res.size(0);
  for (int i = 0; i < nres; ++i) {
    int const irot_local = rotamer_assignment[rotassign_dim0][i];
    int const irot_global = irot_local + oneb_offsets[i];
    
    totalE += energy1b[irot_global];
    for (int j = i+1; j < nres; ++j) {
      int const jrot_local = rotamer_assignment[rotassign_dim0][j];
      if (nenergies[i][j] == 0) {
	pair_energies[rotassign_dim0][i][j] = 0;
	pair_energies[rotassign_dim0][j][i] = 0;
	continue;
      }
      float ij_energy = energy2b[
	twob_offsets[i][j]
	+ nrotamers_for_res[j] * irot_local
	+ jrot_local
      ];
      totalE += ij_energy;
      pair_energies[rotassign_dim0][i][j] = ij_energy;
      pair_energies[rotassign_dim0][j][i] = ij_energy;
    }
  }
  return totalE;
}

template<
  tmol::Device D,
  typename Real,
  typename Int
>
inline
#ifdef __CUDACC__
__device__
#endif
Real
total_energy_for_assignment(
  TView<Int, 1, D> nrotamers_for_res,
  TView<Int, 1, D> oneb_offsets,
  TView<Int, 1, D> res_for_rot,
  TView<Int, 2, D> nenergies,
  TView<Int, 2, D> twob_offsets,
  TView<Real, 1, D> energy1b,
  TView<Real, 1, D> energy2b,
  TView<Int, 2, D> rotamer_assignment, 
  int rotassign_dim0 // i.e. thread_id
)
{
  Real totalE = 0;
  int const nres = nrotamers_for_res.size(0);
  for (int i = 0; i < nres; ++i) {
    int const irot_local = rotamer_assignment[rotassign_dim0][i];
    int const irot_global = irot_local + oneb_offsets[i];
    
    totalE += energy1b[irot_global];
    for (int j = i+1; j < nres; ++j) {
      int const jrot_local = rotamer_assignment[rotassign_dim0][j];
      if (nenergies[i][j] == 0) {
	continue;
      }
      float ij_energy = energy2b[
	twob_offsets[i][j]
	+ nrotamers_for_res[j] * irot_local
	+ jrot_local
      ];
      totalE += ij_energy;
    }
  }
  return totalE;
}

} // namespace compiled
} // namespace pack
} // namespace tmol
