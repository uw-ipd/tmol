#pragma once

namespace tmol {
namespace pack {
namespace compiled {

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
Real
total_energy_for_assignment(
  TView<Int, 1, D> nrotamers_for_res,
  TView<Int, 1, D> oneb_offsets,
  TView<Int, 1, D> res_for_rot,
  TView<Int, 2, D> nenergies,
  TView<Int, 2, D> twob_offsets,
  TView<Real, 1, D> energy1b,
  TView<Real, 1, D> energy2b,
  TensorAccessor<Int, 1, D> rotamer_assignment
)
{
  Real totalE = 0;
  int const nres = nrotamers_for_res.size(0);
  for (int i = 1; i < nres; ++i) {
    int const irot_local = rotamer_assignment[i];
    int const irot_global = irot_local + oneb_offsets[i];
    
    totalE += energy1b[irot_global];
    for (int j = i+1; j < nres; ++j) {
      int const jrot_local = rotamer_assignment[j];
      if (nenergies[i][j] == 0) continue;
      totalE += energy2b[
	twob_offsets[i][j]
	+ nrotamers_for_res[j] * irot_local
	+ jrot_local
      ];
    }
  }
  return totalE;
}

} // namespace compiled
} // namespace pack
} // namespace tmol
