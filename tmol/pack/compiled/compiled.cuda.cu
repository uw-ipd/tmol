#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>

#include "simulated_annealing.hh"

namespace tmol {
namespace pack {
namespace compiled {

template <tmol::Device D>
struct AnnealerDispatch
{
  static
  auto
  forward(
    TView<int, 1, D> nrotamers_for_res,
    TView<int, 1, D> oneb_offsets,
    TView<int, 1, D> res_for_rot,
    TView<int, 2, D> nenergies,
    TView<int, 2, D> twob_offsets,
    TView<float, 1, D> energy1b,
    TView<float, 1, D> energy2b
  )
    -> std::tuple<
      TPack<float, 1, D>,
      TPack<int, 2, D> >
  {
    // No Frills Simulated Annealing!
    int const nres = nrotamers_for_res.size(0);
    int const nrotamers = res_for_rot.size(0);

    auto scores_t = TPack<float, 1, D>::zeros({1});
    auto rotamer_assignments_t = TPack<int, 2, D>::zeros({1,nres});

    auto scores = scores_t.view;
    auto rotamer_assignments = rotamer_assignments_t.view;

    for (int i = 0; i < nres; ++i) {
      int const i_nrots = nrotamers_for_res[i];
      rotamer_assignments[0][i] = i_nrots * random();
    }
    
    float temperature = 100;
    for (int i = 0; i < 20; ++i) {
      for (int j = 0; j < 20*nrotamers; ++j) {
	int const ran_rot = nrotamers * random();
	int const ran_res = res_for_rot[ran_rot];
	int const local_prev_rot = rotamer_assignments[0][ran_res];
	int const ran_res_nrots = nrotamers_for_res[ran_res];
	int const local_ran_rot = ran_rot - ran_res_nrots;
	int const prev_rot = local_prev_rot + ran_res_nrots;

	float new_e = energy1b[ran_rot];
	float prev_e = energy1b[prev_rot];

	// Temp: iterate across all residues instead of just the
	// neighbors of ran_rot_res
	for (int k=0; k < nres; ++k) {
	  if ( k == ran_res ) continue;
	  int const local_k_rot = rotamer_assignments[0][k];
	  if ( local_k_rot < 0 ) continue;

	  //int const ran_k_offset = twob_offsets[ran_res][k];
	  int const k_ran_offset = twob_offsets[k][ran_res];
	  int const kres_nrots = nrotamers_for_res[k];
	  //new_e += energy2b[ran_k_offset + kres_nrots * local_ran_rot + local_k_rot];
	  new_e += energy2b[k_ran_offset + ran_res_nrots * local_k_rot + local_ran_rot];
	  prev_e += energy2b[k_ran_offset + ran_res_nrots * local_k_rot + local_prev_rot];
	}

	float const uniform_random = random();
	float const deltaE = new_e - prev_e;
	if (local_prev_rot < 0 || pass_metropolis(temperature, uniform_random, deltaE)) {
	  rotamer_assignments[0][ran_res] = local_ran_rot;
	}
      }
      // geometric cooling toward 0.3
      temperature = 0.35 * (temperature - 0.3) + 0.3;
    }

    return {scores_t, rotamer_assignments_t};
  }

};

template struct AnnealerDispatch<tmol::Device::CUDA>;

} // namespace compiled
} // namespace pack
} // namespace tmol
