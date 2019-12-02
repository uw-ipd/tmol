#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>

// ??? #include "annealer.hh"
#include "simulated_annealing.hh"

namespace tmol {
namespace pack {
namespace compiled {

template<tmol::Device D>
void
set_quench_order(
  TView<int, 1, D> quench_order
){
  // Create a random permutation of all the rotamers
  // and visit them in this order to ensure all of them
  // are seen during the quench step
  int const nrots = quench_order.size(0);
  for (int i = 0; i < nrots; ++i) {
    quench_order[i] = i;
  }
  for (int i = 0; i <= nrots-2; ++i) {
    int j = i + rand() % (nrots-i);
    // swap i and j;
    int jval = quench_order[j];
    quench_order[j] = quench_order[i];
    quench_order[i] = jval;
  }
}

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
    auto best_rotamer_assignments_t = TPack<int, 2, D>::zeros({1,nres});

    auto quench_order_t = TPack<int, 1, D>::zeros({nrotamers});

    auto scores = scores_t.view;
    auto rotamer_assignments = rotamer_assignments_t.view;
    auto best_rotamer_assignments = rotamer_assignments_t.view;
    auto quench_order = quench_order_t.view;

    // TEMP!
    return {scores_t,rotamer_assignments_t};
    
    for (int i = 0; i < nres; ++i) {
      int const i_nrots = nrotamers_for_res[i];
      rotamer_assignments[0][i] = rand() % i_nrots;
      best_rotamer_assignments[0][i] = rand() % i_nrots;
      //std::cout << "Assigning random rotamer " << rotamer_assignments[0][i] << " of " << i_nrots << std::endl;
    }

    float temperature = 100;
    float best_energy = total_energy_for_assignment(nrotamers_for_res, oneb_offsets,
      res_for_rot, nenergies, twob_offsets, energy1b, energy2b, rotamer_assignments[0]);
    float current_total_energy = best_energy;
    int naccepts = 0;
    for (int i = 0; i < 20; ++i) {

      bool quench = false;
      if (i == 19) {
	quench = true;
	temperature = 0;
	for (int j = 0; j < nres; ++j) {
	  rotamer_assignments[0][j] = best_rotamer_assignments[0][j];
	}
	current_total_energy = total_energy_for_assignment(nrotamers_for_res, oneb_offsets,
	  res_for_rot, nenergies, twob_offsets, energy1b, energy2b, rotamer_assignments[0]);
      }

      for (int j = 0; j < 20*nrotamers; ++j) {
	int ran_rot;
	if (quench) {
	  if (j % nrotamers == 0) {
	    set_quench_order(quench_order);
	  }
	  ran_rot = quench_order[j%nrotamers];
	} else {
	  ran_rot = rand() % nrotamers;
	}
	int const ran_res = res_for_rot[ran_rot];
	int const local_prev_rot = rotamer_assignments[0][ran_res];
	int const ran_res_nrots = nrotamers_for_res[ran_res];
	int const local_ran_rot = ran_rot - oneb_offsets[ran_res];
	int const prev_rot = local_prev_rot + ran_res_nrots;

	//std::cout << "Consider substitution " << ran_rot << " " << ran_res << " " << ran_res_nrots << " " << local_prev_rot << " " << local_ran_rot << " " << ran_res_nrots << std::endl;

	float new_e = energy1b[ran_rot];
	float prev_e = energy1b[prev_rot];

	// Temp: iterate across all residues instead of just the
	// neighbors of ran_rot_res
	for (int k=0; k < nres; ++k) {
	  if (k == ran_res) continue;
	  if (nenergies[ran_res][k] == 0) continue;
	  int const local_k_rot = rotamer_assignments[0][k];


	  //int const ran_k_offset = twob_offsets[ran_res][k];
	  int const k_ran_offset = twob_offsets[k][ran_res];
	  int const kres_nrots = nrotamers_for_res[k];
	  //new_e += energy2b[ran_k_offset + kres_nrots * local_ran_rot + local_k_rot];
	  new_e += energy2b[k_ran_offset + ran_res_nrots * local_k_rot + local_ran_rot];
	  prev_e += energy2b[k_ran_offset + ran_res_nrots * local_k_rot + local_prev_rot];
	}

	float const uniform_random = random();
	float const deltaE = new_e - prev_e;
	if (local_prev_rot < 0 || pass_metropolis(temperature, uniform_random, deltaE, quench)) {
	  rotamer_assignments[0][ran_res] = local_ran_rot;
	  current_total_energy = current_total_energy + deltaE;
	  ++naccepts;
	  if (naccepts > 1000) {
	    naccepts = 0;
	    current_total_energy = total_energy_for_assignment(
	      nrotamers_for_res, oneb_offsets, res_for_rot,
	      nenergies, twob_offsets, energy1b, energy2b,
	      rotamer_assignments[0]);
	  }
	  if (current_total_energy < best_energy) {
	    for (int k=0; k < nres; ++k) {
	      best_rotamer_assignments[0][k] = rotamer_assignments[0][k];
	    }
	    best_energy = current_total_energy;
	  }
	}
      }
      // geometric cooling toward 0.3
      std::cout << "temperature " << temperature << " energy " <<
	total_energy_for_assignment(nrotamers_for_res, oneb_offsets,
	  res_for_rot, nenergies, twob_offsets, energy1b, energy2b, rotamer_assignments[0]) << std::endl;
      temperature = 0.35 * (temperature - 0.3) + 0.3;
    }

    
    scores[0] = total_energy_for_assignment(nrotamers_for_res, oneb_offsets,
      res_for_rot, nenergies, twob_offsets, energy1b, energy2b, rotamer_assignments[0]);

    return {scores_t, rotamer_assignments_t};
  }

};

template struct AnnealerDispatch<tmol::Device::CPU>;

} // namespace compiled
} // namespace pack
} // namespace tmol
