#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>

// ??? #include "annealer.hh"
#include "simulated_annealing.hh"

#include <ctime>


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
    TView<int, 2, D> respair_nenergies,
    TView<int, 1, D> chunk_size_t,
    TView<int, 2, D> chunk_offset_offsets,
    TView<int64_t, 2, D> twob_offsets,
    TView<int, 1, D> fine_chunk_offsets,
    TView<float, 1, D> energy1b,
    TView<float, 1, D> energy2b
  )
    -> std::tuple<
      TPack<float, 2, D>,
      TPack<int, 2, D> >
  {

    clock_t start = clock();

    // No Frills Simulated Annealing!
    int const nres = nrotamers_for_res.size(0);
    int const nrotamers = res_for_rot.size(0);
    int const chunk_size = chunk_size_t[0];

    int ntraj = 1;
    int const n_outer_iterations = 20;
    int const n_inner_iterations_factor = 20;
    int const n_inner_iterations = n_inner_iterations_factor * nrotamers;

    auto scores_t = TPack<float, 2, D>::zeros({1, ntraj});
    auto rotamer_assignments_t = TPack<int, 2, D>::zeros({ntraj, nres});
    auto best_rotamer_assignments_t = TPack<int, 2, D>::zeros({ntraj, nres});
    auto quench_order_t = TPack<int, 1, D>::zeros({nrotamers});
    // auto rotamer_attempts_t = TPack<int, 1, D>::zeros({nrotamers});

    auto scores = scores_t.view;
    auto rotamer_assignments = rotamer_assignments_t.view;
    auto best_rotamer_assignments = rotamer_assignments_t.view;
    auto quench_order = quench_order_t.view;
    // auto rotamer_attempts = rotamer_attempts_t.view;

    float const high_temp = 100;
    float const low_temp = 0.2;
    
    for (int traj = 0; traj < ntraj; ++traj) {
      // std::cout << "Starting trajectory " << traj+1 << std::endl;

      for (int i = 0; i < nres; ++i) {
        int const i_nrots = nrotamers_for_res[i];
        rotamer_assignments[traj][i] = rand() % i_nrots;
        best_rotamer_assignments[traj][i] = rand() % i_nrots;
      }
      // std::cout << "Assigned random rotamers to all residues" << std::endl;

      float temperature = high_temp;
      double best_energy = total_energy_for_assignment(
	nrotamers_for_res, oneb_offsets, res_for_rot,
	respair_nenergies, chunk_size_t, chunk_offset_offsets,
	twob_offsets, fine_chunk_offsets, energy1b, energy2b,
	rotamer_assignments, traj);
      double current_total_energy = best_energy;
      int naccepts = 0;
      for (int i = 0; i < n_outer_iterations; ++i) {

	// std::cout << "starting round " << i+1 << std::endl;
        bool quench = false;
        if (i == n_outer_iterations - 1) {
          quench = true;
          temperature = 0;
          for (int j = 0; j < nres; ++j) {
            rotamer_assignments[traj][j] = best_rotamer_assignments[traj][j];
          }
          current_total_energy = total_energy_for_assignment(
	    nrotamers_for_res, oneb_offsets, res_for_rot,
	    respair_nenergies, chunk_size_t, chunk_offset_offsets,
	    twob_offsets, fine_chunk_offsets, energy1b, energy2b,
	    rotamer_assignments, traj);
        }

        for (int j = 0; j < n_inner_iterations; ++j) {
          int global_ran_rot;
          if (quench) {
            if (j % nrotamers == 0) {
              set_quench_order(quench_order);
	      
            }
            global_ran_rot = quench_order[j%nrotamers];
          } else {
            global_ran_rot = rand() % nrotamers;
          }
	  // ++rotamer_attempts[ran_rot];

          int const ran_res = res_for_rot[global_ran_rot];
          int const local_prev_rot = rotamer_assignments[traj][ran_res];
          int const ran_res_nrots = nrotamers_for_res[ran_res];
	  int const ran_res_nchunks = (ran_res_nrots-1) / chunk_size + 1;
	  int const ran_res_offset = oneb_offsets[ran_res];
          int const local_ran_rot = global_ran_rot - ran_res_offset;
	  int const ran_rot_chunk = local_ran_rot / chunk_size;
	  int const prev_rot_chunk = local_prev_rot / chunk_size;
	  int const ran_rot_in_chunk = local_ran_rot - chunk_size * ran_rot_chunk;
	  int const prev_rot_in_chunk = local_prev_rot - chunk_size * prev_rot_chunk;
	  int const ran_rot_chunk_size = std::min(chunk_size,
	    ran_res_nrots - chunk_size * ran_rot_chunk);
	  int const prev_rot_chunk_size = std::min(chunk_size,
	    ran_res_nrots - chunk_size * prev_rot_chunk);
	  int const global_prev_rot = local_prev_rot + ran_res_offset;

          double new_e = energy1b[global_ran_rot];
          double prev_e = energy1b[global_prev_rot];
	  double deltaE = new_e - prev_e;

          // Temp: iterate across all residues instead of just the
          // neighbors of ran_rot_res
          for (int k=0; k < nres; ++k) {
            if (k == ran_res) continue;
            if (respair_nenergies[ran_res][k] == 0) continue;
            int const local_k_rot = rotamer_assignments[traj][k];
	    int const k_nrots = nrotamers_for_res[k];
	    int const kres_nchunks = (k_nrots - 1) / chunk_size + 1;
	    int const krot_chunk = local_k_rot / chunk_size;
	    int const krot_in_chunk = local_k_rot - krot_chunk * chunk_size;
	    int const k_ran_chunk_offset_offset = chunk_offset_offsets[k][ran_res];
	    int const krot_ranrot_chunk_offset = fine_chunk_offsets[
	      k_ran_chunk_offset_offset
	      + krot_chunk * ran_res_nchunks
	      + ran_rot_chunk
	    ];
	    int const krot_prevrot_chunk_offset = fine_chunk_offsets[
	      k_ran_chunk_offset_offset
	      + krot_chunk * ran_res_nchunks
	      + prev_rot_chunk
	    ];
	    
            int64_t const k_ran_offset = twob_offsets[k][ran_res];

            //new_e += energy2b[ran_k_offset + kres_nrots * local_ran_rot + local_k_rot];
	    double k_new_e = 0;
	    double k_prev_e = 0;

	    if (krot_ranrot_chunk_offset >= 0) {
	      k_new_e = energy2b[
		k_ran_offset
		+ krot_ranrot_chunk_offset
		+ krot_in_chunk * ran_rot_chunk_size
		+ ran_rot_in_chunk];
	    }
	    if (krot_prevrot_chunk_offset >= 0) {
	      k_prev_e = energy2b[
		k_ran_offset
		+ krot_prevrot_chunk_offset
		+ krot_in_chunk * prev_rot_chunk_size
		+ prev_rot_in_chunk];
	    }
	    deltaE += k_new_e - k_prev_e;
	    new_e += k_new_e;
	    prev_e += k_prev_e;
          }

          float const uniform_random = float(rand()) / RAND_MAX;

	  if (pass_metropolis(temperature, uniform_random, deltaE, prev_e, quench)) {
            rotamer_assignments[traj][ran_res] = local_ran_rot;
            current_total_energy +=  deltaE;
            ++naccepts;
            if (naccepts > 1000) {
              naccepts = 0;
              float new_current_total_energy = total_energy_for_assignment(
                nrotamers_for_res, oneb_offsets, res_for_rot,
                respair_nenergies, chunk_size_t, chunk_offset_offsets,
		twob_offsets, fine_chunk_offsets, energy1b, energy2b,
                rotamer_assignments, traj
	      );
	      current_total_energy = new_current_total_energy;
            }
            if (current_total_energy < best_energy) {
              for (int k=0; k < nres; ++k) {
                best_rotamer_assignments[traj][k] = rotamer_assignments[traj][k];
              }
              best_energy = current_total_energy;
            }
          }

        } // end inner loop

	// std::cout << "temperature " << temperature << " energy " <<
	//   total_energy_for_assignment(nrotamers_for_res, oneb_offsets,
	//     res_for_rot, nenergies, twob_offsets, energy1b, energy2b,
	//     rotamer_assignments, traj) <<
	//   std::endl;

        // geometric cooling toward 0.3
        // temperature = 0.35 * (temperature - 0.3) + 0.3;
	temperature = (high_temp - low_temp) * std::exp(-1 * (i+1)) + low_temp;

      } // end outer loop


      scores[0][traj] = total_energy_for_assignment(
	nrotamers_for_res, oneb_offsets, res_for_rot,
	respair_nenergies, chunk_size_t, chunk_offset_offsets,
	twob_offsets, fine_chunk_offsets, energy1b, energy2b,
	rotamer_assignments, traj);
      // std::cout << "Traj " << traj << " with score " << scores[traj] << std::endl;
    } // end trajectory loop

    // find the stdev of rotamer attempts
    // float variance = 0;
    // float mean = n_outer_iterations * n_inner_iterations_factor;
    // for (int i = 0; i < nrotamers; ++i) {
    //   int iattempts = rotamer_attempts[i];
    //   variance += (iattempts - mean)*(iattempts - mean);
    // }
    // variance /= nrotamers;
    // float sdev = std::sqrt(variance);
    // std::cout << "attempts variance" << variance << std::endl;
    // for (int i = 0; i < nrotamers; ++i) {
    //   int iattempts = rotamer_attempts[i];
    //   if (std::abs(iattempts - mean) > 2*sdev) {
    // 	std::cout << "Rotamer " << i << " on res " << res_for_rot[i] << " attempted " <<
    // 	  iattempts << " times." << std::endl;
    //   }
    // }
    clock_t stop = clock();
    std::cout << "CPU simulated annealing in " << ((double) stop - start)/CLOCKS_PER_SEC <<
      " seconds" << std::endl;

    return {scores_t, rotamer_assignments_t};
  }

};

template struct AnnealerDispatch<tmol::Device::CPU>;

} // namespace compiled
} // namespace pack
} // namespace tmol
