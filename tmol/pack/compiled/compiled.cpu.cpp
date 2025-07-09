#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>

#include <tmol/score/common/device_operations.cpu.impl.hh>

// ??? #include "annealer.hh"
#include "simulated_annealing.hh"
#include "compiled.impl.hh"

#include <ctime>

namespace tmol {
namespace pack {
namespace compiled {

template <tmol::Device D>
void set_quench_order(TView<int, 1, D> quench_order, int const n_rots) {
  // Create a random permutation of all the rotamers
  // and visit them in this order to ensure all of them
  // are seen during the quench step
  for (int i = 0; i < n_rots; ++i) {
    quench_order[i] = i;
  }
  for (int i = 0; i <= n_rots - 2; ++i) {
    int j = i + rand() % (n_rots - i);
    // swap i and j;
    int jval = quench_order[j];
    quench_order[j] = quench_order[i];
    quench_order[i] = jval;
  }
}

template <tmol::Device D>
auto AnnealerDispatch<D>::forward(
    int max_n_rotamers_per_pose,
    TView<int, 1, D> pose_n_res,               // n-poses
    TView<int, 1, D> n_rotamers_for_pose,      // n-poses
    TView<int, 1, D> rotamer_offset_for_pose,  // n-poses
    TView<int, 2, D> n_rotamers_for_res,       // n-poses x max-n-res
    TView<int, 2, D> oneb_offsets,             // n-poses x max-n-res
    TView<int, 1, D> res_for_rot,              // n-rots
    int32_t chunk_size,
    TView<int64_t, 3, D>
        chunk_offset_offsets,            // n-poses x max-n-res x max-n-res
    TView<int64_t, 1, D> chunk_offsets,  // n-chunks-on-interacting-res
    TView<float, 1, D> energy1b,
    TView<float, 1, D> energy2b)
    -> std::tuple<TPack<float, 2, D>, TPack<int, 3, D> > {
  clock_t start = clock();

  // No Frills Simulated Annealing!
  int const n_poses = pose_n_res.size(0);
  int const max_n_res = n_rotamers_for_res.size(1);
  int const n_rotamers = res_for_rot.size(0);

  int n_traj = 1;
  int const n_outer_iterations = 20;
  int const n_inner_iterations_factor = 20;

  // move to inside for(n_poses) loop:
  // int const n_inner_iterations = n_inner_iterations_factor * n_rotamers;

  auto scores_t = TPack<float, 2, D>::zeros({n_poses, n_traj});
  auto current_rotamer_assignments_t =
      TPack<int, 3, D>::zeros({n_poses, n_traj, max_n_res});
  auto best_rotamer_assignments_t =
      TPack<int, 3, D>::zeros({n_poses, n_traj, max_n_res});
  auto quench_order_t = TPack<int, 1, D>::zeros({n_rotamers});
  // auto rotamer_attempts_t = TPack<int, 1, D>::zeros({n_rotamers});

  auto scores = scores_t.view;
  auto current_rotamer_assignments = current_rotamer_assignments_t.view;
  auto best_rotamer_assignments = best_rotamer_assignments_t.view;
  auto quench_order = quench_order_t.view;
  // auto rotamer_attempts = rotamer_attempts_t.view;

  float const high_temp = 100;
  float const low_temp = 0.2;

  for (int pose = 0; pose < n_poses; ++pose) {
    int const n_res = pose_n_res[pose];
    int const pose_n_rotamers = n_rotamers_for_pose[pose];
    int const pose_rotamer_offset = rotamer_offset_for_pose[pose];
    int const n_inner_iterations = n_inner_iterations_factor * pose_n_rotamers;

    for (int traj = 0; traj < n_traj; ++traj) {
      // std::cout << "Starting trajectory " << traj+1 << std::endl;

      // Initial assignment: assign a rotamer to every residue
      for (int i = 0; i < max_n_res; ++i) {
        int const i_n_rots = n_rotamers_for_res[pose][i];
        int rand_rot = rand() % i_n_rots;
        current_rotamer_assignments[pose][traj][i] = rand_rot;
        best_rotamer_assignments[pose][traj][i] = rand_rot;
      }
      // std::cout << "Assigned random rotamers to all residues" << std::endl;

      float temperature = high_temp;
      double best_energy = total_energy_for_assignment(
          n_rotamers_for_res[pose],
          oneb_offsets[pose],
          chunk_size,
          chunk_offset_offsets[pose],
          chunk_offsets,
          energy1b,
          energy2b,
          current_rotamer_assignments[pose][traj]);
      double current_total_energy = best_energy;
      int naccepts = 0;
      for (int i = 0; i < n_outer_iterations; ++i) {
        // std::cout << "starting round " << i+1 << std::endl;
        bool quench = false;
        if (i == n_outer_iterations - 1) {
          quench = true;
          temperature = 0;
          for (int j = 0; j < n_res; ++j) {
            current_rotamer_assignments[pose][traj][j] =
                best_rotamer_assignments[pose][traj][j];
          }
          current_total_energy = total_energy_for_assignment(
              n_rotamers_for_res[pose],
              oneb_offsets[pose],
              chunk_size,
              chunk_offset_offsets[pose],
              chunk_offsets,
              energy1b,
              energy2b,
              current_rotamer_assignments[pose][traj]);
        }

        for (int j = 0; j < n_inner_iterations; ++j) {
          int global_ran_rot;
          if (quench) {
            if (j % n_rotamers == 0) {
              set_quench_order(quench_order, n_rotamers);
            }
            global_ran_rot = quench_order[j % n_rotamers];
          } else {
            global_ran_rot = rand() % pose_n_rotamers + pose_rotamer_offset;
          }
          // ++rotamer_attempts[ran_rot];

          int const ran_res = res_for_rot[global_ran_rot];
          int const local_prev_rot =
              current_rotamer_assignments[pose][traj][ran_res];
          int const ran_res_n_rots = n_rotamers_for_res[pose][ran_res];
          int const ran_res_n_chunks = (ran_res_n_rots - 1) / chunk_size + 1;
          int const ran_res_offset = oneb_offsets[pose][ran_res];
          int const local_ran_rot = global_ran_rot - ran_res_offset;
          int const ran_rot_chunk = local_ran_rot / chunk_size;
          int const prev_rot_chunk = local_prev_rot / chunk_size;
          int const ran_rot_in_chunk =
              local_ran_rot - chunk_size * ran_rot_chunk;
          int const prev_rot_in_chunk =
              local_prev_rot - chunk_size * prev_rot_chunk;
          int const ran_rot_chunk_size =
              std::min(chunk_size, ran_res_n_rots - chunk_size * ran_rot_chunk);
          int const prev_rot_chunk_size = std::min(
              chunk_size, ran_res_n_rots - chunk_size * prev_rot_chunk);
          int const global_prev_rot = local_prev_rot + ran_res_offset;

          double new_e = energy1b[global_ran_rot];
          double prev_e = energy1b[global_prev_rot];
          double deltaE = new_e - prev_e;

          // Temp: iterate across all residues instead of just the
          // neighbors of ran_rot_res
          for (int k = 0; k < n_res; ++k) {
            if (k == ran_res) continue;
            int const k_ran_chunk_offset_offset =
                chunk_offset_offsets[pose][k][ran_res];
            if (k_ran_chunk_offset_offset == -1) {
              continue;
            }
            int const local_k_rot = current_rotamer_assignments[pose][traj][k];
            int const k_n_rots = n_rotamers_for_res[pose][k];
            int const kres_n_chunks = (k_n_rots - 1) / chunk_size + 1;
            int const krot_chunk = local_k_rot / chunk_size;
            int const krot_in_chunk = local_k_rot - krot_chunk * chunk_size;
            int const krot_ranrot_chunk_offset = chunk_offsets
                [k_ran_chunk_offset_offset + krot_chunk * ran_res_n_chunks
                 + ran_rot_chunk];
            int const krot_prevrot_chunk_offset = chunk_offsets
                [k_ran_chunk_offset_offset + krot_chunk * ran_res_n_chunks
                 + prev_rot_chunk];

            double k_new_e = 0;
            double k_prev_e = 0;

            if (krot_ranrot_chunk_offset >= 0) {
              k_new_e = energy2b
                  [krot_ranrot_chunk_offset + krot_in_chunk * ran_rot_chunk_size
                   + ran_rot_in_chunk];
            }
            if (krot_prevrot_chunk_offset >= 0) {
              k_prev_e = energy2b
                  [krot_prevrot_chunk_offset
                   + krot_in_chunk * prev_rot_chunk_size + prev_rot_in_chunk];
            }
            deltaE += k_new_e - k_prev_e;
            new_e += k_new_e;
            prev_e += k_prev_e;
          }

          float const uniform_random = float(rand()) / RAND_MAX;

          if (pass_metropolis(
                  temperature, uniform_random, deltaE, prev_e, quench)) {
            current_rotamer_assignments[pose][traj][ran_res] = local_ran_rot;
            current_total_energy += deltaE;
            ++naccepts;
            if (naccepts > 1000) {
              naccepts = 0;
              float new_current_total_energy = total_energy_for_assignment(
                  n_rotamers_for_res[pose],
                  oneb_offsets[pose],
                  chunk_size,
                  chunk_offset_offsets[pose],
                  chunk_offsets,
                  energy1b,
                  energy2b,
                  current_rotamer_assignments[pose][traj]);
              current_total_energy = new_current_total_energy;
            }
            if (current_total_energy < best_energy) {
              for (int k = 0; k < n_res; ++k) {
                best_rotamer_assignments[pose][traj][k] =
                    current_rotamer_assignments[pose][traj][k];
              }
              best_energy = current_total_energy;
            }
          }

        }  // end inner loop

        // std::cout << "temperature " << temperature << " energy " <<
        //   total_energy_for_assignment(n_rotamers_for_res, oneb_offsets,
        //     res_for_rot, nenergies, twob_offsets, energy1b, energy2b,
        //     rotamer_assignments, traj) <<
        //   std::endl;

        // geometric cooling toward 0.3
        // temperature = 0.35 * (temperature - 0.3) + 0.3;
        temperature =
            (high_temp - low_temp) * std::exp(-1 * (i + 1)) + low_temp;

      }  // end outer loop

      scores[pose][traj] = total_energy_for_assignment(
          n_rotamers_for_res[pose],
          oneb_offsets[pose],
          chunk_size,
          chunk_offset_offsets[pose],
          chunk_offsets,
          energy1b,
          energy2b,
          best_rotamer_assignments[pose][traj]);
      // std::cout << "Traj " << traj << " with score " << scores[traj] <<
      // std::endl;
    }  // end trajectory loop
  }    // end pose loop

  // find the stdev of rotamer attempts
  // float variance = 0;
  // float mean = n_outer_iterations * n_inner_iterations_factor;
  // for (int i = 0; i < n_rotamers; ++i) {
  //   int iattempts = rotamer_attempts[i];
  //   variance += (iattempts - mean)*(iattempts - mean);
  // }
  // variance /= n_rotamers;
  // float sdev = std::sqrt(variance);
  // std::cout << "attempts variance" << variance << std::endl;
  // for (int i = 0; i < n_rotamers; ++i) {
  //   int iattempts = rotamer_attempts[i];
  //   if (std::abs(iattempts - mean) > 2*sdev) {
  // 	std::cout << "Rotamer " << i << " on res " << res_for_rot[i] << "
  // attempted " << 	  iattempts << " times." << std::endl;
  //   }
  // }
  clock_t stop = clock();
  std::cout << "CPU simulated annealing in "
            << ((double)stop - start) / CLOCKS_PER_SEC << " seconds"
            << std::endl;

  return {scores_t, best_rotamer_assignments_t};
}

template struct AnnealerDispatch<tmol::Device::CPU>;

template struct InteractionGraphBuilder<
    score::common::DeviceOperations,
    tmol::Device::CPU,
    float,
    int64_t>;
template struct InteractionGraphBuilder<
    score::common::DeviceOperations,
    tmol::Device::CPU,
    double,
    int64_t>;

}  // namespace compiled
}  // namespace pack
}  // namespace tmol
