#include <c10/core/DeviceType.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/PhiloxUtils.cuh>

/*#include <THC/THCGenerator.hpp>
#include <THC/THCTensorRandom.h>*/

#include <tmol/score/common/device_operations.cuda.impl.cuh>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/torch_context.hh>

// ??? #include "annealer.hh"
#include "simulated_annealing.hh"

#include <moderngpu/cta_reduce.hxx>
#include <moderngpu/kernel_compact.hxx>
// #include <moderngpu/kernel_mergesort.hxx>
#include <moderngpu/kernel_segsort.hxx>
#include <moderngpu/transform.hxx>
#include <cooperative_groups.h>

#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

#include <ctime>

#include "compiled.impl.hh"

namespace tmol {
namespace pack {
namespace compiled {

template <unsigned int n_threads, typename T, typename op_t>
MGPU_DEVICE __inline__ T reduce_shfl_and_broadcast(
    cooperative_groups::thread_block_tile<n_threads> g, T val, op_t op) {
  for (unsigned int i = n_threads / 2; i > 0; i /= 2) {
    T const shfl_val = g.shfl_down(val, i);
    if (g.thread_rank() < 32 - i) {
      val = op(val, shfl_val);
    }
  }

  // thread 0 shares its reduced value with everyone
  // so that there is no disagreement on the partition
  // function value as a result of numerical imprecision
  T val_bcast = g.shfl(val, 0);

  return val_bcast;
}

template <tmol::Device D, typename Int, typename Real>
struct InteractionGraph {
 public:
  int max_n_rotamers_per_pose_;
  TView<Int, 1, D> pose_n_res_;
  TView<Int, 1, D> pose_n_rotamers_;
  TView<Int, 1, D> pose_rotamer_offset_;
  TView<Int, 2, D> n_rotamers_for_res_;
  TView<Int, 2, D> oneb_offsets_;
  TView<Int, 1, D> res_for_rot_;
  int32_t chunk_size_;
  TView<int64_t, 3, D> chunk_offset_offsets_;
  TView<int64_t, 1, D> chunk_offsets_;
  TView<Real, 1, D> energy1b_;
  TView<Real, 1, D> energy2b_;
  TView<Int, 2, D> n_neighbors_;  // [n_poses, max_n_res]
  TView<Int, 2, D>
      neighbor_starts_;  // [n_poses, max_n_res] exclusive prefix sum
  TView<Int, 1, D>
      neighbor_list_;  // [total_neighbors] flat neighbor block indices

  int n_poses_cpu() const { return pose_n_res_.size(0); }
  int max_n_res_cpu() const { return n_rotamers_for_res_.size(1); }
  int n_rotamers_total_cpu() const { return res_for_rot_.size(0); }
  int max_n_rotamers_per_pose_cpu() const { return max_n_rotamers_per_pose_; }

  MGPU_DEVICE
  int n_poses() const { return pose_n_res_.size(0); }

  MGPU_DEVICE
  int n_res(int pose) const { return pose_n_res_[pose]; }

  MGPU_DEVICE
  int n_rotamers(int pose) const { return pose_n_rotamers_[pose]; }

  MGPU_DEVICE
  TView<Int, 2, D> const& n_rotamers_for_res() const {
    return n_rotamers_for_res_;
  }

  MGPU_DEVICE
  TView<Int, 2, D> const& oneb_offsets() const { return oneb_offsets_; }

  MGPU_DEVICE
  TView<Int, 1, D> const& res_for_rot() const { return res_for_rot_; }

  MGPU_DEVICE
  Real energy1b(int global_rot_ind) const { return energy1b_[global_rot_ind]; }

  // Return the 1b + 2b energy for a substited rotamer at a residue
  MGPU_DEVICE
  Real rotamer_energy_against_background(
      int pose,
      int sub_res,
      int sub_res_n_rots,
      int local_sub_rot,
      int global_sub_rot,
      TensorAccessor<Int, 1, D> rotamer_assignment,
      bool this_thread_active) const {
    float new_e = 1e30;
    if (this_thread_active) {
      new_e = energy1b_[global_sub_rot];
    }
    int sub_rot_chunk = local_sub_rot / chunk_size_;
    int sub_rot_in_chunk = local_sub_rot - sub_rot_chunk * chunk_size_;
    int sub_res_n_chunks = (sub_res_n_rots - 1) / chunk_size_ + 1;
    int sub_rot_chunk_size =
        min(chunk_size_, sub_res_n_rots - chunk_size_ * sub_rot_chunk);

    if (this_thread_active) {
      int const nb_start = neighbor_starts_[pose][sub_res];
      int const n_nb = n_neighbors_[pose][sub_res];
      for (int nb_idx = 0; nb_idx < n_nb; ++nb_idx) {
        int const k = neighbor_list_[nb_start + nb_idx];
        int const local_k_rot = rotamer_assignment[k];
        int const k_chunk = local_k_rot / chunk_size_;
        int64_t const k_sub_chunk_offset_offset =
            chunk_offset_offsets_[pose][k][sub_res];
        if (k_sub_chunk_offset_offset == -1) {
          continue;
        }

        int const k_in_chunk = local_k_rot - k_chunk * chunk_size_;
        int const k_res_n_rots = n_rotamers_for_res_[pose][k];
        int const k_chunk_size =
            min(chunk_size_, k_res_n_rots - chunk_size_ * k_chunk);
        int64_t const k_sub_chunk_start = chunk_offsets_
            [k_sub_chunk_offset_offset + k_chunk * sub_res_n_chunks
             + sub_rot_chunk];

        if (k_sub_chunk_start == -1) {
          continue;
        }

        new_e += energy2b_
            [k_sub_chunk_start + sub_rot_chunk_size * k_in_chunk
             + sub_rot_in_chunk];
      }
    }
    return new_e;
  }

  template <unsigned int n_threads>
  MGPU_DEVICE Real total_energy_for_assignment_parallel(
      int pose,
      cooperative_groups::thread_block_tile<n_threads> g,
      TensorAccessor<Int, 1, D> rotamer_assignment) const {
    Real totalE = 0;
    int const n_res = pose_n_res_[pose];
    for (int i = g.thread_rank(); i < n_res; i += n_threads) {
      int const irot_local = rotamer_assignment[i];
      int const irot_global = irot_local + oneb_offsets_[pose][i];

      totalE += energy1b_[irot_global];
    }

    // TO DO: iterate across upper-triangle indices only
    for (int i = g.thread_rank(); i < n_res; i += n_threads) {
      int const irot_local = rotamer_assignment[i];
      int const irot_chunk = irot_local / chunk_size_;
      int const irot_in_chunk = irot_local - chunk_size_ * irot_chunk;
      int const ires_n_rots = n_rotamers_for_res_[pose][i];
      int const ires_n_chunks = (ires_n_rots - 1) / chunk_size_ + 1;
      int const irot_chunk_size =
          min(chunk_size_, ires_n_rots - chunk_size_ * irot_chunk);

      for (int j = i + 1; j < n_res; ++j) {
        int const jrot_local = rotamer_assignment[j];
        int64_t const ij_chunk_offset_offset =
            chunk_offset_offsets_[pose][i][j];
        if (ij_chunk_offset_offset == -1) {
          continue;
        }
        int const jrot_chunk = jrot_local / chunk_size_;
        int const jrot_in_chunk = jrot_local - chunk_size_ * jrot_chunk;

        int const jres_n_rots = n_rotamers_for_res_[pose][j];
        int const jres_n_chunks = (jres_n_rots - 1) / chunk_size_ + 1;
        int const jrot_chunk_size =
            min(chunk_size_, jres_n_rots - chunk_size_ * jrot_chunk);
        int64_t const ij_chunk_offset = chunk_offsets_
            [ij_chunk_offset_offset + irot_chunk * jres_n_chunks + jrot_chunk];
        if (ij_chunk_offset == -1) {
          continue;
        }

        float ij_energy = energy2b_
            [ij_chunk_offset + jrot_chunk_size * irot_in_chunk + jrot_in_chunk];
        totalE += ij_energy;
      }
    }
    totalE = reduce_shfl_and_broadcast(g, totalE, mgpu::plus_t<float>());
    return totalE;
  }
};

/// @brief Return a uniformly-distributed integer in the range
/// between 0 and n-1.
/// Note that curand_uniform() returns a random number in the range
/// (0,1], unlike unlike rand() returns a random number in the range
/// [0,1). Take care with curand_uniform().
MGPU_DEVICE
int curand_in_range(curandStatePhilox4_32_10_t* state, int n) {
  return int(curand_uniform(state) * n) % n;
}

template <unsigned int n_threads, typename T, typename F>
MGPU_DEVICE __inline__ T exclusive_scan_shfl(
    cooperative_groups::thread_block_tile<n_threads> g, T val, F f) {
  for (unsigned int i = 1; i <= n_threads; i *= 2) {
    T const shfl_val = g.shfl_up(val, i);
    if (i < g.thread_rank()) {
      val = f(shfl_val, val);
    }
  }
  val = g.shfl_up(val, 1);
  if (g.thread_rank() == 0) {
    val = 0;
  }
  return val;
}

template <unsigned int n_threads, typename T, typename F>
MGPU_DEVICE __inline__ T inclusive_scan_shfl(
    cooperative_groups::thread_block_tile<n_threads> g, T val, F f) {
  for (unsigned int i = 1; i <= n_threads; i *= 2) {
    T const shfl_val = g.shfl_up(val, i);
    if (g.thread_rank() >= i) {
      val = f(shfl_val, val);
    }
  }
  return val;
}

template <tmol::Device D>
MGPU_DEVICE void set_quench_order(
    TensorAccessor<int, 1, D> quench_order,
    int n_rotamers,
    int rotamer_offset,
    curandStatePhilox4_32_10_t* state) {
  // Create a random permutation of all the rotamers
  // and visit them in this order to ensure all of them
  // are seen during the quench step

  for (int i = 0; i < n_rotamers; ++i) {
    quench_order[i] = i + rotamer_offset;
  }
  for (int i = 0; i <= n_rotamers - 2; ++i) {
    int rand_offset = curand_in_range(state, n_rotamers - i);
    int j = i + rand_offset;
    // swap i and j;
    int jval = quench_order[j];
    quench_order[j] = quench_order[i];
    quench_order[i] = jval;
  }
}

template <tmol::Device D>
MGPU_DEVICE int set_quench_32_order(
    int n_residues,
    TensorAccessor<int, 1, D> n_rotamers_for_res,
    TensorAccessor<int, 1, D> oneb_offsets,
    TensorAccessor<int, 1, D> quench_order,
    curandStatePhilox4_32_10_t* state) {
  // Create a random permutation of all the rotamers
  // and visit them in this order to ensure all of them
  // are seen during the quench step
  // int const n_residues = n_rotamers_for_res.size(0);
  // int const n_rots = quench_order.size(0);
  int count_n_quench_rots = 0;
  for (int i = 0; i < n_residues; ++i) {
    int const i_n_rots = n_rotamers_for_res[i];
    int const i_offset = oneb_offsets[i];
    int const i_n_quench_rots = (i_n_rots - 1) / 31 + 1;
    for (int j = 0; j < i_n_quench_rots; j++) {
      quench_order[count_n_quench_rots + j] = i_offset + 31 * j;
    }
    count_n_quench_rots += i_n_quench_rots;
  }
  for (int i = 0; i <= count_n_quench_rots - 2; ++i) {
    int rand_offset = curand_in_range(state, count_n_quench_rots - i);
    int j = i + rand_offset;
    // swap i and j;
    int jval = quench_order[j];
    quench_order[j] = quench_order[i];
    quench_order[i] = jval;
  }
  return count_n_quench_rots;
}

// Metropolis SA: now one warp per trajectory.
//   inner step:
//     thread 0 picks a candidate rotamer
//     all 32 threads cooperatively sum _dE_ over neighbor list
//     warp-reduce sum gives total dE
//     standard Metropolis accept/reject
template <tmol::Device D, uint n_threads, typename Int, typename Real>
MGPU_DEVICE float warp_wide_sim_annealing(
    int pose,
    int traj_id,  // debugging purposes only
    curandStatePhilox4_32_10_t* state,
    cooperative_groups::thread_block_tile<n_threads> g,
    InteractionGraph<D, Int, Real> ig,
    TensorAccessor<Int, 1, D> current_rotamer_assignment,
    TensorAccessor<Int, 1, D> best_rotamer_assignment,
    TensorAccessor<Int, 1, D> quench_order,
    float hi_temp,
    float lo_temp,
    int n_outer_iterations,
    int n_inner_iterations,
    int n_quench_iterations,
    bool quench_on_last_iteration,
    bool quench_lite) {
  float const cooling_factor = 0.35f;
  int32_t const chunk_size = ig.chunk_size_;
  int const n_res = ig.n_res(pose);
  int const n_rotamers = ig.n_rotamers(pose);
  int const pose_rotamer_offset = ig.pose_rotamer_offset_[pose];

  float temperature = hi_temp;
  float best_energy = ig.total_energy_for_assignment_parallel(
      pose, g, current_rotamer_assignment);
  float current_total_energy = best_energy;
  int n_steps = 0;

  for (int i = 0; i < n_outer_iterations; ++i) {
    bool quench = false;
    int quench_period = n_rotamers;
    int i_n_inner_iterations = n_inner_iterations;

    if (i == n_outer_iterations - 1 && quench_on_last_iteration) {
      i_n_inner_iterations = n_quench_iterations;
      quench = true;
      temperature = 0.0f;
      // recover the lowest energy rotamer assignment encountered
      // and begin quench from there
      for (int j = g.thread_rank(); j < n_res; j += 32) {
        current_rotamer_assignment[j] = best_rotamer_assignment[j];
      }
      current_total_energy = ig.total_energy_for_assignment_parallel(
          pose, g, current_rotamer_assignment);
    }

    for (int j = 0; j < i_n_inner_iterations; ++j) {
      // --- Pick one candidate rotamer (thread 0, broadcast) ---
      int ran_res(0), local_new_rot(0), global_new_rot(0);
      float accept_rand(0);

      if (quench) {
        if (g.thread_rank() == 0) {
          if (j % quench_period == 0) {
            if (quench_lite) {
              quench_period = set_quench_32_order(
                  n_res,
                  ig.n_rotamers_for_res_[pose],
                  ig.oneb_offsets_[pose],
                  quench_order,
                  state);
              i_n_inner_iterations = quench_period;
            } else {
              set_quench_order(
                  quench_order, n_rotamers, pose_rotamer_offset, state);
            }
          }
          int global_ran_rot = quench_order[j % quench_period];
          ran_res = ig.res_for_rot()[global_ran_rot];
          global_new_rot = global_ran_rot;
          local_new_rot = global_ran_rot - ig.oneb_offsets_[pose][ran_res];
        }
        if (j % quench_period == 0 && quench_lite) {
          i_n_inner_iterations = g.shfl(i_n_inner_iterations, 0);
        }
      } else {
        if (g.thread_rank() == 0) {
          float4 rands = curand_uniform4(state);
          int global_ran_rot =
              int(rands.x * n_rotamers) % n_rotamers + pose_rotamer_offset;
          ran_res = ig.res_for_rot()[global_ran_rot];
          int const ran_res_n_rots = ig.n_rotamers_for_res_[pose][ran_res];
          int const ran_res_offset = ig.oneb_offsets_[pose][ran_res];
          local_new_rot = int(rands.y * ran_res_n_rots) % ran_res_n_rots;
          global_new_rot = local_new_rot + ran_res_offset;
          accept_rand = rands.z;
        }
      }

      ran_res = g.shfl(ran_res, 0);
      local_new_rot = g.shfl(local_new_rot, 0);
      global_new_rot = g.shfl(global_new_rot, 0);
      accept_rand = g.shfl(accept_rand, 0);

      int const local_prev_rot = current_rotamer_assignment[ran_res];
      int const global_prev_rot =
          local_prev_rot + ig.oneb_offsets_[pose][ran_res];

      // Skip no-op substitutions
      if (local_new_rot == local_prev_rot) {
        continue;
      }

      // --- Compute dE: all 32 threads split the neighbor list ---
      int const ran_res_n_rots = ig.n_rotamers_for_res_[pose][ran_res];
      int const ran_res_n_chunks = (ran_res_n_rots - 1) / chunk_size + 1;

      int const new_chunk = local_new_rot / chunk_size;
      int const new_in_chunk = local_new_rot - chunk_size * new_chunk;
      int const new_chunk_size =
          min(chunk_size, ran_res_n_rots - chunk_size * new_chunk);

      int const prev_chunk = local_prev_rot / chunk_size;
      int const prev_in_chunk = local_prev_rot - chunk_size * prev_chunk;
      int const prev_chunk_size =
          min(chunk_size, ran_res_n_rots - chunk_size * prev_chunk);

      // Thread 0 contributes the 1b delta; others start at 0
      float delta_e =
          (g.thread_rank() == 0)
              ? ig.energy1b_[global_new_rot] - ig.energy1b_[global_prev_rot]
              : 0.0f;

      int const nb_start = ig.neighbor_starts_[pose][ran_res];
      int const n_nb = ig.n_neighbors_[pose][ran_res];

      for (int nb_idx = g.thread_rank(); nb_idx < n_nb; nb_idx += 32) {
        int const k = ig.neighbor_list_[nb_start + nb_idx];
        int const local_k_rot = current_rotamer_assignment[k];
        int const k_n_rots = ig.n_rotamers_for_res_[pose][k];
        int const k_chunk = local_k_rot / chunk_size;
        int const k_in_chunk = local_k_rot - k_chunk * chunk_size;
        int const k_chunk_size =
            min(chunk_size, k_n_rots - chunk_size * k_chunk);

        int64_t const k_offset_offset =
            ig.chunk_offset_offsets_[pose][k][ran_res];
        if (k_offset_offset == -1) continue;

        int64_t const k_new_chunk_offset =
            ig.chunk_offsets_
                [k_offset_offset + k_chunk * ran_res_n_chunks + new_chunk];
        float new_contrib = 0.0f;
        if (k_new_chunk_offset >= 0) {
          new_contrib = ig.energy2b_
                            [k_new_chunk_offset + new_chunk_size * k_in_chunk
                             + new_in_chunk];
        }

        int64_t const k_prev_chunk_offset =
            ig.chunk_offsets_
                [k_offset_offset + k_chunk * ran_res_n_chunks + prev_chunk];
        float prev_contrib = 0.0f;
        if (k_prev_chunk_offset >= 0) {
          prev_contrib = ig.energy2b_
                             [k_prev_chunk_offset + prev_chunk_size * k_in_chunk
                              + prev_in_chunk];
        }

        delta_e += new_contrib - prev_contrib;
      }

      // Warp-reduce to sum delta_e across all threads
      float const total_delta_e =
          reduce_shfl_and_broadcast(g, delta_e, mgpu::plus_t<float>());

      // --- Metropolis accept/reject (thread 0 decides, broadcasts) ---
      bool accept;
      if (g.thread_rank() == 0) {
        if (quench) {
          accept = total_delta_e < 0.0f;
        } else {
          accept = total_delta_e < 0.0f
                   || accept_rand < expf(-total_delta_e / temperature);
        }
      }
      accept = g.shfl(accept, 0);

      if (accept) {
        if (g.thread_rank() == 0) {
          current_rotamer_assignment[ran_res] = local_new_rot;
          current_total_energy += total_delta_e;
        }
        current_total_energy = g.shfl(current_total_energy, 0);
        if (current_total_energy < best_energy) {
          best_energy = current_total_energy;
          for (int k = g.thread_rank(); k < n_res; k += 32) {
            best_rotamer_assignment[k] = current_rotamer_assignment[k];
          }
        }
      }

      // Periodically recompute total energy to correct floating-point drift
      if (++n_steps > 10000) {
        n_steps = 0;
        current_total_energy = ig.total_energy_for_assignment_parallel(
            pose, g, current_rotamer_assignment);
      }

    }  // end inner loop

    // geometric cooling toward lo_temp
    temperature = cooling_factor * (temperature - lo_temp) + lo_temp;

  }  // end outer loop

  return ig.total_energy_for_assignment_parallel(
      pose, g, current_rotamer_assignment);
}

// IG must respond to
// - nres()
// - nrotamers()
// - nrotamers_for_res()
// - oneb_offsets() <-- i.e. rotamer_offset_for_res
// - res_for_rot()
// - total_energy_for_assignment_parallel(thread_group, rot_assignment)
//

template <tmol::Device D, class IG>
struct Annealer {
  static auto run_simulated_annealing(
      ContextManager& mgr, IG ig, at::CUDAGeneratorImpl* gen)
      -> std::tuple<TPack<float, 2, D>, TPack<int, 3, D> > {
    int const n_poses = ig.n_poses_cpu();
    int const max_n_res = ig.max_n_res_cpu();
    int const n_rotamers_total = ig.n_rotamers_total_cpu();
    int const max_n_rotamers = ig.max_n_rotamers_per_pose_cpu();

    // SA hyperparameters
    int const n_hitemp_simA_traj = 500;
    double const inner_iter_mult_hitemp = 1.5;
    double const inner_iter_mult_lotemp = 0.5;
    float const cooling_factor = 0.35f;

    // Temperature schedule
    float const high_temp_initial = 30.0f;
    float const low_temp_initial = 0.3f;
    float const high_temp_later = 0.2f;
    float const low_temp_later = 0.1f;
    int const n_outer_iterations_hitemp = 10;
    int const n_outer_iterations_lotemp = 10;

    int const n_hitemp_simA_threads = 32 * n_poses * n_hitemp_simA_traj;
    // Each top hitemp trajectory seeds n_lotemp_expansions lotemp trajectories
    int const n_lotemp_expansions = 10;
    float const round1_cut = 0.25f;
    float const round2_cut = 0.25f;
    int const n_lotemp_simA_traj =
        int(n_hitemp_simA_traj * n_lotemp_expansions * round1_cut);
    int const n_lotemp_simA_threads = 32 * n_poses * n_lotemp_simA_traj;
    int const n_fullquench_traj = int(n_lotemp_simA_traj * round2_cut);
    int const n_fullquench_threads = 32 * n_poses * n_fullquench_traj;
    int const n_inner_iterations_hitemp =
        int(max_n_rotamers * inner_iter_mult_hitemp);
    int const n_inner_iterations_lotemp =
        int(max_n_rotamers * inner_iter_mult_lotemp);

    int const max_traj = std::max(
        std::max(n_hitemp_simA_traj, n_lotemp_simA_traj), n_fullquench_traj);

    auto scores_hitemp_t =
        TPack<float, 2, D>::zeros({n_poses, n_hitemp_simA_traj});
    auto current_rotamer_assignments_hitemp_t =
        TPack<int, 3, D>::zeros({n_poses, n_hitemp_simA_traj, max_n_res});
    auto best_rotamer_assignments_hitemp_t =
        TPack<int, 3, D>::zeros({n_poses, n_hitemp_simA_traj, max_n_res});
    auto current_rotamer_assignments_hitemp_quenchlite_t =
        TPack<int, 3, D>::zeros({n_poses, n_hitemp_simA_traj, max_n_res});
    auto sorted_hitemp_traj_t =
        TPack<int, 2, D>::zeros({n_poses, n_hitemp_simA_traj});
    auto segment_heads_hitemp_t = TPack<int, 1, D>::zeros({n_poses});
    auto segment_heads_lotemp_t = TPack<int, 1, D>::zeros({n_poses});
    auto segment_heads_fullquench_t = TPack<int, 1, D>::zeros({n_poses});

    auto scores_lotemp_t =
        TPack<float, 2, D>::zeros({n_poses, n_lotemp_simA_traj});
    auto current_rotamer_assignments_lotemp_t =
        TPack<int, 3, D>::zeros({n_poses, n_lotemp_simA_traj, max_n_res});
    auto best_rotamer_assignments_lotemp_t =
        TPack<int, 3, D>::zeros({n_poses, n_lotemp_simA_traj, max_n_res});
    auto sorted_lotemp_traj_t =
        TPack<int, 2, D>::zeros({n_poses, n_lotemp_simA_traj});

    auto scores_fullquench_t =
        TPack<float, 2, D>::zeros({n_poses, n_fullquench_traj});
    auto current_rotamer_assignments_fullquench_t =
        TPack<int, 3, D>::zeros({n_poses, n_fullquench_traj, max_n_res});
    auto best_rotamer_assignments_fullquench_t =
        TPack<int, 3, D>::zeros({n_poses, n_fullquench_traj, max_n_res});
    auto sorted_fullquench_traj_t =
        TPack<int, 2, D>::zeros({n_poses, n_fullquench_traj});

    auto scores_final_t =
        TPack<float, 2, D>::zeros({n_poses, n_fullquench_traj});
    auto rotamer_assignments_final_t =
        TPack<int, 3, D>::zeros({n_poses, n_fullquench_traj, max_n_res});

    auto quench_order_t =
        TPack<int, 3, D>::zeros({n_poses, max_traj, max_n_rotamers});

    auto scores_hitemp = scores_hitemp_t.view;
    auto current_rotamer_assignments_hitemp =
        current_rotamer_assignments_hitemp_t.view;
    auto best_rotamer_assignments_hitemp =
        best_rotamer_assignments_hitemp_t.view;
    auto current_rotamer_assignments_hitemp_quenchlite =
        current_rotamer_assignments_hitemp_quenchlite_t.view;
    auto sorted_hitemp_traj = sorted_hitemp_traj_t.view;
    auto segment_heads_hitemp = segment_heads_hitemp_t.view;
    auto segment_heads_lotemp = segment_heads_lotemp_t.view;
    auto segment_heads_fullquench = segment_heads_fullquench_t.view;

    auto scores_lotemp = scores_lotemp_t.view;
    auto current_rotamer_assignments_lotemp =
        current_rotamer_assignments_lotemp_t.view;
    auto best_rotamer_assignments_lotemp =
        best_rotamer_assignments_lotemp_t.view;
    auto sorted_lotemp_traj = sorted_lotemp_traj_t.view;

    auto scores_fullquench = scores_fullquench_t.view;
    auto current_rotamer_assignments_fullquench =
        current_rotamer_assignments_fullquench_t.view;
    auto best_rotamer_assignments_fullquench =
        best_rotamer_assignments_fullquench_t.view;
    auto sorted_fullquench_traj = sorted_fullquench_traj_t.view;

    auto scores_final = scores_final_t.view;
    auto rotamer_assignments_final = rotamer_assignments_final_t.view;

    auto quench_order = quench_order_t.view;

    // Increment the seed (and capture the current seed) for the
    // cuda generator. The number of calls to curand per thread
    // must be known. Most curand calls are handled by thread 0,
    // so that's the one we'll count.
    //
    // We will overestimate the number of curand calls because
    // each pose might have a different number of rotamers and will
    // thus have a different number of curand calls.
    //
    // 1: initial random rotamer assignment:
    // (nres-1)/32 + 1 curand calls per thread
    //
    // Warp wide simulated annealing:
    //
    // 2: the outeriterations * inner-iterations
    // -- random rotamer picking
    // -- MC accept-reject calls:
    // n_poses * n_traj * n_outer * n_inner * 4, all performed by
    // thread 0
    //
    // 3: quench ordering during last stage:
    // 4 calls to curand per n-quench-iterations + possibly one
    // extra call to curand per max_n_rotamers if full-quench
    // or + possibly one extra call to curand per max_n_rotamers / 31
    // if quench-lite, all performed by thread 0..

    // Curand budget per thread-0 of each warp.
    int const hitemp_cnt =
        (max_n_res - 1) / 32 + 1 +  // initial random rotamer assignment
        n_outer_iterations_hitemp * n_inner_iterations_hitemp * 4
        + (max_n_rotamers * 4 + max_n_rotamers / 31);  // hitemp quench-lite

    int const lotemp_cnt =
        n_outer_iterations_lotemp * n_inner_iterations_lotemp * 4
        + (max_n_rotamers * 4 + max_n_rotamers / 31);  // lotemp quench-lite

    int const fullquench_cnt =
        max_n_rotamers * 5;  // random permutation + 4 curands per iteration

    at::PhiloxCudaState hitemp_philox_state;
    at::PhiloxCudaState lotemp_philox_state;
    at::PhiloxCudaState quench_philox_state;
    {
      std::lock_guard<std::mutex> lock(gen->mutex_);
      hitemp_philox_state = gen->philox_cuda_state(hitemp_cnt);
      lotemp_philox_state = gen->philox_cuda_state(lotemp_cnt);
      quench_philox_state = gen->philox_cuda_state(fullquench_cnt);
    }

    // Phase 1: run full SA, then score via quench-lite.
    auto hitemp_simulated_annealing = [=] MGPU_DEVICE(int thread_id) {
      auto seeds = at::cuda::philox::unpack(hitemp_philox_state);
      curandStatePhilox4_32_10_t state;
      curand_init(std::get<0>(seeds), thread_id, std::get<1>(seeds), &state);

      cooperative_groups::thread_block_tile<32> g =
          cooperative_groups::tiled_partition<32>(
              cooperative_groups::this_thread_block());
      int const cta_id = thread_id / 32;
      int const pose = cta_id / n_hitemp_simA_traj;
      int const traj_id = cta_id % n_hitemp_simA_traj;
      int const n_res = ig.n_res(pose);
      int const n_rotamers = ig.n_rotamers(pose);

      if (g.thread_rank() == 0) {
        sorted_hitemp_traj[pose][traj_id] = traj_id;
      }
      if (g.thread_rank() == 0 && traj_id == 0) {
        segment_heads_hitemp[pose] = pose * n_hitemp_simA_traj;
        segment_heads_lotemp[pose] = pose * n_lotemp_simA_traj;
        segment_heads_fullquench[pose] = pose * n_fullquench_traj;
      }

      // Random initial assignment
      for (int i = g.thread_rank(); i < n_res; i += 32) {
        int const i_n_rots = ig.n_rotamers_for_res()[pose][i];
        int chosen = int(curand_uniform(&state) * i_n_rots) % i_n_rots;
        current_rotamer_assignments_hitemp[pose][traj_id][i] = chosen;
        best_rotamer_assignments_hitemp[pose][traj_id][i] = chosen;
      }

      // Full SA run with geometric cooling
      warp_wide_sim_annealing(
          pose,
          traj_id,
          &state,
          g,
          ig,
          current_rotamer_assignments_hitemp[pose][traj_id],
          best_rotamer_assignments_hitemp[pose][traj_id],
          quench_order[pose][traj_id],
          high_temp_initial,
          low_temp_initial,
          n_outer_iterations_hitemp,
          n_inner_iterations_hitemp,
          n_rotamers,
          false,
          false);

      // Copy best state into current and quench-lite buffer
      for (int i = g.thread_rank(); i < n_res; i += 32) {
        int i_assignment = best_rotamer_assignments_hitemp[pose][traj_id][i];
        current_rotamer_assignments_hitemp[pose][traj_id][i] = i_assignment;
        current_rotamer_assignments_hitemp_quenchlite[pose][traj_id][i] =
            i_assignment;
      }

      // Quench-lite to produce a score for ranking
      float after_first_quench_lite_totalE = warp_wide_sim_annealing(
          pose,
          traj_id,
          &state,
          g,
          ig,
          current_rotamer_assignments_hitemp_quenchlite[pose][traj_id],
          best_rotamer_assignments_hitemp[pose][traj_id],
          quench_order[pose][traj_id],
          high_temp_initial,
          low_temp_initial,
          1,  // quench on the (only) iteration
          n_inner_iterations_hitemp,
          n_rotamers,
          true,
          true);
      if (g.thread_rank() == 0) {
        scores_hitemp[pose][traj_id] = after_first_quench_lite_totalE;
      }
    };

    // Phase 2: low-temp SA seeded from top hitemp trajectories (each seeds
    // n_lotemp_expansions independent lotemp runs).
    auto lotemp_simulated_annealing = [=] MGPU_DEVICE(int thread_id) {
      auto seeds = at::cuda::philox::unpack(lotemp_philox_state);
      curandStatePhilox4_32_10_t state;
      curand_init(std::get<0>(seeds), thread_id, std::get<1>(seeds), &state);

      cooperative_groups::thread_block_tile<32> g =
          cooperative_groups::tiled_partition<32>(
              cooperative_groups::this_thread_block());

      int const cta_id = thread_id / 32;
      int const pose = cta_id / n_lotemp_simA_traj;
      int const traj_id = cta_id % n_lotemp_simA_traj;
      int const source_traj =
          sorted_hitemp_traj[pose][traj_id / n_lotemp_expansions];

      int const n_res = ig.n_res(pose);
      int const n_rotamers = ig.n_rotamers(pose);

      if (g.thread_rank() == 0) {
        sorted_lotemp_traj[pose][traj_id] = traj_id;
      }

      // Seed from the selected hitemp trajectory's best state
      for (int i = g.thread_rank(); i < n_res; i += 32) {
        int i_rot = current_rotamer_assignments_hitemp[pose][source_traj][i];
        current_rotamer_assignments_lotemp[pose][traj_id][i] = i_rot;
        best_rotamer_assignments_lotemp[pose][traj_id][i] = i_rot;
      }

      // Low-temperature cooling trajectory
      warp_wide_sim_annealing(
          pose,
          traj_id,
          &state,
          g,
          ig,
          current_rotamer_assignments_lotemp[pose][traj_id],
          best_rotamer_assignments_lotemp[pose][traj_id],
          quench_order[pose][traj_id],
          high_temp_later,
          low_temp_later,
          n_outer_iterations_lotemp,
          n_inner_iterations_lotemp,
          n_rotamers,
          false,
          false);

      // Quench-lite to score for the next round of selection
      float after_lotemp_quench_lite_totalE = warp_wide_sim_annealing(
          pose,
          traj_id,
          &state,
          g,
          ig,
          current_rotamer_assignments_lotemp[pose][traj_id],
          best_rotamer_assignments_lotemp[pose][traj_id],
          quench_order[pose][traj_id],
          high_temp_later,
          low_temp_later,
          1,  // quench on the (only) iteration
          n_inner_iterations_lotemp,
          n_rotamers,
          true,
          true);
      if (g.thread_rank() == 0) {
        scores_lotemp[pose][traj_id] = after_lotemp_quench_lite_totalE;
      }
    };

    // Phase 3: full greedy quench from top lotemp trajectories.
    auto fullquench = ([=] MGPU_DEVICE(int thread_id) {
      auto seeds = at::cuda::philox::unpack(quench_philox_state);
      curandStatePhilox4_32_10_t state;
      curand_init(std::get<0>(seeds), thread_id, std::get<1>(seeds), &state);

      cooperative_groups::thread_block_tile<32> g =
          cooperative_groups::tiled_partition<32>(
              cooperative_groups::this_thread_block());
      int const cta_id = thread_id / 32;
      int const pose = cta_id / n_fullquench_traj;
      int const traj_id = cta_id % n_fullquench_traj;
      int const source_traj = sorted_lotemp_traj[pose][traj_id];

      int const n_res = ig.n_res(pose);
      int const n_rotamers = ig.n_rotamers(pose);

      for (int i = g.thread_rank(); i < n_res; i += 32) {
        int i_rot = current_rotamer_assignments_lotemp[pose][source_traj][i];
        current_rotamer_assignments_fullquench[pose][traj_id][i] = i_rot;
        best_rotamer_assignments_fullquench[pose][traj_id][i] = i_rot;
      }

      float after_full_quench_totalE = warp_wide_sim_annealing(
          pose,
          traj_id,
          &state,
          g,
          ig,
          current_rotamer_assignments_fullquench[pose][traj_id],
          best_rotamer_assignments_fullquench[pose][traj_id],
          quench_order[pose][traj_id],
          high_temp_later,
          low_temp_later,
          1,  // quench on the (only) iteration
          n_inner_iterations_lotemp,
          n_rotamers,
          true,
          false);
      if (g.thread_rank() == 0) {
        scores_fullquench[pose][traj_id] = after_full_quench_totalE;
      }
    });

    auto final_reindexing = ([=] MGPU_DEVICE(int thread_id) {
      cooperative_groups::thread_block_tile<32> g =
          cooperative_groups::tiled_partition<32>(
              cooperative_groups::this_thread_block());
      int const cta_id = thread_id / 32;
      int const pose = cta_id / n_fullquench_traj;
      int const traj_id = cta_id % n_fullquench_traj;
      int const source_traj = sorted_fullquench_traj[pose][traj_id];
      int const n_res = ig.n_res(pose);
      if (g.thread_rank() == 0) {
        scores_final[pose][traj_id] = scores_fullquench[pose][source_traj];
      }
      for (int i = g.thread_rank(); i < n_res; i += 32) {
        rotamer_assignments_final[pose][traj_id][i] =
            best_rotamer_assignments_fullquench[pose][source_traj][i];
      }
    });

    std::shared_ptr<mgpu::standard_context_t> context = current_context(mgr);

    mgpu::transform<32, 1>(
        hitemp_simulated_annealing, n_hitemp_simA_threads, *context);

    mgpu::segmented_sort(
        scores_hitemp.data(),
        sorted_hitemp_traj.data(),
        n_hitemp_simA_traj,
        segment_heads_hitemp.data(),
        n_poses,
        mgpu::less_t<float>(),
        *context);

    mgpu::transform<32, 1>(
        lotemp_simulated_annealing, n_lotemp_simA_threads, *context);

    mgpu::segmented_sort(
        scores_lotemp.data(),
        sorted_lotemp_traj.data(),
        n_lotemp_simA_traj,
        segment_heads_lotemp.data(),
        n_poses,
        mgpu::less_t<float>(),
        *context);

    mgpu::transform<32, 1>(fullquench, n_fullquench_threads, *context);

    mgpu::segmented_sort(
        scores_fullquench.data(),
        sorted_fullquench_traj.data(),
        n_fullquench_traj,
        segment_heads_fullquench.data(),
        n_poses,
        mgpu::less_t<float>(),
        *context);

    mgpu::transform<32, 1>(final_reindexing, n_fullquench_threads, *context);

    return {scores_final_t, rotamer_assignments_final_t};
  }
};

template <tmol::Device D>
auto AnnealerDispatch<D>::forward(
    ContextManager& mgr,
    int max_n_rotamers_per_pose,
    TView<int, 1, D> pose_n_res,
    TView<int, 1, D> pose_n_rotamers,
    TView<int, 1, D> pose_rotamer_offset,
    TView<int, 2, D> n_rotamers_for_res,
    TView<int, 2, D> oneb_offsets,
    TView<int, 1, D> res_for_rot,
    int32_t chunk_size,
    TView<int64_t, 3, D> chunk_offset_offsets,
    TView<int64_t, 1, D> chunk_offsets,
    TView<float, 1, D> energy1b,
    TView<float, 1, D> energy2b)
    -> std::tuple<TPack<float, 2, D>, TPack<int, 3, D> > {
  clock_t start = clock();

  int const n_poses_cpu = pose_n_res.size(0);
  int const max_n_res_cpu = chunk_offset_offsets.size(1);

  // Build CSR neighbor list from chunk_offset_offsets.
  // chunk_offset_offsets[pose][b1][b2] == -1 means no interaction.
  // It is already symmetric, so we iterate each row to find neighbors.

  auto n_neighbors_t = TPack<int, 2, D>::zeros({n_poses_cpu, max_n_res_cpu});
  auto n_neighbors = n_neighbors_t.view;
  std::shared_ptr<mgpu::standard_context_t> context = current_context(mgr);

  {
    int const count = n_poses_cpu * max_n_res_cpu;
    mgpu::transform<128, 1>(
        [=] MGPU_DEVICE(int idx) {
          if (idx >= count) return;
          int pose = idx / max_n_res_cpu;
          int b = idx % max_n_res_cpu;
          int n = pose_n_res[pose];
          int cnt = 0;
          for (int b2 = 0; b2 < n; ++b2) {
            if (b2 != b && chunk_offset_offsets[pose][b][b2] != -1) ++cnt;
          }
          n_neighbors[pose][b] = cnt;
        },
        count,
        *context);
  }

  auto neighbor_starts_t =
      TPack<int, 2, D>::zeros({n_poses_cpu, max_n_res_cpu});
  auto neighbor_starts = neighbor_starts_t.view;
  int total_neighbors;
  {
    mgpu::mem_t<int> total(1, *context, mgpu::memory_space_host);
    mgpu::scan<mgpu::scan_type_exc>(
        n_neighbors.data(),
        n_poses_cpu * max_n_res_cpu,
        neighbor_starts.data(),
        mgpu::plus_t<int>(),
        total.data(),
        *context);
    cudaStreamSynchronize(0);
    total_neighbors = total.data()[0];
  }

  auto neighbor_list_t =
      TPack<int, 1, D>::zeros({total_neighbors > 0 ? total_neighbors : 1});
  auto neighbor_list = neighbor_list_t.view;

  {
    int const count = n_poses_cpu * max_n_res_cpu;
    mgpu::transform<128, 1>(
        [=] MGPU_DEVICE(int idx) {
          if (idx >= count) return;
          int pose = idx / max_n_res_cpu;
          int b = idx % max_n_res_cpu;
          int n = pose_n_res[pose];
          int offset = neighbor_starts[pose][b];
          for (int b2 = 0; b2 < n; ++b2) {
            if (b2 != b && chunk_offset_offsets[pose][b][b2] != -1) {
              neighbor_list[offset++] = b2;
            }
          }
        },
        count,
        *context);
  }

  InteractionGraph<D, int, float> ig(
      {max_n_rotamers_per_pose,
       pose_n_res,
       pose_n_rotamers,
       pose_rotamer_offset,
       n_rotamers_for_res,
       oneb_offsets,
       res_for_rot,
       chunk_size,
       chunk_offset_offsets,
       chunk_offsets,
       energy1b,
       energy2b,
       n_neighbors,
       neighbor_starts,
       neighbor_list});

  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      std::nullopt, at::cuda::detail::getDefaultCUDAGenerator());

  auto result =
      Annealer<D, InteractionGraph<D, int, float> >::run_simulated_annealing(
          mgr, ig, gen);

  return result;
}

template struct AnnealerDispatch<tmol::Device::CUDA>;

template struct InteractionGraphBuilder<
    score::common::DeviceOperations,
    tmol::Device::CUDA,
    float,
    int64_t>;
template struct InteractionGraphBuilder<
    score::common::DeviceOperations,
    tmol::Device::CUDA,
    double,
    int64_t>;

}  // namespace compiled
}  // namespace pack
}  // namespace tmol
