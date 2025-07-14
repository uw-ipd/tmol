#include <c10/core/DeviceType.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/PhiloxUtils.cuh>

/*#include <THC/THCGenerator.hpp>
#include <THC/THCTensorRandom.h>*/

#include <tmol/score/common/device_operations.cuda.impl.cuh>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>

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

// Stolen from torch, v1.0.0
// Expose part of the torch library that otherwise is
// not part of the API.
// THCGenerator* THCRandom_getGenerator(THCState* state);

// Stolen from torch, v1.0.0;
// unnecessary in the latest release, where this function
// is built in to CUDAGenerator.
// Modified slightly as the input Generator is unused.
// increment should be at least the number of curand() random numbers used in
// each thread.
/*std::pair<uint64_t, uint64_t> next_philox_seed(uint64_t increment) {
  auto gen_ = THCRandom_getGenerator(at::globalContext().getTHCState());
  uint64_t offset = gen_->state.philox_seed_offset.fetch_add(increment);
  return std::make_pair(gen_->state.initial_seed, offset);
}*/

namespace tmol {
namespace pack {
namespace compiled {

template <unsigned int n_threads, typename T, typename op_t>
MGPU_DEVICE __inline__ T reduce_shfl_and_broadcast(
    cooperative_groups::thread_block_tile<n_threads> g, T val, op_t op) {
  // T val_orig(val);
  // mgpu::shfl_reduce_t<T, n_threads> reducer;
  // val = reducer. template reduce<op_t>(
  //   g.thread_rank(), val, n_threads, op);
  //
  // T hand_rolled_val(val_orig);
  for (unsigned int i = n_threads / 2; i > 0; i /= 2) {
    T const shfl_val = g.shfl_down(val, i);
    if (g.thread_rank() < 32 - i) {
      val = op(val, shfl_val);
    }
  }

  // thread 0 shares its reduced value with everyone
  // so that there is no disagreement on the
  // partition function value
  T val_bcast = g.shfl(val, 0);

  // printf("%d %d shfl orig %f reduce %f bcast %f vs %f\n", g.thread_rank(),
  // threadIdx.x, float(val_orig), float(val), float(val_bcast),
  // float(hand_rolled_val));

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

    // Temp: iterate across all residues instead of just the
    // neighbors of ran_rot_res
    if (this_thread_active) {
      for (int k = 0; k < n_res(pose); ++k) {
        if (k == sub_res) {
          continue;
        }
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
        // printf("%d inds %d %d, %lld, %lld, %d * %d + %d = %d; e2b[%lld] = %f
        // \n",
        //   threadIdx.x, sub_res, k,
        //   chunk_offset_offsets_[pose][k][sub_res],
        //   k_sub_chunk_start, sub_rot_chunk_size, k_in_chunk,
        //   sub_rot_in_chunk, sub_rot_chunk_size * k_in_chunk +
        //   sub_rot_in_chunk, k_sub_chunk_start + sub_rot_chunk_size *
        //   k_in_chunk + sub_rot_in_chunk, new_e
        // );
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
  int const n_res = ig.n_res(pose);
  int const n_rotamers = ig.n_rotamers(pose);
  int const pose_rotamer_offset = ig.pose_rotamer_offset_[pose];

  float temperature = hi_temp;
  float best_energy = ig.total_energy_for_assignment_parallel(
      pose, g, current_rotamer_assignment);
  float current_total_energy = best_energy;
  int n_trials = 0;
  for (int i = 0; i < n_outer_iterations; ++i) {
    // if (g.thread_rank() == 0 && traj_id == 0) {
    //   printf("p %d t %d top of outer loop %d currentE %f bestE %f temp %f\n",
    // 	pose, traj_id, i, current_total_energy, best_energy, temperature);
    // }
    bool quench = false;
    int quench_period = n_rotamers;
    int i_n_inner_iterations = n_inner_iterations;

    if (i == n_outer_iterations - 1 && quench_on_last_iteration) {
      i_n_inner_iterations = n_quench_iterations;
      quench = true;
      temperature = 1e-20;
      // recover the lowest energy rotamer assignment encountered
      // and begin quench from there
      for (int j = g.thread_rank(); j < n_res; j += 32) {
        current_rotamer_assignment[j] = best_rotamer_assignment[j];
      }
      current_total_energy = ig.total_energy_for_assignment_parallel(
          pose, g, current_rotamer_assignment);
    }

    for (int j = 0; j < i_n_inner_iterations; ++j) {
      int ran_rot(0);
      float accept_prob(0);
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
          ran_rot = quench_order[j % n_rotamers];
        }
        ran_rot = g.shfl(ran_rot, 0);
        if (j % quench_period == 0 && quench_lite) {
          i_n_inner_iterations = g.shfl(i_n_inner_iterations, 0);
        }
        accept_prob = .5;
      } else {
        if (g.thread_rank() == 0) {
          // TO DO: Make more efficient by having each thread call curand and
          // then broadcast to other threads their rngs % 32; also, use all 4
          // rands and not just the first two.
          float4 four_rands = curand_uniform4(state);
          ran_rot =
              int(four_rands.x * n_rotamers) % n_rotamers + pose_rotamer_offset;
          accept_prob = four_rands.y;
        }
        ran_rot = g.shfl(ran_rot, 0);
        accept_prob = g.shfl(accept_prob, 0);
      }
      int const ran_res = ig.res_for_rot()[ran_rot];
      int const local_prev_rot = current_rotamer_assignment[ran_res];
      int const ran_res_n_rots = ig.n_rotamers_for_res()[pose][ran_res];
      int const ran_res_rotamer_offset = ig.oneb_offsets()[pose][ran_res];

      bool prev_rot_in_range = false;
      int thread_w_prev_rot = 0;
      {  // scope
        int const local_ran_rot_orig = ran_rot - ran_res_rotamer_offset;
        int const local_prev_rot_wrapped =
            local_ran_rot_orig < local_prev_rot
                ? local_prev_rot
                : local_prev_rot + ran_res_n_rots;
        prev_rot_in_range = local_ran_rot_orig + 32 > local_prev_rot_wrapped;
        thread_w_prev_rot =
            prev_rot_in_range ? local_prev_rot_wrapped - local_ran_rot_orig : 0;
      }
      int const local_ran_rot =
          prev_rot_in_range
              ? ((ran_rot - ran_res_rotamer_offset + g.thread_rank())
                 % ran_res_n_rots)
              : (g.thread_rank() == 0
                     ? local_prev_rot
                     : (ran_rot - ran_res_rotamer_offset + g.thread_rank() - 1)
                           % ran_res_n_rots);
      ran_rot = local_ran_rot + ran_res_rotamer_offset;

      // If there are fewer rotamers on this residue than there are threads
      // active in the warp, do not wrap and consider a rotamer more than once
      bool const this_thread_active = ran_res_n_rots > g.thread_rank();
      bool const this_thread_last_active =
          ran_res_n_rots == g.thread_rank() || g.thread_rank() == 32 - 1;

      float new_e = ig.rotamer_energy_against_background(
          pose,
          ran_res,
          ran_res_n_rots,
          local_ran_rot,
          ran_rot,
          current_rotamer_assignment,
          this_thread_active);

      // if (g.thread_rank() == 0) {
      //   printf("minimum<float>\n");
      // }
      float const min_e =
          reduce_shfl_and_broadcast(g, new_e, mgpu::minimum_t<float>());
      // if (traj_id == 0) {
      // 	printf("thread %d new_e %f min_e %f\n", g.thread_rank(), new_e,
      // min_e);
      // }
      float myexp = expf(-1 * (new_e - min_e) / temperature);
      // printf("thread %d myexp %f\n", thread_id, myexp);
      // if (g.thread_rank() == 0) {
      //   printf("plus<float>\n");
      // }
      float const partition =
          reduce_shfl_and_broadcast(g, myexp, mgpu::plus_t<float>());
      // printf("thread %d partition %f\n", thread_id, partition);
      float const myprob = this_thread_active ? myexp / partition : 0;
      // printf("thread %d myprob %f\n", thread_id, myprob);
      // if (g.thread_rank() == 0) {
      //   printf("inclusive scan plus<float>\n");
      // }
      float scan_prob = inclusive_scan_shfl(g, myprob, mgpu::plus_t<float>());
      // printf("thread %d prev rotamer %d new rotamer %d new_e %f active? %d
      // temp %f\n", thread_id, local_prev_rot, local_ran_rot, new_e,
      // this_thread_active, temperature); printf("thread %d myexp %f part %f
      // myprob %f scan_prob %f accept_prob %f\n", thread_id, myexp, partition,
      // myprob, scan_prob, accept_prob);
      if (this_thread_last_active) {
        // due to numerical imprecision, it's entirely likely that the scan
        // probability for the last active thread to be slightly more or
        // slightly less than 1, and we want to ensure that there's a winner for
        // each thread.
        scan_prob = 1;
      }
      int accept_rank = (this_thread_active && accept_prob <= scan_prob);
      // printf("thread %d accept_rank %d\n", thread_id, accept_rank);
      // if (g.thread_rank() == 0) {
      //   printf("inclusive scan plus<int>\n");
      // }
      accept_rank = inclusive_scan_shfl(g, accept_rank, mgpu::plus_t<int>());
      // printf("thread %d accept_rank after scan %d\n", thread_id,
      // accept_rank);

      bool accept = accept_rank == 1 && this_thread_active;
      // printf("thread %d accept %d\n", thread_id, accept);
      // if (g.thread_rank() == 0) {
      //   printf("max<int>\n");
      // }
      int const accept_thread = reduce_shfl_and_broadcast(
          g, accept ? int(g.thread_rank()) : int(-1), mgpu::maximum_t<int>());
      // if (g.thread_rank() == 0) {
      //   printf("thread %d accept_thread %d\n", thread_id, accept_thread);
      // }

      float prev_e = g.shfl(new_e, thread_w_prev_rot);
      // printf("thread %d prev_e %f\n", thread_id, prev_e);

      bool new_best = false;
      if (accept) {
        float deltaE = new_e - prev_e;
        // if (traj_id == 0 ) {
        //   printf("deltaE: %f (%f - %f)\n", deltaE, new_e, prev_e);
        // }
        current_rotamer_assignment[ran_res] = local_ran_rot;
        current_total_energy = current_total_energy + deltaE;
        // for (int k=0; k < nres; ++k) {
        //   float k_energy = alt_energies[k][thread_id];
        //   curr_pair_energies[ran_res][k][thread_id] = k_energy;
        //   curr_pair_energies[k][ran_res][thread_id] = k_energy;
        // }
        if (current_total_energy < best_energy) {
          new_best = true;
          best_energy = current_total_energy;
        }
      }
      current_total_energy = g.shfl(current_total_energy, accept_thread);
      new_best = g.shfl(new_best, accept_thread);
      if (new_best) {
        for (int k = g.thread_rank(); k < n_res; k += 32) {
          best_rotamer_assignment[k] = current_rotamer_assignment[k];
        }
        best_energy =
            current_total_energy;  // g.shfl(best_energy, accept_thread);
      }

      ++n_trials;
      if (n_trials > 1000) {
        n_trials = 0;
        current_total_energy = ig.total_energy_for_assignment_parallel(
            pose, g, current_rotamer_assignment);
        // if (g.thread_rank() == 0) {
        //   printf("refresh total energy currentE %f\n", current_total_energy);
        // }
      }

    }  // end inner loop

    // geometric cooling toward 0.3
    // std::cout << "temperature " << temperature << " energy " <<
    //  total_energy_for_assignment(n_rotamers_for_res, oneb_offsets,
    //    res_for_rot, nenergies, twob_offsets, energy1b, energy2b,
    //    my_rotamer_assignment) << std::endl;
    temperature = 0.35 * (temperature - lo_temp) + lo_temp;

  }  // end outer loop

  float totalE = ig.total_energy_for_assignment_parallel(
      pose, g, current_rotamer_assignment);

  return totalE;
}

// template <tmol::Device D, uint n_threads, class Int, class Real>
// MGPU_DEVICE float spbr(
//     curandStatePhilox4_32_10_t* state,
//     cooperative_groups::thread_block_tile<n_threads> g,
//     InteractionGraph<D, Int, Real> ig,
//     int warp_id,
//     int n_spbr,
//     TView<int, 2, D> spbr_rotamer_assignments,
//     TView<int, 2, D> spbr_perturbed_assignments) {
//   int const nres = ig.n_rotamers_for_res.size(0);
//   int const nrotamers = ig.res_for_rot.size(0);
//
//   float energy = ig.total_energy_for_assignment_parallel(
//       g, spbr_rotamer_assignments[warp_id]);
//
//   for (int spbr_iteration = 0; spbr_iteration < n_spbr; ++spbr_iteration) {
//     // 1. pick a rotamer
//     int ran_rot;
//     if (g.thread_rank() == 0) {
//       float rand_num = curand_uniform(state);
//       ran_rot = int(rand_num * nrotamers) % nrotamers;
//     }
//     ran_rot = g.shfl(ran_rot, 0);
//     int const ran_res = ig.res_for_rot[ran_rot];
//     int const ran_res_nrots = ig.nrotamers_for_res[ran_res];
//     int const ran_rot_local = ran_rot - ig.oneb_offsets[ran_res];
//
//     // initialize the perturbed assignments array for this iteration.
//     // many of these will be overwritten, but for memory access efficiency
//     // copy everything over now.
//     for (int i = g.thread_rank(); i < nres; i += 32) {
//       int irot =
//           i == ran_res ? ran_rot_local :
//           spbr_rotamer_assignments[warp_id][i];
//       spbr_perturbed_assignments[warp_id][i] = irot;
//     }
//
//     // 2. relax the neighbors of this residue
//     for (int i = 0; i < nres; ++i) {
//       // 4a. Find the lowest energy rotamer for residue i
//       if (ran_res == i || ig.nenergies[ran_res][i] == 0) {
//         continue;
//       }
//
//       int my_best_rot = 0;
//       float my_best_rot_E = 9999;
//       int i_nrots = ig.nrotamers_for_res[i];
//       for (int j = g.thread_rank(); j < i_nrots; j += 32) {
//         int const j_global = j + ig.oneb_offsets[i];
//         float jE = ig.energy1b[j_global];
//         for (int k = 0; k < nres; ++k) {
//           if (k == i || ig.nenergies[k][i] == 0) continue;
//
//           int const k_rotamer = k == ran_res
//                                     ? ran_rot_local
//                                     : spbr_rotamer_assignments[warp_id][k];
//           jE += ig.energy2b[ig.twob_offsets[k][i] + i_nrots * k_rotamer + j];
//         }
//
//         if (j == g.thread_rank() || jE < my_best_rot_E) {
//           my_best_rot = j;
//           my_best_rot_E = jE;
//         }
//       }
//       // now all threads compare: who has the lowest energy
//       // if (g.thread_rank() == 0) {
//       //   printf("minimum<float>\n");
//       // }
//       float best_rot_E =
//           reduce_shfl_and_broadcast(g, my_best_rot_E,
//           mgpu::minimum_t<float>());
//       int mine_is_best = best_rot_E == my_best_rot_E;
//       int scan_val = inclusive_scan_shfl(g, mine_is_best,
//       mgpu::plus_t<int>()); if (mine_is_best && scan_val == 1) {
//         // exactly one thread saves the assigned rotamer to the
//         // spbr_perturbed_assignemnt array
//         spbr_perturbed_assignments[warp_id][i] = my_best_rot;
//       }
//     }
//
//     // 5. compute the new total energy after relaxation
//     float alt_energy = ig.total_energy_for_assignment_parallel(
//         g, spbr_perturbed_assignments[warp_id]);
//
//     // 6. if the energy decreases, accept the perturbed conformation
//     if (alt_energy < energy) {
//       // if (g.thread_rank() == 0) {
//       //   printf("%d prevE %f newE %f\n", warp_id, energy, alt_energy);
//       // }
//
//       energy = alt_energy;
//       for (int i = g.thread_rank(); i < nres; i += 32) {
//         spbr_rotamer_assignments[warp_id][i] =
//             spbr_perturbed_assignments[warp_id][i];
//       }
//     }
//   }
//   return energy;
// }

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
  static auto run_simulated_annealing(IG ig, at::CUDAGeneratorImpl* gen)
      -> std::tuple<TPack<float, 2, D>, TPack<int, 3, D> > {
    int const n_poses = ig.n_poses_cpu();
    int const max_n_res = ig.max_n_res_cpu();  // nrotamers_for_res.size(0);
    int const n_rotamers_total =
        ig.n_rotamers_total_cpu();  // res_for_rot.size(0);
    int const max_n_rotamers = ig.max_n_rotamers_per_pose_cpu();
    // printf(
    //     "n poses: %d, max_n_res %d, n_rotamers_total %d, max_n_rotamers
    //     %d\n", n_poses, max_n_res, n_rotamers_total, max_n_rotamers);

    int const n_hitemp_simA_traj = 2000;
    int const n_hitemp_simA_threads = 32 * n_poses * n_hitemp_simA_traj;
    float const round1_cut = 0.25;
    int const n_lotemp_expansions = 10;
    int const n_lotemp_simA_traj =
        int(n_hitemp_simA_traj * n_lotemp_expansions * round1_cut);
    int const n_lotemp_simA_threads = 32 * n_poses * n_lotemp_simA_traj;
    float const round2_cut = 0.25;
    int const n_fullquench_traj = int(n_lotemp_simA_traj * round2_cut);
    int const n_fullquench_threads = 32 * n_poses * n_fullquench_traj;
    int const n_outer_iterations_hitemp = 10;
    int const n_inner_iterations_hitemp = max_n_rotamers / 8;
    int const n_outer_iterations_lotemp = 10;
    int const n_inner_iterations_lotemp = max_n_rotamers / 16;
    float const high_temp_initial = 30;
    float const low_temp_initial = 0.3;
    float const high_temp_later = 0.2;
    float const low_temp_later = 0.1;

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
    auto segment_heads_hitemp_t = TPack<int, 1, D>::zeros(
        {n_poses});  // arange(n_poses) * n_hitemp_simA_traj
    auto segment_heads_lotemp_t = TPack<int, 1, D>::zeros(
        {n_poses});  // arange(n_poses) * n_lotemp_simA_traj
    auto segment_heads_fullquench_t = TPack<int, 1, D>::zeros(
        {n_poses});  // arange(n_poses) * n_fulquench_traj

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
        TPack<int, 2, D>::zeros({n_poses, n_hitemp_simA_traj});

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
    // auto rotamer_assignments_lotemp_quenchlite =
    // rotamer_assignments_lotemp_quenchlite_t.view;
    auto sorted_lotemp_traj = sorted_lotemp_traj_t.view;

    auto scores_fullquench = scores_fullquench_t.view;
    auto current_rotamer_assignments_fullquench =
        current_rotamer_assignments_fullquench_t.view;
    auto best_rotamer_assignments_fullquench =
        best_rotamer_assignments_fullquench_t.view;
    auto sorted_fullquench_traj = sorted_fullquench_traj_t.view;
    // auto sorted_fullquench_traj = sorted_lotem_traj_t.view;

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

    int const hitemp_cnt =
        (max_n_res - 1) / 32 + 1 +  // initial random rotamer assignment
        n_outer_iterations_hitemp * n_inner_iterations_hitemp * 4
        +  // hitemp annealing; curand4
        (max_n_rotamers * 4
         + max_n_rotamers
               / 31);  // hitemp random permutation of quenchlite rotamers

    int const lotemp_cnt =
        n_outer_iterations_lotemp * n_outer_iterations_lotemp * 4
        +  // lotemp annealing
        (max_n_rotamers * 4
         + max_n_rotamers
               / 31);  // lotemp random permuation of quenchlite rotamers

    int const fullquench_cnt =
        max_n_rotamers * 5;  // random permutation + 4 curands per iteration

    // Increment the cuda generator
    at::PhiloxCudaState hitemp_philox_state;
    at::PhiloxCudaState lotemp_philox_state;
    at::PhiloxCudaState quench_philox_state;
    {
      std::lock_guard<std::mutex> lock(gen->mutex_);
      hitemp_philox_state = gen->philox_cuda_state(hitemp_cnt);
      lotemp_philox_state = gen->philox_cuda_state(lotemp_cnt);
      quench_philox_state = gen->philox_cuda_state(fullquench_cnt);
    }

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
      // printf("hitemp thread %d cta %d pose %d traj_id %d\n", thread_id,
      // cta_id,
      //   pose, traj_id);

      int const n_res = ig.n_res(pose);
      int const n_rotamers = ig.n_rotamers(pose);

      if (g.thread_rank() == 0) {
        sorted_hitemp_traj[pose][traj_id] = traj_id;
      }
      if (g.thread_rank() == 0 && traj_id == 0) {
        // later we will run segmented sort for the trajectories
        // for each Pose, so we need tensors of "segment heads"
        // to state the indices at which the trajectory lists
        // begin.
        segment_heads_hitemp[pose] = pose * n_hitemp_simA_traj;
        segment_heads_lotemp[pose] = pose * n_lotemp_simA_traj;
        segment_heads_fullquench[pose] = pose * n_fullquench_traj;
      }

      for (int i = g.thread_rank(); i < n_res; i += 32) {
        int const i_n_rots = ig.n_rotamers_for_res()[pose][i];
        int chosen = int(curand_uniform(&state) * i_n_rots) % i_n_rots;
        current_rotamer_assignments_hitemp[pose][traj_id][i] = chosen;
        best_rotamer_assignments_hitemp[pose][traj_id][i] = chosen;
      }

      float rotstate_energy_after_high_temp = warp_wide_sim_annealing(
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
          n_rotamers,  // irrelevant; no quench here
          false,
          false);
      // if (g.thread_rank() == 0) {
      //   printf(
      //       "hitemp wwsa done: pose %d traj %d E = %f\n",
      //       pose,
      //       traj_id,
      //       rotstate_energy_after_high_temp);
      // }

      // Save the state before moving into quench
      for (int i = g.thread_rank(); i < n_res; i += 32) {
        int i_assignment = best_rotamer_assignments_hitemp[pose][traj_id][i];
        current_rotamer_assignments_hitemp[pose][traj_id][i] = i_assignment;
        current_rotamer_assignments_hitemp_quenchlite[pose][traj_id][i] =
            i_assignment;
      }
      float best_energy_after_high_temp =
          ig.total_energy_for_assignment_parallel(
              pose, g, best_rotamer_assignments_hitemp[pose][traj_id]);

      // ok, run quench lite as a way to predict where this rotamer assignment
      // will end up after low-temperature annealing
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
          1,  // perform quench in first (ie last) iteration
          n_inner_iterations_hitemp,  // irrelevant
          n_rotamers,
          true,
          true);
      if (g.thread_rank() == 0) {
        scores_hitemp[pose][traj_id] = after_first_quench_lite_totalE;
      }
    };

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

      // printf("lotemp thread %d cta %d pose %d traj_id %d source_traj %d\n",
      // thread_id, cta_id,
      //   pose, traj_id, source_traj);

      if (g.thread_rank() == 0) {
        sorted_lotemp_traj[pose][traj_id] = traj_id;
      }

      // initialize the rotamer assignment from one of the top trajectories
      // of the high-temperature annealing trajectory
      for (int i = g.thread_rank(); i < n_res; i += 32) {
        int i_rot = current_rotamer_assignments_hitemp[pose][source_traj][i];
        current_rotamer_assignments_lotemp[pose][traj_id][i] = i_rot;
        best_rotamer_assignments_lotemp[pose][traj_id][i] = i_rot;
      }

      // Now run a low-temperature cooling trajectory
      float low_temp_totalE = warp_wide_sim_annealing(
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

      // if (g.thread_rank() == 0) {
      //   printf(
      //       "lotemp wwsa done: pose %d traj %d E = %f\n",
      //       pose,
      //       traj_id,
      //       low_temp_totalE);
      // }
      // now we'll run a quench-lite
      // ok, we will run quench lite on first state
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
          1,  // run quench on first (i.e. last) iteration
          n_inner_iterations_lotemp,  // irrelevant
          n_rotamers,
          true,
          true);
      if (g.thread_rank() == 0) {
        scores_lotemp[pose][traj_id] = after_lotemp_quench_lite_totalE;
      }
    };

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
      // if (g.thread_rank() == 0) {
      //         printf("warp %d fullquench source_traj %d (%d) %f\n", warp_id,
      //         source_traj,
      //           n_lotemp_simA_traj, scores_lotemp[warp_id]);
      // }

      // initialize the rotamer assignment from one of the top trajectories
      // of the high-temperature annealing trajectory
      for (int i = g.thread_rank(); i < n_res; i += 32) {
        int i_rot = current_rotamer_assignments_lotemp[pose][source_traj][i];
        current_rotamer_assignments_fullquench[pose][traj_id][i] = i_rot;
        best_rotamer_assignments_fullquench[pose][traj_id][i] = i_rot;
      }

      float after_full_quench_totalE = 0;
      for (int i = 0; i < 1; ++i) {
        after_full_quench_totalE = warp_wide_sim_annealing(
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
            1,  // run quench on first (ie last) iteration
            n_inner_iterations_lotemp,
            n_rotamers,
            true,
            false);
      }
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

    mgpu::standard_context_t context;

    // printf("launch hitemp\n");
    mgpu::transform<32, 1>(
        hitemp_simulated_annealing, n_hitemp_simA_threads, context);

    // now let's rank the trajectories for each pose
    // printf("launch segsort\n");
    mgpu::segmented_sort(
        scores_hitemp.data(),
        sorted_hitemp_traj.data(),
        n_hitemp_simA_traj,
        segment_heads_hitemp.data(),
        n_poses,
        mgpu::less_t<float>(),
        context);

    // printf("temp no launch lotemp\n");
    mgpu::transform<32, 1>(
        lotemp_simulated_annealing, n_lotemp_simA_threads, context);

    // printf("temp no launch segsort2\n");
    mgpu::segmented_sort(
        scores_lotemp.data(),
        sorted_lotemp_traj.data(),
        n_lotemp_simA_traj,
        segment_heads_lotemp.data(),
        n_poses,
        mgpu::less_t<float>(),
        context);

    // printf("temp no launch fullquench\n");
    mgpu::transform<32, 1>(fullquench, n_fullquench_threads, context);

    mgpu::segmented_sort(
        scores_fullquench.data(),
        sorted_fullquench_traj.data(),
        n_fullquench_traj,
        segment_heads_fullquench.data(),
        n_poses,
        mgpu::less_t<float>(),
        context);

    // printf("temp no launch fullquench\n");
    mgpu::transform<32, 1>(final_reindexing, n_fullquench_threads, context);

    printf("done!\n");
    return {scores_final_t, rotamer_assignments_final_t};
  }
};

template <tmol::Device D>
auto AnnealerDispatch<D>::forward(
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
       energy2b});

  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      std::nullopt, at::cuda::detail::getDefaultCUDAGenerator());

  auto result =
      Annealer<D, InteractionGraph<D, int, float> >::run_simulated_annealing(
          ig, gen);

  cudaDeviceSynchronize();
  clock_t stop = clock();
  std::cout << "GPU simulated annealing in "
            << ((double)stop - start) / CLOCKS_PER_SEC << " seconds"
            << std::endl;

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
