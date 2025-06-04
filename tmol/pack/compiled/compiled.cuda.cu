#include <c10/core/DeviceType.h>
#include <ATen/Context.h>
/*#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <THC/THCGenerator.hpp>
#include <THC/THCTensorRandom.h>*/

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>

// ??? #include "annealer.hh"
#include "simulated_annealing.hh"

#include <moderngpu/cta_reduce.hxx>
#include <moderngpu/kernel_compact.hxx>
#include <moderngpu/kernel_mergesort.hxx>
#include <moderngpu/transform.hxx>
#include <cooperative_groups.h>

#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

#include <ctime>

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

template <tmol::Device D, typename Int, typename Real>
struct InteractionGraph {
 public:
  TView<Int, 1, D> nrotamers_for_res_;
  TView<Int, 1, D> oneb_offsets_;
  TView<Int, 1, D> res_for_rot_;
  TView<Int, 2, D> respair_nenergies_;
  TView<Int, 1, D> chunk_size_;
  TView<Int, 2, D> chunk_offset_offsets_;
  TView<int64_t, 2, D> twob_offsets_;
  TView<Int, 1, D> fine_chunk_offsets_;
  TView<Real, 1, D> energy1b_;
  TView<Real, 1, D> energy2b_;

  int nres_cpu() const { return nrotamers_for_res_.size(0); }
  int nrotamers_cpu() const { return res_for_rot_.size(0); }

  MGPU_DEVICE
  int nres() const { return nrotamers_for_res_.size(0); }

  MGPU_DEVICE
  int nrotamers() const { return res_for_rot_.size(0); }

  MGPU_DEVICE
  TView<Int, 1, D> const& nrotamers_for_res() const {
    return nrotamers_for_res_;
  }

  MGPU_DEVICE
  TView<Int, 1, D> const& oneb_offsets() const { return oneb_offsets_; }

  MGPU_DEVICE
  TView<Int, 1, D> const& res_for_rot() const { return res_for_rot_; }

  MGPU_DEVICE
  Real energy1b(int global_rot_ind) const { return energy1b_[global_rot_ind]; }

  // Return the 1b + 2b energy for a substited rotamer at a residue
  MGPU_DEVICE
  Real rotamer_energy_against_background(
      int sub_res,
      int sub_res_nrots,
      int local_sub_rot,
      int global_sub_rot,
      TensorAccessor<Int, 1, D> rotamer_assignments,
      bool this_thread_active) const {
    float new_e = 1e30;
    if (this_thread_active) {
      new_e = energy1b_[global_sub_rot];
    }
    int sub_rot_chunk = local_sub_rot / chunk_size_[0];
    int sub_rot_in_chunk = local_sub_rot - sub_rot_chunk * chunk_size_[0];
    int sub_res_nchunks = (sub_res_nrots - 1) / chunk_size_[0] + 1;
    int sub_rot_chunk_size =
        min(chunk_size_[0], sub_res_nrots - chunk_size_[0] * sub_rot_chunk);

    // Temp: iterate across all residues instead of just the
    // neighbors of ran_rot_res
    if (this_thread_active) {
      for (int k = 0; k < nres(); ++k) {
        if (k == sub_res || respair_nenergies_[sub_res][k] == 0) {
          continue;
        }
        int const local_k_rot = rotamer_assignments[k];
        int const k_chunk = local_k_rot / chunk_size_[0];
        int const k_sub_chunk_offset_offset = chunk_offset_offsets_[k][sub_res];

        int const k_in_chunk = local_k_rot - k_chunk * chunk_size_[0];
        int const k_res_nrots = nrotamers_for_res_[k];
        int const k_chunk_size =
            min(chunk_size_[0], k_res_nrots - chunk_size_[0] * k_chunk);
        int const k_sub_chunk_start = fine_chunk_offsets_
            [k_sub_chunk_offset_offset + k_chunk * sub_res_nchunks
             + sub_rot_chunk];

        if (k_sub_chunk_start < 0) {
          continue;
        }

        // printf("%d inds %d %d, %d, %d, %d * %d, %d\n", threadIdx.x, sub_res,
        // k,
        //   twob_offsets_[k][sub_res], k_sub_chunk_start, sub_rot_chunk_size,
        //   k_in_chunk, sub_rot_in_chunk);
        new_e += energy2b_
            [twob_offsets_[k][sub_res] + k_sub_chunk_start
             + sub_rot_chunk_size * k_in_chunk + sub_rot_in_chunk];
      }
    }
    return new_e;
  }

  template <unsigned int nthreads>
  MGPU_DEVICE Real total_energy_for_assignment_parallel(
      cooperative_groups::thread_block_tile<nthreads> g,
      TensorAccessor<Int, 1, D> rotamer_assignment) const {
    Real totalE = 0;
    int const nres = nrotamers_for_res_.size(0);
    for (int i = g.thread_rank(); i < nres; i += nthreads) {
      int const irot_local = rotamer_assignment[i];
      int const irot_global = irot_local + oneb_offsets_[i];

      totalE += energy1b_[irot_global];
    }

    for (int i = g.thread_rank(); i < nres; i += nthreads) {
      int const irot_local = rotamer_assignment[i];
      int const irot_chunk = irot_local / chunk_size_[0];
      int const irot_in_chunk = irot_local - chunk_size_[0] * irot_chunk;
      int const ires_nrots = nrotamers_for_res_[i];
      int const ires_nchunks = (ires_nrots - 1) / chunk_size_[0] + 1;
      int const irot_chunk_size =
          min(chunk_size_[0], ires_nrots - chunk_size_[0] * irot_chunk);

      for (int j = i + 1; j < nres; ++j) {
        int const jrot_local = rotamer_assignment[j];
        if (respair_nenergies_[i][j] == 0) {
          continue;
        }
        int const jrot_chunk = jrot_local / chunk_size_[0];
        int const jrot_in_chunk = jrot_local - chunk_size_[0] * jrot_chunk;
        int const ij_chunk_offset_offset = chunk_offset_offsets_[i][j];

        int const jres_nrots = nrotamers_for_res_[j];
        int const jres_nchunks = (jres_nrots - 1) / chunk_size_[0] + 1;
        int const jrot_chunk_size =
            min(chunk_size_[0], jres_nrots - chunk_size_[0] * jrot_chunk);
        int const ij_chunk_offset = fine_chunk_offsets_
            [ij_chunk_offset_offset + irot_chunk * jres_nchunks + jrot_chunk];
        if (ij_chunk_offset < 0) {
          continue;
        }

        float ij_energy = energy2b_
            [twob_offsets_[i][j] + ij_chunk_offset
             + jrot_chunk_size * irot_in_chunk + jrot_in_chunk];
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

template <unsigned int nthreads, typename T, typename op_t>
MGPU_DEVICE __inline__ T reduce_shfl_and_broadcast(
    cooperative_groups::thread_block_tile<nthreads> g, T val, op_t op) {
  // T val_orig(val);
  // mgpu::shfl_reduce_t<T, nthreads> reducer;
  // val = reducer. template reduce<op_t>(
  //   g.thread_rank(), val, nthreads, op);
  //
  // T hand_rolled_val(val_orig);
  for (unsigned int i = nthreads / 2; i > 0; i /= 2) {
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

template <unsigned int nthreads, typename T, typename F>
MGPU_DEVICE __inline__ T exclusive_scan_shfl(
    cooperative_groups::thread_block_tile<nthreads> g, T val, F f) {
  for (unsigned int i = 1; i <= nthreads; i *= 2) {
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

template <unsigned int nthreads, typename T, typename F>
MGPU_DEVICE __inline__ T inclusive_scan_shfl(
    cooperative_groups::thread_block_tile<nthreads> g, T val, F f) {
  for (unsigned int i = 1; i <= nthreads; i *= 2) {
    T const shfl_val = g.shfl_up(val, i);
    if (g.thread_rank() >= i) {
      val = f(shfl_val, val);
    }
  }
  return val;
}

template <tmol::Device D>
MGPU_DEVICE void set_quench_order(
    TensorAccessor<int, 1, D> quench_order, curandStatePhilox4_32_10_t* state) {
  // Create a random permutation of all the rotamers
  // and visit them in this order to ensure all of them
  // are seen during the quench step
  int const nrots = quench_order.size(0);
  for (int i = 0; i < nrots; ++i) {
    quench_order[i] = i;
  }
  for (int i = 0; i <= nrots - 2; ++i) {
    int rand_offset = curand_in_range(state, nrots - i);
    int j = i + rand_offset;
    // swap i and j;
    int jval = quench_order[j];
    quench_order[j] = quench_order[i];
    quench_order[i] = jval;
  }
}

template <tmol::Device D>
MGPU_DEVICE int set_quench_32_order(
    TView<int, 1, D> nrotamers_for_res,
    TView<int, 1, D> oneb_offsets,
    TensorAccessor<int, 1, D> quench_order,
    curandStatePhilox4_32_10_t* state) {
  // Create a random permutation of all the rotamers
  // and visit them in this order to ensure all of them
  // are seen during the quench step
  int const nresidues = nrotamers_for_res.size(0);
  int const nrots = quench_order.size(0);
  int count_n_quench_rots = 0;
  for (int i = 0; i < nresidues; ++i) {
    int const i_nrots = nrotamers_for_res[i];
    int const i_offset = oneb_offsets[i];
    int const i_nquench_rots = (i_nrots - 1) / 31 + 1;
    for (int j = 0; j < i_nquench_rots; j++) {
      quench_order[count_n_quench_rots + j] = i_offset + 31 * j;
    }
    count_n_quench_rots += i_nquench_rots;
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

template <tmol::Device D, uint nthreads, typename Int, typename Real>
MGPU_DEVICE float warp_wide_sim_annealing(
    curandStatePhilox4_32_10_t* state,
    cooperative_groups::thread_block_tile<nthreads> g,
    InteractionGraph<D, Int, Real> ig,
    int warp_id,
    TensorAccessor<Int, 1, D> rotamer_assignments,
    TensorAccessor<Int, 1, D> best_rotamer_assignments,
    TensorAccessor<Int, 1, D> quench_order,
    float hi_temp,
    float lo_temp,
    int n_outer_iterations,
    int n_inner_iterations,
    int n_quench_iterations,
    bool quench_on_last_iteration,
    bool quench_lite) {
  int const nres = ig.nres();
  int const nrotamers = ig.nrotamers();

  float temperature = hi_temp;
  float best_energy =
      ig.total_energy_for_assignment_parallel(g, rotamer_assignments);
  float current_total_energy = best_energy;
  int ntrials = 0;
  for (int i = 0; i < n_outer_iterations; ++i) {
    // if (g.thread_rank() == 0) {
    //   printf("top of outer loop %d currentE %f bestE %f temp %f\n", i,
    //   current_total_energy, best_energy, temperature);
    // }
    bool quench = false;
    int quench_period = nrotamers;
    int i_n_inner_iterations = n_inner_iterations;

    if (i == n_outer_iterations - 1 && quench_on_last_iteration) {
      i_n_inner_iterations = n_quench_iterations;
      quench = true;
      temperature = 1e-20;
      // recover the lowest energy rotamer assignment encountered
      // and begin quench from there
      for (int j = g.thread_rank(); j < nres; j += 32) {
        rotamer_assignments[j] = best_rotamer_assignments[j];
      }
      current_total_energy =
          ig.total_energy_for_assignment_parallel(g, rotamer_assignments);
    }

    for (int j = 0; j < i_n_inner_iterations; ++j) {
      int ran_rot(0);
      float accept_prob(0);
      if (quench) {
        if (g.thread_rank() == 0) {
          if (j % quench_period == 0) {
            if (quench_lite) {
              quench_period = set_quench_32_order(
                  ig.nrotamers_for_res(),
                  ig.oneb_offsets(),
                  quench_order,
                  state);
              i_n_inner_iterations = quench_period;
            } else {
              set_quench_order(quench_order, state);
            }
          }
          ran_rot = quench_order[j % nrotamers];
        }
        ran_rot = g.shfl(ran_rot, 0);
        if (j % quench_period == 0 && quench_lite) {
          i_n_inner_iterations = g.shfl(i_n_inner_iterations, 0);
        }
        accept_prob = .5;
      } else {
        if (g.thread_rank() == 0) {
          float4 four_rands = curand_uniform4(state);
          ran_rot = int(four_rands.x * nrotamers) % nrotamers;
          accept_prob = four_rands.y;
        }
        ran_rot = g.shfl(ran_rot, 0);
        accept_prob = g.shfl(accept_prob, 0);
      }
      int const ran_res = ig.res_for_rot()[ran_rot];
      int const local_prev_rot = rotamer_assignments[ran_res];
      int const ran_res_nrots = ig.nrotamers_for_res()[ran_res];
      int const ran_res_rotamer_offset = ig.oneb_offsets()[ran_res];

      bool prev_rot_in_range = false;
      int thread_w_prev_rot = 0;
      {  // scope
        int const local_ran_rot_orig = ran_rot - ran_res_rotamer_offset;
        int const local_prev_rot_wrapped = local_ran_rot_orig < local_prev_rot
                                               ? local_prev_rot
                                               : local_prev_rot + ran_res_nrots;
        prev_rot_in_range = local_ran_rot_orig + 32 > local_prev_rot_wrapped;
        thread_w_prev_rot =
            prev_rot_in_range ? local_prev_rot_wrapped - local_ran_rot_orig : 0;
      }
      int const local_ran_rot =
          prev_rot_in_range
              ? ((ran_rot - ran_res_rotamer_offset + g.thread_rank())
                 % ran_res_nrots)
              : (g.thread_rank() == 0
                     ? local_prev_rot
                     : (ran_rot - ran_res_rotamer_offset + g.thread_rank() - 1)
                           % ran_res_nrots);
      ran_rot = local_ran_rot + ran_res_rotamer_offset;

      // If there are fewer rotamers on this residue than there are threads
      // active in the warp, do not wrap and consider a rotamer more than once
      bool const this_thread_active = ran_res_nrots > g.thread_rank();
      bool const this_thread_last_active =
          ran_res_nrots == g.thread_rank() || g.thread_rank() == 32 - 1;

      float new_e = ig.rotamer_energy_against_background(
          ran_res,
          ran_res_nrots,
          local_ran_rot,
          ran_rot,
          rotamer_assignments,
          this_thread_active);

      // if (g.thread_rank() == 0) {
      //   printf("minimum<float>\n");
      // }
      float const min_e =
          reduce_shfl_and_broadcast(g, new_e, mgpu::minimum_t<float>());
      // printf("thread %d min_e %f\n", thread_id, min_e);
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
        // printf("deltaE: %f (%f - %f)\n", deltaE, new_e, prev_e);
        rotamer_assignments[ran_res] = local_ran_rot;
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
        for (int k = g.thread_rank(); k < nres; k += 32) {
          best_rotamer_assignments[k] = rotamer_assignments[k];
        }
        best_energy =
            current_total_energy;  // g.shfl(best_energy, accept_thread);
      }

      ++ntrials;
      if (ntrials > 1000) {
        ntrials = 0;
        current_total_energy =
            ig.total_energy_for_assignment_parallel(g, rotamer_assignments);
        // if (g.thread_rank() == 0) {
        //   printf("refresh total energy currentE %f\n", current_total_energy);
        // }
      }

    }  // end inner loop

    // geometric cooling toward 0.3
    // std::cout << "temperature " << temperature << " energy " <<
    //  total_energy_for_assignment(nrotamers_for_res, oneb_offsets,
    //    res_for_rot, nenergies, twob_offsets, energy1b, energy2b,
    //    my_rotamer_assignment) << std::endl;
    temperature = 0.35 * (temperature - lo_temp) + lo_temp;

  }  // end outer loop

  float totalE =
      ig.total_energy_for_assignment_parallel(g, rotamer_assignments);

  return totalE;
}

template <tmol::Device D, uint nthreads, class Int, class Real>
MGPU_DEVICE float spbr(
    curandStatePhilox4_32_10_t* state,
    cooperative_groups::thread_block_tile<nthreads> g,
    InteractionGraph<D, Int, Real> ig,
    int warp_id,
    int n_spbr,
    TView<int, 2, D> spbr_rotamer_assignments,
    TView<int, 2, D> spbr_perturbed_assignments) {
  int const nres = ig.nrotamers_for_res.size(0);
  int const nrotamers = ig.res_for_rot.size(0);

  float energy = ig.total_energy_for_assignment_parallel(
      g, spbr_rotamer_assignments[warp_id]);

  for (int spbr_iteration = 0; spbr_iteration < n_spbr; ++spbr_iteration) {
    // 1. pick a rotamer
    int ran_rot;
    if (g.thread_rank() == 0) {
      float rand_num = curand_uniform(state);
      ran_rot = int(rand_num * nrotamers) % nrotamers;
    }
    ran_rot = g.shfl(ran_rot, 0);
    int const ran_res = ig.res_for_rot[ran_rot];
    int const ran_res_nrots = ig.nrotamers_for_res[ran_res];
    int const ran_rot_local = ran_rot - ig.oneb_offsets[ran_res];

    // initialize the perturbed assignments array for this iteration.
    // many of these will be overwritten, but for memory access efficiency
    // copy everything over now.
    for (int i = g.thread_rank(); i < nres; i += 32) {
      int irot =
          i == ran_res ? ran_rot_local : spbr_rotamer_assignments[warp_id][i];
      spbr_perturbed_assignments[warp_id][i] = irot;
    }

    // 2. relax the neighbors of this residue
    for (int i = 0; i < nres; ++i) {
      // 4a. Find the lowest energy rotamer for residue i
      if (ran_res == i || ig.nenergies[ran_res][i] == 0) {
        continue;
      }

      int my_best_rot = 0;
      float my_best_rot_E = 9999;
      int i_nrots = ig.nrotamers_for_res[i];
      for (int j = g.thread_rank(); j < i_nrots; j += 32) {
        int const j_global = j + ig.oneb_offsets[i];
        float jE = ig.energy1b[j_global];
        for (int k = 0; k < nres; ++k) {
          if (k == i || ig.nenergies[k][i] == 0) continue;

          int const k_rotamer = k == ran_res
                                    ? ran_rot_local
                                    : spbr_rotamer_assignments[warp_id][k];
          jE += ig.energy2b[ig.twob_offsets[k][i] + i_nrots * k_rotamer + j];
        }

        if (j == g.thread_rank() || jE < my_best_rot_E) {
          my_best_rot = j;
          my_best_rot_E = jE;
        }
      }
      // now all threads compare: who has the lowest energy
      // if (g.thread_rank() == 0) {
      //   printf("minimum<float>\n");
      // }
      float best_rot_E =
          reduce_shfl_and_broadcast(g, my_best_rot_E, mgpu::minimum_t<float>());
      int mine_is_best = best_rot_E == my_best_rot_E;
      int scan_val = inclusive_scan_shfl(g, mine_is_best, mgpu::plus_t<int>());
      if (mine_is_best && scan_val == 1) {
        // exactly one thread saves the assigned rotamer to the
        // spbr_perturbed_assignemnt array
        spbr_perturbed_assignments[warp_id][i] = my_best_rot;
      }
    }

    // 5. compute the new total energy after relaxation
    float alt_energy = ig.total_energy_for_assignment_parallel(
        g, spbr_perturbed_assignments[warp_id]);

    // 6. if the energy decreases, accept the perturbed conformation
    if (alt_energy < energy) {
      // if (g.thread_rank() == 0) {
      //   printf("%d prevE %f newE %f\n", warp_id, energy, alt_energy);
      // }

      energy = alt_energy;
      for (int i = g.thread_rank(); i < nres; i += 32) {
        spbr_rotamer_assignments[warp_id][i] =
            spbr_perturbed_assignments[warp_id][i];
      }
    }
  }
  return energy;
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
  static auto run_simulated_annealing(IG ig, int seed)
      -> std::tuple<TPack<float, 2, D>, TPack<int, 2, D> > {
    int const nres = ig.nres_cpu();            // nrotamers_for_res.size(0);
    int const nrotamers = ig.nrotamers_cpu();  // res_for_rot.size(0);

    int const n_hitemp_simA_traj = 2000;
    int const n_hitemp_simA_threads = 32 * n_hitemp_simA_traj;
    float const round1_cut = 0.25;
    int const n_lotemp_expansions = 10;
    int const n_lotemp_simA_traj =
        int(n_hitemp_simA_traj * n_lotemp_expansions * round1_cut);
    int const n_lotemp_simA_threads = 32 * n_lotemp_simA_traj;
    float const round2_cut = 0.25;
    int const n_fullquench_traj = int(n_lotemp_simA_traj * round2_cut);
    int const n_fullquench_threads = 32 * n_fullquench_traj;
    int const n_outer_iterations_hitemp = 10;
    int const n_inner_iterations_hitemp = nrotamers / 8;
    int const n_outer_iterations_lotemp = 10;
    int const n_inner_iterations_lotemp = nrotamers / 16;
    float const high_temp_initial = 30;
    float const low_temp_initial = 0.3;
    float const high_temp_later = 0.2;
    float const low_temp_later = 0.1;

    int const max_traj = std::max(
        std::max(n_hitemp_simA_traj, n_lotemp_simA_traj), n_fullquench_traj);

    auto scores_hitemp_t = TPack<float, 1, D>::zeros(n_hitemp_simA_traj);
    auto rotamer_assignments_hitemp_t =
        TPack<int, 2, D>::zeros({n_hitemp_simA_traj, nres});
    auto best_rotamer_assignments_hitemp_t =
        TPack<int, 2, D>::zeros({n_hitemp_simA_traj, nres});
    auto rotamer_assignments_hitemp_quenchlite_t =
        TPack<int, 2, D>::zeros({n_hitemp_simA_traj, nres});
    auto sorted_hitemp_traj_t = TPack<int, 1, D>::zeros(n_hitemp_simA_traj);

    auto scores_lotemp_t = TPack<float, 1, D>::zeros(n_lotemp_simA_traj);
    auto rotamer_assignments_lotemp_t =
        TPack<int, 2, D>::zeros({n_lotemp_simA_traj, nres});
    auto best_rotamer_assignments_lotemp_t =
        TPack<int, 2, D>::zeros({n_lotemp_simA_traj, nres});
    // auto rotamer_assignments_lotemp_quenchlite_t = TPack<int, 2,
    // D>::zeros({n_lotemp_simA_traj, nres});
    auto sorted_lotemp_traj_t = TPack<int, 1, D>::zeros(n_lotemp_simA_traj);

    auto scores_fullquench_t =
        TPack<float, 2, D>::zeros({1, n_fullquench_traj});
    auto rotamer_assignments_fullquench_t =
        TPack<int, 2, D>::zeros({n_fullquench_traj, nres});
    auto best_rotamer_assignments_fullquench_t =
        TPack<int, 2, D>::zeros({n_fullquench_traj, nres});

    auto quench_order_t = TPack<int, 2, D>::zeros({max_traj, nrotamers});

    auto scores_hitemp = scores_hitemp_t.view;
    auto rotamer_assignments_hitemp = rotamer_assignments_hitemp_t.view;
    auto best_rotamer_assignments_hitemp =
        best_rotamer_assignments_hitemp_t.view;
    auto rotamer_assignments_hitemp_quenchlite =
        rotamer_assignments_hitemp_quenchlite_t.view;
    auto sorted_hitemp_traj = sorted_hitemp_traj_t.view;

    auto scores_lotemp = scores_lotemp_t.view;
    auto rotamer_assignments_lotemp = rotamer_assignments_lotemp_t.view;
    auto best_rotamer_assignments_lotemp =
        best_rotamer_assignments_lotemp_t.view;
    // auto rotamer_assignments_lotemp_quenchlite =
    // rotamer_assignments_lotemp_quenchlite_t.view;
    auto sorted_lotemp_traj = sorted_lotemp_traj_t.view;

    auto scores_fullquench = scores_fullquench_t.view;
    auto rotamer_assignments_fullquench = rotamer_assignments_fullquench_t.view;
    auto best_rotamer_assignments_fullquench =
        best_rotamer_assignments_fullquench_t.view;
    // auto sorted_fullquench_traj = sorted_lotem_traj_t.view;

    auto quench_order = quench_order_t.view;

    // This code will work for future versions of the torch/aten libraries, but
    // not this one.
    // // Increment the cuda generator
    // // I know I need to increment this, but I am unsure by how much!
    // std::pair<uint64_t, uint64_t> rng_engine_inputs;
    // at::CUDAGenerator * gen = at::cuda::detail::getDefaultCUDAGenerator();
    // {
    //   std::lock_guard<std::mutex> lock(gen->mutex_);
    //   rng_engine_inputs = gen->philox_engine_inputs(nrotamers * 400 + nres);
    // }

    // Increment the seed (and capture the current seed) for the
    // cuda generator. The number of calls to curand must be known
    // by this statement.
    // 1: nrotmaers*400 = 20 outer loop * nrotamers * 20 inner loop
    // calls to either curand_uniform or curand_uniform4 in either
    // the quench / non-quench cycles +
    // 2: nres = the initial seed state of the system is created by
    // picking a single random rotamer per residue.
    /*auto philox_seed_hitemp = next_philox_seed(
        nrotamers +  // initial random rotamer assignment
        n_outer_iterations_hitemp * n_inner_iterations_hitemp
        +  // hitemp annealing
        (nrotamers / 31
         + nres)  // hitemp random permutation of quenchlite rotamers
    );*/

    int hitemp_cnt =
        nrotamers +  // initial random rotamer assignment
        n_outer_iterations_hitemp * n_inner_iterations_hitemp
        +  // hitemp annealing
        (nrotamers / 31
         + nres);  // hitemp random permutation of quenchlite rotamers

    /*auto philox_seed_lotemp = next_philox_seed(
        n_outer_iterations_lotemp * n_outer_iterations_lotemp
        +  // lotemp annealing
        (nrotamers / 31
         + nres)  // lowtemp random permuation of quenchlite rotamers
    );*/

    int lotemp_cnt =
        n_outer_iterations_lotemp * n_outer_iterations_lotemp
        +  // lotemp annealing
        (nrotamers / 31
         + nres);  // lowtemp random permuation of quenchlite rotamers

    // auto philox_seed_quench = next_philox_seed(nrotamers);

    int quench_cnt = nrotamers;

    auto hitemp_simulated_annealing = [=] MGPU_DEVICE(int thread_id) {
      curandStatePhilox4_32_10_t state;
      curand_init(seed, thread_id, 0, &state);

      cooperative_groups::thread_block_tile<32> g =
          cooperative_groups::tiled_partition<32>(
              cooperative_groups::this_thread_block());
      int const warp_id = thread_id / 32;

      if (g.thread_rank() == 0) {
        sorted_hitemp_traj[warp_id] = warp_id;
      }

      for (int i = g.thread_rank(); i < nres; i += 32) {
        int const i_nrots = ig.nrotamers_for_res()[i];
        int chosen = int(curand_uniform(&state) * i_nrots) % i_nrots;
        rotamer_assignments_hitemp[warp_id][i] = chosen;
        best_rotamer_assignments_hitemp[warp_id][i] = chosen;
      }

      float rotstate_energy_after_high_temp = warp_wide_sim_annealing(
          &state,
          g,
          ig,
          warp_id,
          rotamer_assignments_hitemp[warp_id],
          best_rotamer_assignments_hitemp[warp_id],
          quench_order[warp_id],
          high_temp_initial,
          low_temp_initial,
          n_outer_iterations_hitemp,
          n_inner_iterations_hitemp,
          nrotamers,
          false,
          false);

      // Save the state before moving into quench
      for (int i = g.thread_rank(); i < nres; i += 32) {
        int i_assignment = best_rotamer_assignments_hitemp[warp_id][i];
        rotamer_assignments_hitemp[warp_id][i] = i_assignment;
        rotamer_assignments_hitemp_quenchlite[warp_id][i] = i_assignment;
      }
      float best_energy_after_high_temp =
          ig.total_energy_for_assignment_parallel(
              g, best_rotamer_assignments_hitemp[warp_id]);

      // ok, run quench lite as a way to predict where this rotamer assignment
      // will end up after low-temperature annealing
      float after_first_quench_lite_totalE = warp_wide_sim_annealing(
          &state,
          g,
          ig,
          warp_id,
          rotamer_assignments_hitemp_quenchlite[warp_id],
          best_rotamer_assignments_hitemp[warp_id],
          quench_order[warp_id],
          high_temp_initial,
          low_temp_initial,
          1,  // perform quench in first (ie last) iteration
          n_inner_iterations_hitemp,  // irrelevant
          nrotamers,
          true,
          true);
      if (g.thread_rank() == 0) {
        scores_hitemp[warp_id] = after_first_quench_lite_totalE;
      }
    };

    auto lotemp_simulated_annealing = [=] MGPU_DEVICE(int thread_id) {
      curandStatePhilox4_32_10_t state;
      curand_init(seed, thread_id, hitemp_cnt, &state);

      cooperative_groups::thread_block_tile<32> g =
          cooperative_groups::tiled_partition<32>(
              cooperative_groups::this_thread_block());

      int const warp_id = thread_id / 32;
      int const source_traj = sorted_hitemp_traj[warp_id / n_lotemp_expansions];

      if (g.thread_rank() == 0) {
        sorted_lotemp_traj[warp_id] = warp_id;
      }

      // initialize the rotamer assignment from one of the top trajectories
      // of the high-temperature annealing trajectory
      for (int i = g.thread_rank(); i < nres; i += 32) {
        int i_rot = rotamer_assignments_hitemp[source_traj][i];
        rotamer_assignments_lotemp[warp_id][i] = i_rot;
        best_rotamer_assignments_lotemp[warp_id][i] = i_rot;
      }

      // Now run a low-temperature cooling trajectory
      float low_temp_totalE = warp_wide_sim_annealing(
          &state,
          g,
          ig,
          warp_id,
          rotamer_assignments_lotemp[warp_id],
          best_rotamer_assignments_lotemp[warp_id],
          quench_order[warp_id],
          high_temp_later,
          low_temp_later,
          n_outer_iterations_lotemp,
          n_inner_iterations_lotemp,
          nrotamers,
          false,
          false);

      // now we'll run a quench-lite
      // ok, we will run quench lite on first state
      float after_lotemp_quench_lite_totalE = warp_wide_sim_annealing(
          &state,
          g,
          ig,
          warp_id,
          rotamer_assignments_lotemp[warp_id],
          best_rotamer_assignments_lotemp[warp_id],
          quench_order[warp_id],
          high_temp_later,
          low_temp_later,
          1,  // run quench on first (i.e. last) iteration
          n_inner_iterations_lotemp,  // irrelevant
          nrotamers,
          true,
          true);
      if (g.thread_rank() == 0) {
        scores_lotemp[warp_id] = after_lotemp_quench_lite_totalE;
      }
    };

    auto fullquench = [=] MGPU_DEVICE(int thread_id) {
      curandStatePhilox4_32_10_t state;
      curand_init(seed, thread_id, hitemp_cnt + lotemp_cnt, &state);

      cooperative_groups::thread_block_tile<32> g =
          cooperative_groups::tiled_partition<32>(
              cooperative_groups::this_thread_block());
      int const warp_id = thread_id / 32;
      int const source_traj = sorted_lotemp_traj[warp_id];
      // if (g.thread_rank() == 0) {
      //         printf("warp %d fullquench source_traj %d (%d) %f\n", warp_id,
      //         source_traj,
      //           n_lotemp_simA_traj, scores_lotemp[warp_id]);
      // }

      // initialize the rotamer assignment from one of the top trajectories
      // of the high-temperature annealing trajectory
      for (int i = g.thread_rank(); i < nres; i += 32) {
        int i_rot = rotamer_assignments_lotemp[source_traj][i];
        rotamer_assignments_fullquench[warp_id][i] = i_rot;
        best_rotamer_assignments_fullquench[warp_id][i] = i_rot;
      }

      float after_full_quench_totalE = 0;
      for (int i = 0; i < 1; ++i) {
        after_full_quench_totalE = warp_wide_sim_annealing(
            &state,
            g,
            ig,
            warp_id,
            rotamer_assignments_fullquench[warp_id],
            best_rotamer_assignments_fullquench[warp_id],
            quench_order[warp_id],
            high_temp_later,
            low_temp_later,
            1,  // run quench on first (ie last) iteration
            n_inner_iterations_lotemp,
            nrotamers,
            true,
            false);
      }
      if (g.thread_rank() == 0) {
        scores_fullquench[0][warp_id] = after_full_quench_totalE;
      }
    };

    mgpu::standard_context_t context;

    mgpu::transform<128, 1>(
        hitemp_simulated_annealing, n_hitemp_simA_threads, context);
    mgpu::mergesort(
        scores_hitemp.data(),
        sorted_hitemp_traj.data(),
        n_hitemp_simA_traj,
        mgpu::less_t<float>(),
        context);

    mgpu::transform<128, 1>(
        lotemp_simulated_annealing, n_lotemp_simA_threads, context);
    mgpu::mergesort(
        scores_lotemp.data(),
        sorted_lotemp_traj.data(),
        n_lotemp_simA_traj,
        mgpu::less_t<float>(),
        context);

    mgpu::transform<128, 1>(fullquench, n_fullquench_threads, context);

    return {scores_fullquench_t, rotamer_assignments_fullquench_t};
  }
};

template <tmol::Device D>
struct AnnealerDispatch {
  static auto forward(
      TView<int, 1, D> nrotamers_for_res,
      TView<int, 1, D> oneb_offsets,
      TView<int, 1, D> res_for_rot,
      TView<int, 2, D> respair_nenergies,
      TView<int, 1, D> chunk_size,
      TView<int, 2, D> chunk_offset_offsets,
      TView<int64_t, 2, D> twob_offsets,
      TView<int, 1, D> fine_chunk_offsets,
      TView<float, 1, D> energy1b,
      TView<float, 1, D> energy2b,
      int seed) -> std::tuple<TPack<float, 2, D>, TPack<int, 2, D> > {
    clock_t start = clock();

    InteractionGraph<D, int, float> ig(
        {nrotamers_for_res,
         oneb_offsets,
         res_for_rot,
         respair_nenergies,
         chunk_size,
         chunk_offset_offsets,
         twob_offsets,
         fine_chunk_offsets,
         energy1b,
         energy2b});

    auto result =
        Annealer<D, InteractionGraph<D, int, float> >::run_simulated_annealing(
            ig, seed);

    cudaDeviceSynchronize();
    clock_t stop = clock();
    std::cout << "GPU simulated annealing in "
              << ((double)stop - start) / CLOCKS_PER_SEC << " seconds"
              << std::endl;

    return result;
  }
};

template struct AnnealerDispatch<tmol::Device::CUDA>;

}  // namespace compiled
}  // namespace pack
}  // namespace tmol
