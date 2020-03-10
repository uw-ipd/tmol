#include <c10/DeviceType.h>
#include <ATen/Context.h>
#include <ATen/CUDAGenerator.h>
#include <THC/THCGenerator.hpp>
#include <THC/THCTensorRandom.h>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>

// ??? #include "annealer.hh"
#include "simulated_annealing.hh"
#include <tmol/pack/compiled/params.hh>

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
THCGenerator* THCRandom_getGenerator(THCState* state);

// Stolen from torch, v1.0.0;
// unnecessary in the latest release, where this function
// is built in to CUDAGenerator.
// Modified slightly as the input Generator is unused.
// increment should be at least the number of curand() random numbers used in
// each thread.
std::pair<uint64_t, uint64_t> next_philox_seed(uint64_t increment) {
  auto gen_ = THCRandom_getGenerator(at::globalContext().getTHCState());
  uint64_t offset = gen_->state.philox_seed_offset.fetch_add(increment);
  return std::make_pair(gen_->state.initial_seed, offset);
}


namespace tmol {
namespace pack {
namespace compiled {

template<tmol::Device D, typename Int, typename Real>
struct InteractionGraph {
public:
  TView<Int, 1, D> nrotamers_for_res;
  TView<Int, 1, D> oneb_offsets;
  TView<Int, 1, D> res_for_rot;
  TView<Int, 2, D> nenergies;
  TView<int64_t, 2, D> twob_offsets;
  TView<Real, 2, D> energy1b;
  TView<Real, 1, D> energy2b;
};


/// @brief Return a uniformly-distributed integer in the range
/// between 0 and n-1.
/// Note that curand_uniform() returns a random number in the range
/// (0,1], unlike unlike rand() returns a random number in the range
/// [0,1). Take care with curand_uniform().
__device__
inline
int
curand_in_range(
  curandStatePhilox4_32_10_t * state,
  int n
)
{
  return int(curand_uniform(state)*n) % n;
}

template <unsigned int nthreads, typename T, typename op_t>
__device__
__inline__
T
reduce_shfl_and_broadcast(
  cooperative_groups::thread_block_tile<nthreads> g,
  T val,
  op_t op
)
{
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

  // printf("%d %d shfl orig %f reduce %f bcast %f vs %f\n", g.thread_rank(), threadIdx.x, float(val_orig), float(val), float(val_bcast), float(hand_rolled_val));

  return val_bcast;
}

template <unsigned int nthreads, typename T, typename F>
__device__
__inline__
T
exclusive_scan_shfl(
  cooperative_groups::thread_block_tile<nthreads> g,
  T val,
  F f
)
{
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
__device__
__inline__
T
inclusive_scan_shfl(
  cooperative_groups::thread_block_tile<nthreads> g,
  T val,
  F f
)
{
  for (unsigned int i = 1; i <= nthreads; i *= 2) {
    T const shfl_val = g.shfl_up(val, i);
    if (g.thread_rank() >= i) {
      val = f(shfl_val, val);
    }
  }
  return val;
}


template<tmol::Device D>
inline
__device__
void
set_quench_order(
  TensorAccessor<int, 1, D> quench_order,
  curandStatePhilox4_32_10_t * state
){
  // Create a random permutation of all the rotamers
  // and visit them in this order to ensure all of them
  // are seen during the quench step
  int const nrots = quench_order.size(0);
  for (int i = 0; i < nrots; ++i) {
    quench_order[i] = i;
  }
  for (int i = 0; i <= nrots-2; ++i) {
    int rand_offset = curand_in_range(state, nrots-i);
    int j = i + rand_offset;
    // swap i and j;
    int jval = quench_order[j];
    quench_order[j] = quench_order[i];
    quench_order[i] = jval;
  }
}

template<tmol::Device D>
inline
__device__
int
set_quench_32_order(
  TView<int, 1, D> nrotamers_for_res,
  TView<int, 1, D> oneb_offsets,
  TensorAccessor<int, 1, D> quench_order,
  curandStatePhilox4_32_10_t * state
){
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
      quench_order[count_n_quench_rots + j] = i_offset + 31*j;
    }
    count_n_quench_rots += i_nquench_rots;
  }
  for (int i = 0; i <= count_n_quench_rots-2; ++i) {
    int rand_offset = curand_in_range(state, count_n_quench_rots-i);
    int j = i + rand_offset;
    // swap i and j;
    int jval = quench_order[j];
    quench_order[j] = quench_order[i];
    quench_order[i] = jval;
  }
  return count_n_quench_rots;
}



template<
  unsigned int nthreads,
  tmol::Device D,
  typename Real,
  typename Int
>
inline
#ifdef __CUDACC__
__device__
#endif
Real
total_energy_for_assignment_parallel(
  cooperative_groups::thread_block_tile<nthreads> g,
  InteractionGraph<D, Int, Real> ig,
  TensorAccessor<Int, 1, D> rotamer_assignment,
  int background_id,
  bool verbose = false
)
{
  Real totalE = 0;
  int const nres = ig.nrotamers_for_res.size(0);
  for (int i = g.thread_rank(); i < nres; i += nthreads) {
    int const irot_local = rotamer_assignment[i];
    int const irot_global = irot_local + ig.oneb_offsets[i];

    totalE += ig.energy1b[background_id][irot_global];
  }

  for (int i = g.thread_rank(); i < nres; i += nthreads) {
    int const irot_local = rotamer_assignment[i];

    for (int j = i+1; j < nres; ++j) {
      int const jrot_local = rotamer_assignment[j];
      if (ig.nenergies[i][j] == 0) {
        continue;
      }
      float ij_energy = ig.energy2b[
        ig.twob_offsets[i][j]
        + ig.nrotamers_for_res[j] * irot_local
        + jrot_local
      ];
      totalE += ij_energy;
    }
  }
  // if (verbose && (blockDim.x * blockIdx.x + threadIdx.x)/32 == 374) {
  //   printf("warp 374 subtotal: %d %f\n", threadIdx.x % 32, totalE);
  // }
  totalE = reduce_shfl_and_broadcast(g, totalE, mgpu::plus_t<float>());
  // if (verbose && (blockDim.x * blockIdx.x + threadIdx.x)/32 == 374) {
  //   printf("warp 374 total: %d %f\n", threadIdx.x % 32, totalE);
  // }
  return totalE;
}

template<
  tmol::Device D,
  uint nthreads,
  typename Int,
  typename Real
>
MGPU_DEVICE
float
warp_wide_sim_annealing(
  curandStatePhilox4_32_10_t * state,
  cooperative_groups::thread_block_tile<nthreads> g,
  InteractionGraph<D, Int, Real> ig,
  int warp_id,
  int background_id,
  TensorAccessor<Int, 1, D> rotamer_assignments,
  TensorAccessor<Int, 1, D> best_rotamer_assignments,
  TensorAccessor<Int, 1, D> quench_order,
  float hi_temp,
  float lo_temp,
  int n_outer_iterations,
  int n_inner_iterations,
  int n_quench_iterations,
  bool quench_on_last_iteration,
  bool quench_lite
)
{
  int const nres = ig.nrotamers_for_res.size(0);
  int const nrotamers = ig.res_for_rot.size(0);

  float temperature = hi_temp;
  float best_energy = total_energy_for_assignment_parallel(
    g, ig, rotamer_assignments, background_id);
  float current_total_energy = best_energy;
  int ntrials = 0;
  for (int i = 0; i < n_outer_iterations; ++i) {

    // if (g.thread_rank() == 0) {
    //   printf("top of outer loop %d currentE %f bestE %f temp %f\n", i, current_total_energy, best_energy, temperature);
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
      current_total_energy = total_energy_for_assignment_parallel(
	g, ig, rotamer_assignments, background_id);
    }

    for (int j = 0; j < i_n_inner_iterations; ++j) {
      int ran_rot(0);
      float accept_prob(0);
      if (quench) {
        if (g.thread_rank() == 0) {
          if (j % quench_period == 0) {
            if (quench_lite) {
              quench_period = set_quench_32_order(
                ig.nrotamers_for_res, ig.oneb_offsets,
                quench_order, state);
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

      int const ran_res = ig.res_for_rot[ran_rot];
      int const local_prev_rot = rotamer_assignments[ran_res];
      int const ran_res_nrots = ig.nrotamers_for_res[ran_res];
      int const ran_res_rotamer_offset = ig.oneb_offsets[ran_res];

      bool prev_rot_in_range = false;
      int thread_w_prev_rot = 0;
      { // scope
        int const local_ran_rot_orig = ran_rot - ran_res_rotamer_offset;
        int const local_prev_rot_wrapped = local_ran_rot_orig <= local_prev_rot ?
          local_prev_rot :
          local_prev_rot + ran_res_nrots;
        prev_rot_in_range = local_ran_rot_orig + 32 > local_prev_rot_wrapped;
        thread_w_prev_rot = prev_rot_in_range ?
	  ( local_ran_rot_orig == local_prev_rot ?
	    0 : local_prev_rot_wrapped - local_ran_rot_orig)
	  : 0;
      }


      int const local_ran_rot = prev_rot_in_range ? (
        (ran_rot - ran_res_rotamer_offset + g.thread_rank()) % ran_res_nrots) :
        (g.thread_rank() == 0 ?
          local_prev_rot :
          (ran_rot - ran_res_rotamer_offset  + g.thread_rank() - 1) % ran_res_nrots);
      ran_rot = local_ran_rot + ran_res_rotamer_offset;

      // if (local_ran_rot < 0 || local_ran_rot >= ran_res_nrots) {
      // 	printf("!!! local_ran_rot out of range; ran_res_nrots %d local_ran_rot %d\n", ran_res_nrots, local_ran_rot);
      // 	this_thread_active = false;
      // }

      // If there are fewer rotamers on this residue than there are threads
      // active in the warp, do not wrap and consider a rotamer more than once
      bool const this_thread_active = ran_res_nrots > g.thread_rank();
      bool const this_thread_last_active = ran_res_nrots == g.thread_rank() || g.thread_rank() == 32 - 1;

      float new_e = 9999;
      if (this_thread_active) {
        new_e = ig.energy1b[background_id][ran_rot];
      }

      // Temp: iterate across all residues instead of just the
      // neighbors of ran_rot_res
      if (this_thread_active) {
        for (int k=0; k < nres; ++k) {
          if (k == ran_res || ig.nenergies[ran_res][k] == 0) {
            // alt_energies[k][warp_id] = 0;
            continue;
          }
          int const local_k_rot = rotamer_assignments[k];

          int64_t const k_ran_offset = ig.twob_offsets[k][ran_res];
          new_e += ig.energy2b[k_ran_offset + ran_res_nrots * local_k_rot + local_ran_rot];
        }
      }

      // if (g.thread_rank() == 0) {
      //   printf("minimum<float>\n");
      // }
      float const min_e = reduce_shfl_and_broadcast(g, new_e, mgpu::minimum_t<float>());
      // printf("thread %d min_e %f\n", thread_id, min_e);
      float myexp = expf( -1 * ( new_e - min_e ) / temperature );
      // printf("thread %d myexp %f\n", thread_id, myexp);
      // if (g.thread_rank() == 0) {
      //   printf("plus<float>\n");
      // }
      float const partition = reduce_shfl_and_broadcast(g, myexp, mgpu::plus_t<float>());
      // printf("thread %d partition %f\n", thread_id, partition);
      float const myprob = this_thread_active ? myexp / partition : 0;
      // printf("thread %d myprob %f\n", thread_id, myprob);
      // if (g.thread_rank() == 0) {
      //   printf("inclusive scan plus<float>\n");
      // }
      float scan_prob = inclusive_scan_shfl(g, myprob, mgpu::plus_t<float>());
      // printf("thread %d prev rotamer %d new rotamer %d new_e %f active? %d temp %f\n", thread_id, local_prev_rot, local_ran_rot, new_e, this_thread_active, temperature);
       // printf("thread %d myexp %f part %f myprob %f scan_prob %f accept_prob %f\n", thread_id, myexp, partition, myprob, scan_prob, accept_prob);
      if ( this_thread_last_active ) {
        // due to numerical imprecision, it's entirely likely that the scan probability
        // for the last active thread to be slightly more or slightly less than 1,
        // and we want to ensure that there's a winner for each thread.
        scan_prob = 1;
      }
      int accept_rank = ( this_thread_active && accept_prob <= scan_prob);
      // printf("thread %d accept_rank %d\n", thread_id, accept_rank);
      // if (g.thread_rank() == 0) {
      //   printf("inclusive scan plus<int>\n");
      // }
      accept_rank = inclusive_scan_shfl(g, accept_rank, mgpu::plus_t<int>());
      // printf("thread %d accept_rank after scan %d\n", thread_id, accept_rank);

      bool accept = accept_rank == 1 && this_thread_active;
      // printf("thread %d accept %d\n", thread_id, accept);
      // if (g.thread_rank() == 0) {
      //   printf("max<int>\n");
      // }
      int const accept_thread = reduce_shfl_and_broadcast(g, accept ? int(g.thread_rank()) : int(-1), mgpu::maximum_t<int>());
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
        for (int k=g.thread_rank(); k < nres; k += 32) {
          best_rotamer_assignments[k] = rotamer_assignments[k];
        }
        best_energy = current_total_energy; // g.shfl(best_energy, accept_thread);
      }

      ++ntrials;
      if (ntrials > 1000) {
        ntrials = 0;
        current_total_energy = total_energy_for_assignment_parallel(
	  g, ig, rotamer_assignments, background_id);
        // if (g.thread_rank() == 0) {
        //   printf("refresh total energy currentE %f\n", current_total_energy);
        // }
      }

    } // end inner loop

    // geometric cooling toward 0.3
    // std::cout << "temperature " << temperature << " energy " <<
    //  total_energy_for_assignment(nrotamers_for_res, oneb_offsets,
    //    res_for_rot, nenergies, twob_offsets, energy1b, energy2b, my_rotamer_assignment) << std::endl;
    temperature = 0.35 * (temperature - lo_temp) + lo_temp;

  } // end outer loop


  float totalE = total_energy_for_assignment_parallel(
    g, ig, rotamer_assignments, background_id, true);

  // float totalE2 = total_energy_for_assignment_parallel(
  //   g, ig, best_rotamer_assignments, background_id, true);

  return totalE;
}


template <
  tmol::Device D,
  uint nthreads,
  class Int,
  class Real
>
__device__
__inline__
float
spbr(
  curandStatePhilox4_32_10_t * state,
  cooperative_groups::thread_block_tile<nthreads> g,
  InteractionGraph<D, Int, Real> ig,
  int warp_id,
  int n_spbr,
  TView<int, 2, D> spbr_rotamer_assignments,
  TView<int, 2, D> spbr_perturbed_assignments
)
{
  int const nres = ig.nrotamers_for_res.size(0);
  int const nrotamers = ig.res_for_rot.size(0);
  
  float energy = total_energy_for_assignment_parallel(
    g, ig, spbr_rotamer_assignments[warp_id], 0 /*temp!*/);

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
      int irot = i == ran_res ? ran_rot_local :
        spbr_rotamer_assignments[warp_id][i];
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
        float jE = ig.energy1b[0][j_global];
        for (int k = 0; k < nres; ++k) {
          if (k == i || ig.nenergies[k][i] == 0) continue;

          int const k_rotamer = k == ran_res ?
            ran_rot_local : spbr_rotamer_assignments[warp_id][k];
          jE += ig.energy2b[
            ig.twob_offsets[k][i] +
            i_nrots * k_rotamer + j];
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
      float best_rot_E = reduce_shfl_and_broadcast(g, my_best_rot_E, mgpu::minimum_t<float>());
      int mine_is_best = best_rot_E == my_best_rot_E;
      int scan_val = inclusive_scan_shfl(g, mine_is_best, mgpu::plus_t<int>());
      if (mine_is_best && scan_val == 1) {
        // exactly one thread saves the assigned rotamer to the
        // spbr_perturbed_assignemnt array
        spbr_perturbed_assignments[warp_id][i] = my_best_rot;
      }
    }

    // 5. compute the new total energy after relaxation
    float alt_energy = total_energy_for_assignment_parallel(
      g, ig, spbr_perturbed_assignments[warp_id], 0 /*temp*/);

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

template <tmol::Device D>
struct OneStageAnnealerDispatch
{
  static
  auto
  forward(
    TView<SimAParams<float>, 1, tmol::Device::CPU> simA_params,
    TView<int, 1, D> nrotamers_for_res,
    TView<int, 1, D> oneb_offsets,
    TView<int, 1, D> res_for_rot,
    TView<int, 2, D> nenergies,
    TView<int64_t, 2, D> twob_offsets,
    TView<float, 2, D> energy1b,
    TView<float, 1, D> energy2b
  )
    -> std::tuple<
      TPack<float, 2, D>, // energies of assignments
      TPack<int, 2, D>, // assignments
      TPack<int, 1, D> // backgrounds indices for assignments
      >
  {
    clock_t start = clock();
    InteractionGraph<D, int, float> ig({
	nrotamers_for_res,
        oneb_offsets,
        res_for_rot,
        nenergies,
        twob_offsets,
        energy1b,
        energy2b});
    
    int const nres = nrotamers_for_res.size(0);
    int const nrotamers = res_for_rot.size(0);
    int const n_backgrounds = energy1b.size(0);

    float const hi_temp = simA_params[0].hitemp;
    float const lo_temp = simA_params[0].lotemp;
    int const n_outer = (int) simA_params[0].n_outer;
    int const n_inner = (int) (simA_params[0].n_inner_scale * nrotamers);

    int const n_traj = 1200;
    int const n_threads = n_traj * 32;
    
    auto scores_t = TPack<float, 2, D>::zeros({1,n_traj});
    auto rotamer_assignments_t = TPack<int, 2, D>::zeros({n_traj, nres});
    auto best_rotamer_assignments_t = TPack<int, 2, D>::zeros({n_traj, nres});
    auto rotamer_assignments_final_t = TPack<int, 2, D>::zeros({n_traj, nres});
    auto quench_order_t = TPack<int, 2, D>::zeros({n_traj, nrotamers});
    auto sorted_traj_t = TPack<int, 1, D>::zeros({n_traj});
    auto background_inds_t = TPack<int, 1, D>::zeros(n_traj);
    auto final_background_inds_t = TPack<int, 1, D>::zeros(n_traj);

    auto scores = scores_t.view;
    auto rotamer_assignments = rotamer_assignments_t.view;
    auto best_rotamer_assignments = best_rotamer_assignments_t.view;
    auto rotamer_assignments_final = rotamer_assignments_final_t.view;
    auto quench_order = quench_order_t.view;
    auto sorted_traj = sorted_traj_t.view;
    auto background_inds = background_inds_t.view;
    auto final_background_inds = final_background_inds_t.view;
    
    auto philox_seed = next_philox_seed(
      nrotamers +  // initial random rotamer assignment
      n_outer * n_inner + // n rotamer substitutions
      nrotamers // random permutation of rotamers
    );

    bool run_quench = simA_params[0].quench != 0;
    
    auto simulated_annealing = [=] MGPU_DEVICE (int thread_id){
      curandStatePhilox4_32_10_t state;
      curand_init(
        philox_seed.first,
        thread_id,
        philox_seed.second,
        &state);

      cooperative_groups::thread_block_tile<32> g = cooperative_groups::tiled_partition<32>(
        cooperative_groups::this_thread_block());
      int const warp_id = thread_id / 32;
      int const background_ind = warp_id % n_backgrounds;

      if (g.thread_rank() == 0) {
        sorted_traj[warp_id] = warp_id;
	background_inds[warp_id] = background_ind;
      }

      for (int i = g.thread_rank(); i < nres; i += 32) {
        int const i_nrots = nrotamers_for_res[i];
        int chosen = int(curand_uniform(&state) * i_nrots) % i_nrots;
        rotamer_assignments[warp_id][i] = chosen;
        best_rotamer_assignments[warp_id][i] = chosen;
      }

      float rotstate_energy_after_simA = warp_wide_sim_annealing(
        &state,
        g,
	ig,
        warp_id,
	background_ind,
        rotamer_assignments[warp_id],
        best_rotamer_assignments[warp_id],
        quench_order[warp_id],
        hi_temp,
        lo_temp,
        n_outer,
        n_inner,
        nrotamers,
        run_quench,
        false
      );

      if (g.thread_rank() == 0) {
        scores[0][warp_id] = rotstate_energy_after_simA;
      }
    };

    auto take_best = [=] MGPU_DEVICE (int thread_id) {
      cooperative_groups::thread_block_tile<32> g = cooperative_groups::tiled_partition<32>(
        cooperative_groups::this_thread_block());
      int const warp_id = thread_id / 32;
      int const source_traj = sorted_traj[warp_id];
      for (int i = g.thread_rank(); i < nres; i += 32) {
	rotamer_assignments_final[warp_id][i] =
	  best_rotamer_assignments[source_traj][i];
      }
      if (g.thread_rank() == 0) {
	final_background_inds[warp_id] = background_inds[source_traj];
      }
      // if (g.thread_rank() == 0 && warp_id < 10 ) {
      // 	printf("best energy %d %f from traj %d\n", warp_id, scores[0][warp_id], source_traj);
      // }
    };

    mgpu::standard_context_t context;
  
    mgpu::transform<128, 1>(simulated_annealing, n_threads, context);
    mgpu::mergesort(
      scores.data(), sorted_traj.data(),
      n_traj, mgpu::less_t<float>(), context);
    mgpu::transform<32, 1>(take_best, n_threads, context);
    
    cudaDeviceSynchronize();
    clock_t stop = clock();
    std::cout << "GPU simulated annealing in " <<
       ((double) stop - start)/CLOCKS_PER_SEC << " seconds" << std::endl;

    return {scores_t, rotamer_assignments_final_t, final_background_inds_t};
  }

};



template <tmol::Device D>
struct MultiStageAnnealerDispatch
{
  static
  auto
  forward(
    TView<SimAParams<float>, 1, tmol::Device::CPU> simA_params,
    TView<int, 1, D> nrotamers_for_res,
    TView<int, 1, D> oneb_offsets,
    TView<int, 1, D> res_for_rot,
    TView<int, 2, D> nenergies,
    TView<int64_t, 2, D> twob_offsets,
    TView<float, 2, D> energy1b,
    TView<float, 1, D> energy2b
  )
    -> std::tuple<
      TPack<float, 2, D>, // energies of assignments
      TPack<int, 2, D>, // assignments
      TPack<int, 1, D> // backgrounds indices for assignments
      >
  {
    clock_t start = clock();

    InteractionGraph<D, int, float> ig({
	nrotamers_for_res,
        oneb_offsets,
        res_for_rot,
        nenergies,
        twob_offsets,
        energy1b,
        energy2b});
    
    int const nres = nrotamers_for_res.size(0);
    int const nrotamers = res_for_rot.size(0);
    int const n_backgrounds = energy1b.size(0);

    int const n_hitemp_simA_traj = 2000;
    int const n_hitemp_simA_threads = 32 * n_hitemp_simA_traj;
    float const round1_cut = 0.25;
    int const n_lotemp_expansions = 10;
    int const n_lotemp_simA_traj = int(n_hitemp_simA_traj * n_lotemp_expansions * round1_cut);
    int const n_lotemp_simA_threads = 32 * n_lotemp_simA_traj;
    float const round2_cut = 0.25;
    int const n_fullquench_traj = int(n_lotemp_simA_traj * round2_cut);
    int const n_fullquench_threads = 32 * n_fullquench_traj;
    int const n_outer_iterations_hitemp = 10;
    int const n_inner_iterations_hitemp = nrotamers / 8;
    int const n_outer_iterations_lotemp = 10;
    int const n_inner_iterations_lotemp = nrotamers / 16;
    float const high_temp_initial = simA_params[0].hitemp;
    float const low_temp_initial = simA_params[0].lotemp;
    float const high_temp_later = 0.2;
    float const low_temp_later = 0.1;

    int const max_traj = std::max(std::max(n_hitemp_simA_traj, n_lotemp_simA_traj), n_fullquench_traj);

    auto scores_hitemp_t = TPack<float, 1, D>::zeros(n_hitemp_simA_traj);
    auto rotamer_assignments_hitemp_t = TPack<int, 2, D>::zeros({n_hitemp_simA_traj, nres});
    auto best_rotamer_assignments_hitemp_t = TPack<int, 2, D>::zeros({n_hitemp_simA_traj, nres});
    auto rotamer_assignments_hitemp_quenchlite_t = TPack<int, 2, D>::zeros({n_hitemp_simA_traj, nres});
    auto sorted_hitemp_traj_t = TPack<int, 1, D>::zeros(n_hitemp_simA_traj);
    // auto hitemp_background_ind_t = TPack<int, 1, D>::zeros(n_hitemp_simA_traj);

    auto scores_lotemp_t = TPack<float, 1, D>::zeros(n_lotemp_simA_traj);
    auto rotamer_assignments_lotemp_t = TPack<int, 2, D>::zeros({n_lotemp_simA_traj, nres});
    auto best_rotamer_assignments_lotemp_t = TPack<int, 2, D>::zeros({n_lotemp_simA_traj, nres});
    //auto rotamer_assignments_lotemp_quenchlite_t = TPack<int, 2, D>::zeros({n_lotemp_simA_traj, nres});
    auto sorted_lotemp_traj_t = TPack<int, 1, D>::zeros(n_lotemp_simA_traj);
    auto lotemp_background_ind_t = TPack<int, 1, D>::zeros(n_lotemp_simA_traj);

    auto scores_fullquench_t = TPack<float, 2, D>::zeros({1, n_fullquench_traj});
    auto rotamer_assignments_fullquench_t = TPack<int, 2, D>::zeros({n_fullquench_traj, nres});
    auto best_rotamer_assignments_fullquench_t = TPack<int, 2, D>::zeros({n_fullquench_traj, nres});
    auto sorted_fullquench_traj_t = TPack<int, 1, D>::zeros({n_fullquench_traj});
    auto fullquench_background_ind_t = TPack<int, 1, D>::zeros(n_fullquench_traj);

    auto quench_order_t = TPack<int, 2, D>::zeros({max_traj, nrotamers});
    auto final_background_ind_t = TPack<int, 1, D>::zeros(n_fullquench_traj);
    
    auto scores_hitemp = scores_hitemp_t.view;
    auto rotamer_assignments_hitemp = rotamer_assignments_hitemp_t.view;    
    auto best_rotamer_assignments_hitemp = best_rotamer_assignments_hitemp_t.view;
    auto rotamer_assignments_hitemp_quenchlite = rotamer_assignments_hitemp_quenchlite_t.view;
    auto sorted_hitemp_traj = sorted_hitemp_traj_t.view;
    // auto hitemp_background_ind = hitemp_background_ind_t.view;

    auto scores_lotemp = scores_lotemp_t.view;
    auto rotamer_assignments_lotemp = rotamer_assignments_lotemp_t.view;    
    auto best_rotamer_assignments_lotemp = best_rotamer_assignments_lotemp_t.view;
    //auto rotamer_assignments_lotemp_quenchlite = rotamer_assignments_lotemp_quenchlite_t.view;
    auto sorted_lotemp_traj = sorted_lotemp_traj_t.view;
    auto lotemp_background_ind = lotemp_background_ind_t.view;

    auto scores_fullquench = scores_fullquench_t.view;
    auto rotamer_assignments_fullquench = rotamer_assignments_fullquench_t.view;    
    auto best_rotamer_assignments_fullquench = best_rotamer_assignments_fullquench_t.view;
    //auto sorted_fullquench_traj = sorted_lotem_traj_t.view;
    auto sorted_fullquench_traj = sorted_fullquench_traj_t.view;
    auto fullquench_background_ind = fullquench_background_ind_t.view;

    auto quench_order = quench_order_t.view;
    auto final_background_ind = final_background_ind_t.view;

    // This code will work for future versions of the torch/aten libraries, but not
    // this one.
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
    auto philox_seed_hitemp = next_philox_seed(
      nrotamers +  // initial random rotamer assignment
      n_outer_iterations_hitemp * n_inner_iterations_hitemp + // hitemp annealing
      (nrotamers / 31 + nres) // hitemp random permutation of quenchlite rotamers
    );


    auto philox_seed_lotemp = next_philox_seed(
      n_outer_iterations_lotemp * n_outer_iterations_lotemp + // lotemp annealing
      (nrotamers / 31 + nres) // lowtemp random permuation of quenchlite rotamers
    );

    auto philox_seed_quench = next_philox_seed(nrotamers);
    
    auto hitemp_simulated_annealing = [=] MGPU_DEVICE (int thread_id){
      curandStatePhilox4_32_10_t state;
      curand_init(
        philox_seed_hitemp.first,
        thread_id,
        philox_seed_hitemp.second,
        &state);

      cooperative_groups::thread_block_tile<32> g = cooperative_groups::tiled_partition<32>(
        cooperative_groups::this_thread_block());
      int const warp_id = thread_id / 32;
      int const background_ind = warp_id % n_backgrounds;


      if (g.thread_rank() == 0) {
        sorted_hitemp_traj[warp_id] = warp_id;
      }

      for (int i = g.thread_rank(); i < nres; i += 32) {
        int const i_nrots = nrotamers_for_res[i];
        int chosen = int(curand_uniform(&state) * i_nrots) % i_nrots;
        rotamer_assignments_hitemp[warp_id][i] = chosen;
        best_rotamer_assignments_hitemp[warp_id][i] = chosen;
      }

      float rotstate_energy_after_high_temp = warp_wide_sim_annealing(
        &state,
        g,
	ig,
        warp_id,
	background_ind,
        rotamer_assignments_hitemp[warp_id],
        best_rotamer_assignments_hitemp[warp_id],
        quench_order[warp_id],
        high_temp_initial,
        low_temp_initial,
        n_outer_iterations_hitemp,
        n_inner_iterations_hitemp,
        nrotamers,
        false,
        false
      );

      // Save the state before moving into quench
      for (int i = g.thread_rank(); i < nres; i += 32) {
        int i_assignment = best_rotamer_assignments_hitemp[warp_id][i];
        rotamer_assignments_hitemp[warp_id][i] = i_assignment;
        rotamer_assignments_hitemp_quenchlite[warp_id][i] = i_assignment;
      }
      float best_energy_after_high_temp = total_energy_for_assignment_parallel(
	g, ig, best_rotamer_assignments_hitemp[warp_id], background_ind);

      // ok, run quench lite as a way to predict where this rotamer assignment will
      // end up after low-temperature annealing
      float after_first_quench_lite_totalE = warp_wide_sim_annealing(
        &state,
        g,
	ig,
        warp_id,
	background_ind,
        rotamer_assignments_hitemp_quenchlite[warp_id],
        best_rotamer_assignments_hitemp[warp_id],
        quench_order[warp_id],
        high_temp_initial,
        low_temp_initial,
        1, // perform quench in first (ie last) iteration
        n_inner_iterations_hitemp, // irrelevant
        nrotamers,
        true,
        true
      );
      if (g.thread_rank() == 0) {
        scores_hitemp[warp_id] = after_first_quench_lite_totalE;
      }
    };

    auto lotemp_simulated_annealing = [=] MGPU_DEVICE (int thread_id){
      curandStatePhilox4_32_10_t state;
      curand_init(
        philox_seed_lotemp.first,
        thread_id,
        philox_seed_lotemp.second,
        &state);

      cooperative_groups::thread_block_tile<32> g = cooperative_groups::tiled_partition<32>(
        cooperative_groups::this_thread_block());

      int const warp_id = thread_id / 32;
      int const source_traj = sorted_hitemp_traj[warp_id / n_lotemp_expansions];
      int const background_ind = source_traj % n_backgrounds;
      
      if (g.thread_rank() == 0) {
        sorted_lotemp_traj[warp_id] = warp_id;
	lotemp_background_ind[warp_id] = background_ind;
      }
      
      // initialize the rotamer assignment from one of the top trajectories
      // of the high-temperature annealing trajectory
      for (int i = g.thread_rank(); i < nres; i += 32) {
        int i_rot = rotamer_assignments_hitemp[source_traj][i];
        rotamer_assignments_lotemp[warp_id][i] = i_rot;
        best_rotamer_assignments_lotemp[warp_id][i] = i_rot;
      }

      // float lotemp_startE = total_energy_for_assignment_parallel(g,
      //   nrotamers_for_res, oneb_offsets, res_for_rot, nenergies, twob_offsets,
      //   energy1b, energy2b, rotamer_assignments_hitemp_quenchlite[source_traj]);
      // if (g.thread_rank() == 0) {
      //         printf("warp %d lotemp source_traj %d (%d) %f vs %f\n", warp_id, source_traj,
      //           n_hitemp_simA_traj, scores_hitemp[warp_id / n_lotemp_expansions], lotemp_startE);
      // }

      
      // Now run a low-temperature cooling trajectory
      float low_temp_totalE = warp_wide_sim_annealing(
        &state,
        g,
	ig,
        warp_id,
	background_ind,
        rotamer_assignments_lotemp[warp_id],
        best_rotamer_assignments_lotemp[warp_id],
        quench_order[warp_id],
        high_temp_later,
        low_temp_later,
        n_outer_iterations_lotemp,
        n_inner_iterations_lotemp,
        nrotamers,
        false,
        false
      ); 

      // float best_energy_after_low_temp = total_energy_for_assignment_parallel(g,
      //   nrotamers_for_res, oneb_offsets, res_for_rot, nenergies, twob_offsets,
      //   energy1b, energy2b, best_rotamer_assignments_lotemp[warp_id]);

      // now we'll run a quench-lite
      // ok, we will run quench lite on first state
      float after_lotemp_quench_lite_totalE = warp_wide_sim_annealing(
        &state,
        g,
	ig,
        warp_id,
	background_ind,
        rotamer_assignments_lotemp[warp_id],
        best_rotamer_assignments_lotemp[warp_id],
        quench_order[warp_id],
        high_temp_later,
        low_temp_later,
        1, // run quench on first (i.e. last) iteration
        n_inner_iterations_lotemp, // irrelevant
        nrotamers,
        true,
        true
      );
      if (g.thread_rank() == 0) {
        scores_lotemp[warp_id] = after_lotemp_quench_lite_totalE;
      }
    };

    auto fullquench = [=] MGPU_DEVICE (int thread_id){
      curandStatePhilox4_32_10_t state;
      curand_init(
        philox_seed_quench.first,
        thread_id,
        philox_seed_quench.second,
        &state);

      cooperative_groups::thread_block_tile<32> g = cooperative_groups::tiled_partition<32>(
        cooperative_groups::this_thread_block());
      int const warp_id = thread_id / 32;
      int const source_traj = sorted_lotemp_traj[warp_id];
      int const background_ind = lotemp_background_ind[source_traj];

      if (g.thread_rank() == 0) {
        sorted_fullquench_traj[warp_id] = warp_id;
	fullquench_background_ind[warp_id] = background_ind;
      }

      // if (g.thread_rank() == 0) {
      //         printf("warp %d fullquench source_traj %d (%d) %f\n", warp_id, source_traj,
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
	background_ind,
        rotamer_assignments_fullquench[warp_id],
        best_rotamer_assignments_fullquench[warp_id],
        quench_order[warp_id],
        high_temp_later,
        low_temp_later,
        1, // run quench on first (ie last) iteration
        n_inner_iterations_lotemp,
        nrotamers,
        true,
        false
      );
      }
      if (g.thread_rank() == 0) {
        scores_fullquench[0][warp_id] = after_full_quench_totalE;
      }
    };

    
    auto take_best = [=] MGPU_DEVICE (int thread_id) {
      cooperative_groups::thread_block_tile<32> g = cooperative_groups::tiled_partition<32>(
        cooperative_groups::this_thread_block());
      int const warp_id = thread_id / 32;
      int const source_traj = sorted_fullquench_traj[warp_id];
      for (int i = g.thread_rank(); i < nres; i += 32) {
	rotamer_assignments_fullquench[warp_id][i] =
	  best_rotamer_assignments_fullquench[source_traj][i];
      }
      if (g.thread_rank() == 0) {
	final_background_ind[warp_id] = fullquench_background_ind[source_traj];
      }
    };

    mgpu::standard_context_t context;
  
    mgpu::transform<128, 1>(hitemp_simulated_annealing, n_hitemp_simA_threads, context);
    mgpu::mergesort(
      scores_hitemp.data(), sorted_hitemp_traj.data(),
      n_hitemp_simA_traj, mgpu::less_t<float>(), context);
  
    mgpu::transform<128, 1>(lotemp_simulated_annealing, n_lotemp_simA_threads, context);
    mgpu::mergesort(
      scores_lotemp.data(), sorted_lotemp_traj.data(),
      n_lotemp_simA_traj, mgpu::less_t<float>(), context);

    mgpu::transform<128, 1>(fullquench, n_fullquench_threads, context);
    mgpu::mergesort(
      scores_fullquench.data(), sorted_fullquench_traj.data(),
      n_fullquench_traj, mgpu::less_t<float>(), context);

    mgpu::transform<32, 1>(take_best, n_fullquench_threads, context);

    cudaDeviceSynchronize();
    clock_t stop = clock();
    std::cout << "GPU simulated annealing in " <<
       ((double) stop - start)/CLOCKS_PER_SEC << " seconds" << std::endl;

    return {scores_fullquench_t, rotamer_assignments_fullquench_t, final_background_ind_t};
  }

};

template struct OneStageAnnealerDispatch<tmol::Device::CUDA>;
template struct MultiStageAnnealerDispatch<tmol::Device::CUDA>;

} // namespace compiled
} // namespace pack
} // namespace tmol
