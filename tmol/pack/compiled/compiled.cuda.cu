#include <c10/DeviceType.h>
#include <ATen/Context.h>
#include <ATen/CUDAGenerator.h>
#include <THC/THCGenerator.hpp>
#include <THC/THCTensorRandom.h>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>

// ??? #include "annealer.hh"
#include "simulated_annealing.hh"

#include <moderngpu/kernel_compact.hxx>
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

template <unsigned int nthreads, typename T, typename F>
__device__
__inline__
T
reduce_shfl(
  cooperative_groups::thread_block_tile<nthreads> g,
  T val,
  F f
)
{
  for (unsigned int i = nthreads / 2; i > 0; i /= 2) {
    T const shfl_val = g.shfl_down(val, i);
    val = f(val, shfl_val);
  }
  // thread 0 shares its sum with everyone
  // so that there is no disagreement on the
  // partition function value
  val = g.shfl(val, 0);
  return val;
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
  TView<int, 2, D> quench_order,
  int dim1_ind,
  curandStatePhilox4_32_10_t * state
){
  // Create a random permutation of all the rotamers
  // and visit them in this order to ensure all of them
  // are seen during the quench step
  int const nrots = quench_order.size(0);
  for (int i = 0; i < nrots; ++i) {
    quench_order[i][dim1_ind] = i;
  }
  for (int i = 0; i <= nrots-2; ++i) {
    int rand_offset = curand_in_range(state, nrots-i);
    int j = i + rand_offset;
    // swap i and j;
    int jval = quench_order[j][dim1_ind];
    quench_order[j][dim1_ind] = quench_order[i][dim1_ind];
    quench_order[i][dim1_ind] = jval;
  }
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
  TView<Int, 1, D> nrotamers_for_res,
  TView<Int, 1, D> oneb_offsets,
  TView<Int, 1, D> res_for_rot,
  TView<Int, 2, D> nenergies,
  TView<int64_t, 2, D> twob_offsets,
  TView<Real, 1, D> energy1b,
  TView<Real, 1, D> energy2b,
  TensorAccessor<Int, 1, D> rotamer_assignment
)
{
  Real totalE = 0;
  int const nres = nrotamers_for_res.size(0);
  for (int i = g.thread_rank(); i < nres; i += nthreads) {
    int const irot_local = rotamer_assignment[i];
    int const irot_global = irot_local + oneb_offsets[i];
    
    totalE += energy1b[irot_global];
  }

  for (int i = g.thread_rank(); i < nres; i += nthreads) {
    int const irot_local = rotamer_assignment[i];

    for (int j = i+1; j < nres; ++j) {
      int const jrot_local = rotamer_assignment[j];
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
  totalE = reduce_shfl(g, totalE, mgpu::plus_t<float>());
  return totalE;
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
    TView<int64_t, 2, D> twob_offsets,
    TView<float, 1, D> energy1b,
    TView<float, 1, D> energy2b
  )
    -> std::tuple<
      TPack<float, 1, D>,
      TPack<int, 2, D> >
  {
    clock_t start = clock();

    int const nres = nrotamers_for_res.size(0);
    int const nrotamers = res_for_rot.size(0);

    int const n_blocks = 1200;
    int const n_simA_threads = 32 * n_blocks;
    int const n_outer_iterations = 10;
    int const n_inner_iterations = nrotamers / 4;
    float const high_temp = 100;
    float const low_temp = 0.3;

    auto scores_t = TPack<float, 1, D>::zeros({n_blocks});
    auto rotamer_assignments_t = TPack<int, 2, D>::zeros({n_blocks, nres});
    auto best_rotamer_assignments_t = TPack<int, 2, D>::zeros({n_blocks, nres});

    // auto curr_pair_energies_t = TPack<float, 3, D>::zeros({nres, nres, n_simA_threads});
    // auto alt_energies_t = TPack<float, 2, D>::zeros({nres, n_simA_threads});

    auto quench_order_t = TPack<int, 2, D>::zeros({nrotamers, n_blocks});

    auto scores = scores_t.view;
    auto rotamer_assignments = rotamer_assignments_t.view;
    auto best_rotamer_assignments = rotamer_assignments_t.view;
    auto quench_order = quench_order_t.view;

    // auto curr_pair_energies = curr_pair_energies_t.view;
    // auto alt_energies = alt_energies_t.view;

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
    auto philox_seed = next_philox_seed( nrotamers * 400 + nres);

    auto run_simulated_annealing = [=] MGPU_DEVICE (int thread_id){
      curandStatePhilox4_32_10_t state;
      curand_init(
        philox_seed.first,
        thread_id,
        philox_seed.second,
        &state);

      cooperative_groups::thread_block_tile<32> g = cooperative_groups::tiled_partition<32>(
        cooperative_groups::this_thread_block());
      int const warp_id = thread_id / 32;

      for (int i = g.thread_rank(); i < nres; i += 32) {
        int const i_nrots = nrotamers_for_res[i];
        int chosen = int(curand_uniform(&state) * i_nrots) % i_nrots;
        rotamer_assignments[warp_id][i] = chosen;
        best_rotamer_assignments[warp_id][i] = chosen;
      }

      float temperature = high_temp;
      float best_energy = total_energy_for_assignment_parallel(g,
	nrotamers_for_res, oneb_offsets, res_for_rot, nenergies, twob_offsets,
	energy1b, energy2b, rotamer_assignments[warp_id]
      );
      float current_total_energy = best_energy;
      int ntrials = 0;
      for (int i = 0; i < n_outer_iterations; ++i) {

	// if (g.thread_rank() == 0) {
	//   printf("top of outer loop %d currentE %f bestE %f temp %f\n", i, current_total_energy, best_energy, temperature);
	// }
        bool quench = false;
	int i_n_inner_iterations = n_inner_iterations;
        if (i == n_outer_iterations - 1) {
	  i_n_inner_iterations = nrotamers;
          quench = true;
          temperature = 1e-20;
          for (int j = g.thread_rank(); j < nres; j += 32) {
            rotamer_assignments[warp_id][j] = best_rotamer_assignments[warp_id][j];
          }
          current_total_energy = total_energy_for_assignment_parallel(g,
	    nrotamers_for_res, oneb_offsets, res_for_rot, nenergies, twob_offsets,
	    energy1b, energy2b, rotamer_assignments[warp_id]
          );
        }

        for (int j = 0; j < i_n_inner_iterations; ++j) {
          int ran_rot;
          float accept_prob(0);
          if (quench) {
	    if (g.thread_rank() == 0) {
              if (j % nrotamers == 0) {
                set_quench_order(quench_order, warp_id, &state);
              }
              ran_rot = quench_order[j%nrotamers][warp_id];
	    }
	    ran_rot = g.shfl(ran_rot, 0);
	    accept_prob = .5;
          } else {
            if (g.thread_rank() == 0) {
              float4 four_rands = curand_uniform4(&state);
              ran_rot = int(four_rands.x * nrotamers) % nrotamers;
              accept_prob = four_rands.y;
            }
            ran_rot = g.shfl(ran_rot, 0);
            accept_prob = g.shfl(accept_prob, 0);
          }
          int const ran_res = res_for_rot[ran_rot];
          int const local_prev_rot = rotamer_assignments[warp_id][ran_res];
          int const ran_res_nrots = nrotamers_for_res[ran_res];
          int const ran_res_rotamer_offset = oneb_offsets[ran_res];

          bool prev_rot_in_range = false;
	  int thread_w_prev_rot = 0;
          { // scope
            int const local_ran_rot_orig = ran_rot - ran_res_rotamer_offset;
            int const local_prev_rot_wrapped = local_ran_rot_orig < local_prev_rot ?
              local_prev_rot :
              local_prev_rot + ran_res_nrots;
            prev_rot_in_range = local_ran_rot_orig + 32 > local_prev_rot_wrapped;
	    thread_w_prev_rot = prev_rot_in_range ?
	      local_prev_rot_wrapped - local_ran_rot_orig : 0;
          }
          int const local_ran_rot = prev_rot_in_range ? (
            (ran_rot - ran_res_rotamer_offset + g.thread_rank()) % ran_res_nrots) :    
            (g.thread_rank() == 0 ?
              local_prev_rot :
              (ran_rot - ran_res_rotamer_offset  + g.thread_rank() - 1) % ran_res_nrots);
          ran_rot = local_ran_rot + ran_res_rotamer_offset;


          // If there are fewer rotamers on this residue than there are threads
          // active in the warp, do not wrap and consider a rotamer more than once
          bool const this_thread_active = ran_res_nrots > g.thread_rank();
	  bool const this_thread_last_active = ran_res_nrots == g.thread_rank() || g.thread_rank() == 32 - 1;

          float new_e = 9999;
	  if (this_thread_active) {
	    new_e = energy1b[ran_rot];
	  }

          // Temp: iterate across all residues instead of just the
          // neighbors of ran_rot_res
          if (this_thread_active) {
            for (int k=0; k < nres; ++k) {
              if (k == ran_res || nenergies[ran_res][k] == 0) {
                // alt_energies[k][warp_id] = 0;
                continue;
              }
              int const local_k_rot = rotamer_assignments[warp_id][k];
  
              int64_t const k_ran_offset = twob_offsets[k][ran_res];
              //int const kres_nrots = nrotamers_for_res[k];
  
              new_e += energy2b[k_ran_offset + ran_res_nrots * local_k_rot + local_ran_rot];
            }
	  }

          float const min_e = reduce_shfl(g, new_e, mgpu::minimum_t<float>());
	  // printf("thread %d min_e %f\n", thread_id, min_e);
          float myexp = expf( -1 * ( new_e - min_e ) / temperature );
	  // printf("thread %d myexp %f\n", thread_id, myexp);
          float const partition = reduce_shfl(g, myexp, mgpu::plus_t<float>());
	  // printf("thread %d partition %f\n", thread_id, partition);
          float const myprob = this_thread_active ? myexp / partition : 0;
	  // printf("thread %d myprob %f\n", thread_id, myprob);
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
	  accept_rank = inclusive_scan_shfl(g, accept_rank, mgpu::plus_t<int>());
	  // printf("thread %d accept_rank after scan %d\n", thread_id, accept_rank);
	  
	  bool accept = accept_rank == 1 && this_thread_active;
	  // printf("thread %d accept %d\n", thread_id, accept);
	  int const accept_thread = reduce_shfl(g, accept ? g.thread_rank() : -1, mgpu::maximum_t<int>());
	  // if (g.thread_rank() == 0) {
	  //   printf("thread %d accept_thread %d\n", thread_id, accept_thread);
	  // }

	  float prev_e = g.shfl(new_e, thread_w_prev_rot);
	  // printf("thread %d prev_e %f\n", thread_id, prev_e);

	  bool new_best = false;
          if (accept) {
	    float deltaE = new_e - prev_e;
	    // printf("deltaE: %f (%f - %f)\n", deltaE, new_e, prev_e);
            rotamer_assignments[warp_id][ran_res] = local_ran_rot;
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
	      best_rotamer_assignments[warp_id][k] = rotamer_assignments[warp_id][k];
	    }
	    best_energy = current_total_energy; // g.shfl(best_energy, accept_thread);
	  }

          ++ntrials;
          if (ntrials > 1000) {
            ntrials = 0;
            current_total_energy = total_energy_for_assignment_parallel(g,
              nrotamers_for_res, oneb_offsets, res_for_rot, nenergies, twob_offsets,
	      energy1b, energy2b, rotamer_assignments[warp_id]);
	    // if (g.thread_rank() == 0) {
	    //   printf("refresh total energy currentE %f\n", current_total_energy);
	    // }
          }

        } // end inner loop

	// geometric cooling toward 0.3
	// std::cout << "temperature " << temperature << " energy " <<
	//  total_energy_for_assignment(nrotamers_for_res, oneb_offsets,
	//    res_for_rot, nenergies, twob_offsets, energy1b, energy2b, my_rotamer_assignment) << std::endl;
	temperature = 0.35 * (temperature - low_temp) + low_temp;

      } // end outer loop


      scores[warp_id] = total_energy_for_assignment_parallel(g, 
	nrotamers_for_res, oneb_offsets, res_for_rot, nenergies, twob_offsets,
	energy1b, energy2b, rotamer_assignments[warp_id]
      );
    };

    // typedef typename conditional_typedef_t<
    //     launch_arg_t,
    //     launch_box_t<
    //         arch_20_cta<128, 11, 8>,
    //         arch_35_cta<128, 7, 5>,
    //         arch_52_cta<128, 11, 8> > >::type_t launch_t;


    mgpu::standard_context_t context;
    mgpu::transform<128, 1>(run_simulated_annealing, n_simA_threads, context);

    cudaDeviceSynchronize();
    clock_t stop = clock();
    std::cout << "GPU simulated annealing in " <<
      ((double) stop - start)/CLOCKS_PER_SEC << std::endl;

    return {scores_t, rotamer_assignments_t};
  }

};

template struct AnnealerDispatch<tmol::Device::CUDA>;

} // namespace compiled
} // namespace pack
} // namespace tmol
