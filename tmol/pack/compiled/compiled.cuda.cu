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
    clock_t start = clock();

    // No Frills Simulated Annealing!
    int const nres = nrotamers_for_res.size(0);
    int const nrotamers = res_for_rot.size(0);

    int n_simA_runs = 32 * 1000;
    
    auto scores_t = TPack<float, 1, D>::zeros({n_simA_runs});
    auto rotamer_assignments_t = TPack<int, 2, D>::zeros({nres, n_simA_runs});
    auto best_rotamer_assignments_t = TPack<int, 2, D>::zeros({nres, n_simA_runs});

    auto quench_order_t = TPack<int, 2, D>::zeros({nrotamers, n_simA_runs});

    auto scores = scores_t.view;
    auto rotamer_assignments = rotamer_assignments_t.view;
    auto best_rotamer_assignments = rotamer_assignments_t.view;
    auto quench_order = quench_order_t.view;


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

    auto run_simulated_annealing = [=] __device__ (int thread_id){
      curandStatePhilox4_32_10_t state;
      curand_init(
	philox_seed.first,
	thread_id,
	philox_seed.second,
	&state);

      // auto my_quench_order = quench_order[thread_id];
      // auto my_rotamer_assignment = rotamer_assignments[thread_id];
      // auto my_best_rotamer_assignment = rotamer_assignments[thread_id];

      for (int i = 0; i < nres; ++i) {
        int const i_nrots = nrotamers_for_res[i];
	int chosen = int(curand_uniform(&state) * i_nrots) % i_nrots;
	rotamer_assignments[i][thread_id] = chosen;
        best_rotamer_assignments[i][thread_id] = chosen;
      }

      float temperature = 100;
      float best_energy = total_energy_for_assignment(nrotamers_for_res, oneb_offsets,
        res_for_rot, nenergies, twob_offsets, energy1b, energy2b, rotamer_assignments,
	thread_id
      );
      float current_total_energy = best_energy;
      int ntrials = 0;
      for (int i = 0; i < 6; ++i) {

        bool quench = false;
        if (i == 5) {
	  quench = true;
	  temperature = 0;
	  for (int j = 0; j < nres; ++j) {
	    rotamer_assignments[j][thread_id] = best_rotamer_assignments[j][thread_id];
	  }
	  current_total_energy = total_energy_for_assignment(nrotamers_for_res, oneb_offsets,
	    res_for_rot, nenergies, twob_offsets, energy1b, energy2b, rotamer_assignments,
	    thread_id
	  );
        }

        for (int j = 0; j < nrotamers; ++j) {

	  int ran_rot;
	  float accept_prob(0);
	  if (quench) {
	    if (j % nrotamers == 0) {
	      set_quench_order(quench_order, thread_id, &state);
	    }
	    ran_rot = quench_order[j%nrotamers][thread_id];
	  } else {
	    float4 four_rands = curand_uniform4(&state);
	    ran_rot = int(four_rands.x * nrotamers) % nrotamers;
	    accept_prob = four_rands.y;
	  }
	  int const ran_res = res_for_rot[ran_rot];
	  int const local_prev_rot = rotamer_assignments[ran_res][thread_id];
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
	    int const local_k_rot = rotamer_assignments[k][thread_id];

	    //int const ran_k_offset = twob_offsets[ran_res][k];
	    int const k_ran_offset = twob_offsets[k][ran_res];
	    int const kres_nrots = nrotamers_for_res[k];
	    //new_e += energy2b[ran_k_offset + kres_nrots * local_ran_rot + local_k_rot];
	    new_e += energy2b[k_ran_offset + ran_res_nrots * local_k_rot + local_ran_rot];
	    prev_e += energy2b[k_ran_offset + ran_res_nrots * local_k_rot + local_prev_rot];
	  }

	  float const deltaE = new_e - prev_e;
	  if (local_prev_rot < 0 || pass_metropolis(temperature, accept_prob, deltaE, quench)) {
	    rotamer_assignments[ran_res][thread_id] = local_ran_rot;
	    current_total_energy = current_total_energy + deltaE;
	    if (current_total_energy < best_energy) {
	      for (int k=0; k < nres; ++k) {
		best_rotamer_assignments[k][thread_id] = rotamer_assignments[k][thread_id];
	      }
	      best_energy = current_total_energy;
	    }
      	  }


	  ++ntrials;
	  if (ntrials > 10000) {
	    ntrials = 0;
	    current_total_energy = total_energy_for_assignment(
	      nrotamers_for_res, oneb_offsets, res_for_rot,
	      nenergies, twob_offsets, energy1b, energy2b,
	      rotamer_assignments, thread_id);
	  }

	  // geometric cooling toward 0.3
	  // std::cout << "temperature " << temperature << " energy " <<
	  //  total_energy_for_assignment(nrotamers_for_res, oneb_offsets,
	  //    res_for_rot, nenergies, twob_offsets, energy1b, energy2b, my_rotamer_assignment) << std::endl;
	  temperature = 0.25 * (temperature - 0.3) + 0.3;
	}
      }


      scores[thread_id] = total_energy_for_assignment(nrotamers_for_res, oneb_offsets,
        res_for_rot, nenergies, twob_offsets, energy1b, energy2b, rotamer_assignments,
	thread_id
      );
    };

    mgpu::standard_context_t context;
    mgpu::transform(run_simulated_annealing, n_simA_runs, context);

    cudaDeviceSynchronize();
    clock_t stop = clock();
    std::cout << "Simulated annealing completed in " <<
      ((double) stop - start)/CLOCKS_PER_SEC << std::endl;

    return {scores_t, rotamer_assignments_t};
  }

};

template struct AnnealerDispatch<tmol::Device::CUDA>;

} // namespace compiled
} // namespace pack
} // namespace tmol
