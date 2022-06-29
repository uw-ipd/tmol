
// The location of CUDAGeneratorImpl changed in torch 1.11
#if TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR < 11
#include <ATen/CUDAGeneratorImpl.h>
#else
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

#include <ATen/Context.h>
// #include <THC/THCTensorRandom.h>
#include <c10/core/DeviceType.h>
#include <c10/cuda/CUDAStream.h>
//#include <THC/THCGenerator.hpp>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/gpu_error_check.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/forall_dispatch.cuda.impl.cuh>

#include "simulated_annealing.hh"

#include <cooperative_groups.h>
#include <moderngpu/cta_reduce.hxx>
#include <moderngpu/kernel_compact.hxx>
#include <moderngpu/kernel_mergesort.hxx>
#include <moderngpu/transform.hxx>

#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

#include <ctime>

// TEMP // Stolen from torch, v1.0.0
// TEMP // Expose part of the torch library that otherwise is
// TEMP // not part of the API.
// TEMP THCGenerator* THCRandom_getGenerator(THCState* state);
// TEMP
// TEMP // Stolen from torch, v1.0.0;
// TEMP // unnecessary in the latest release, where this function
// TEMP // is built in to CUDAGenerator.
// TEMP // Modified slightly as the input Generator is unused.
// TEMP // increment should be at least the number of curand() random numbers
// used in TEMP // each thread. TEMP std::pair<uint64_t, uint64_t>
// next_philox_seed(uint64_t increment) { TEMP   // static bool seeded = false;
// TEMP   // if ( ! seeded ) {
// TEMP   //   std::cout << "Setting RNG seed" << std::endl;
// TEMP   //   THCRandom_manualSeed(at::globalContext().getTHCState(), 0);
// TEMP   //   seeded = true;
// TEMP   // }
// TEMP   auto gen_ = THCRandom_getGenerator(at::globalContext().getTHCState());
// TEMP   uint64_t offset = gen_->state.philox_seed_offset.fetch_add(increment);
// TEMP   return std::make_pair(gen_->state.initial_seed, offset);
// TEMP }

namespace tmol {
namespace pack {
namespace sim_anneal {
namespace compiled {

cudaStream_t packer_stream(0);
int count_pick_passes(0);
int count_mc_passes(0);

/// @brief Return a uniformly-distributed integer in the range
/// between 0 and n-1.
/// Note that curand_uniform() returns a random number in the range
/// (0,1], unlike unlike rand() returns a random number in the range
/// [0,1). Take care with curand_uniform().
__device__ inline int curand_in_range(
    curandStatePhilox4_32_10_t* state, int n) {
  return int(curand_uniform(state) * n) % n;
}

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct PickRotamers {
  static auto f(
      TView<Real, 4, D> context_coords,
      TView<Int, 2, D> context_block_type,
      TView<Int, 1, D> pose_id_for_context,
      TView<Int, 1, D> n_rots_for_pose,
      TView<Int, 1, D> rot_offset_for_pose,
      TView<Int, 1, D> block_type_ind_for_rot,
      TView<Int, 1, D> block_ind_for_rot,
      TView<Real, 3, D> rotamer_coords,
      TView<Real, 3, D> alternate_coords,
      TView<Int, 2, D> alternate_block_id,
      TView<Int, 1, D> random_rots,
      TView<int64_t, 1, tmol::Device::CPU> annealer_event) -> void {
    // Increment the cuda generator and capture the set for this execution
    std::pair<uint64_t, uint64_t> rng_engine_inputs;
    auto gen = at::check_generator<at::CUDAGeneratorImpl>(
        at::cuda::detail::getDefaultCUDAGenerator());
    {
      // aquire lock when using random generators
      std::lock_guard<std::mutex> lock(gen->mutex_);
      rng_engine_inputs = gen->philox_engine_inputs(1);
    }

    // Increment the seed (and capture the current seed) for the
    // cuda generator. The number of calls to curand must be known
    // by this statement -- there will be only a single call to curand
    //
    // auto philox_seed = next_philox_seed(1);

    int const n_contexts = context_coords.size(0);
    int const max_n_blocks = context_coords.size(1);
    int const max_n_atoms = context_coords.size(2);
    int const n_poses = pose_id_for_context.size(0);
    int const n_rots = block_type_ind_for_rot.size(0);

    assert(context_coords.size(3) == 3);
    assert(context_block_type.size(0) == n_contexts);
    assert(context_block_type.size(1) == max_n_blocks);
    assert(n_rots_for_pose.size(0) == n_poses);
    assert(rot_offset_for_pose.size(0) == n_poses);
    assert(block_ind_for_rot.size(0) == n_rots);
    assert(rotamer_coords.size(0) == n_rots);
    assert(rotamer_coords.size(1) == max_n_atoms);
    assert(rotamer_coords.size(2) == 3);
    assert(random_rots.size(0) == n_contexts);
    assert(alternate_coords.size(0) == 2 * n_contexts);
    assert(alternate_coords.size(1) == max_n_atoms);
    assert(alternate_coords.size(2) == 3);
    assert(alternate_block_id.size(0) == 2 * n_contexts);
    assert(alternate_block_id.size(1) == 3);

    auto select_rotamer = [=] MGPU_DEVICE(int i) {
      curandStatePhilox4_32_10_t state;
      curand_init(rng_engine_inputs.first, i, rng_engine_inputs.second, &state);

      Int i_pose = pose_id_for_context[i];
      Int i_n_rots = n_rots_for_pose[i_pose];

      if (i_n_rots == 0) {
        alternate_block_id[i * 2][0] = -1;
        alternate_block_id[i * 2][1] = -1;
        alternate_block_id[i * 2][2] = -1;
        alternate_block_id[i * 2 + 1][0] = -1;
        alternate_block_id[i * 2 + 1][1] = -1;
        alternate_block_id[i * 2 + 1][2] = -1;
        random_rots[i] = -1;
      } else {
        Int i_rot_local = curand_in_range(&state, i_n_rots);
        Int i_rot_global = i_rot_local + rot_offset_for_pose[i_pose];
        Int i_block = block_ind_for_rot[i_rot_global];
        random_rots[i] = i_rot_global;

        alternate_block_id[i * 2][0] = i;
        alternate_block_id[i * 2][1] = i_block;
        alternate_block_id[i * 2][2] = context_block_type[i][i_block];
        alternate_block_id[i * 2 + 1][0] = i;
        alternate_block_id[i * 2 + 1][1] = i_block;
        alternate_block_id[i * 2 + 1][2] = block_type_ind_for_rot[i_rot_global];
      }
    };

    // auto random_rots_cpu_tp = TPack<Int, 1,
    // tmol::Device::CPU>::zeros({n_contexts}); auto random_rots_cpu =
    // random_rots_cpu_tp.view; cudaMemcpy(&random_rots_cpu[0], &random_rots[0],
    // sizeof(Int) * n_contexts, cudaMemcpyDeviceToHost); for (int i = 0; i <
    // n_contexts; ++i) {
    //   std::cout << " " << random_rots_cpu[i];
    // }
    // std::cout << std::endl;

    auto copy_rotamer_coords = [=] EIGEN_DEVICE_FUNC(int i) {
      Int alt_id = i / max_n_atoms;
      Int i_context = alternate_block_id[alt_id][0];
      Int i_block = alternate_block_id[alt_id][1];

      // pretend we're responsible for this atom; treat this like
      // our thread index; it's not, but, it'll do
      Int quasi_atom_ind = (i % max_n_atoms);

      if (alt_id % 2 == 0) {
        // strided iteration, more or less
        for (int j = 0; j < 3; ++j) {
          int j_count = j * max_n_atoms + quasi_atom_ind;
          int atom_id = j_count / 3;
          int dim = j_count % 3;

          alternate_coords[alt_id][atom_id][dim] =
              context_coords[i_context][i_block][atom_id][dim];
        }
      } else {
        // strided iteration, more or less
        Int i_rot = random_rots[i_context];
        for (int j = 0; j < 3; ++j) {
          int j_count = j * max_n_atoms + quasi_atom_ind;
          int atom_id = j_count / 3;
          int dim = j_count % 3;
          alternate_coords[alt_id][atom_id][dim] =
              rotamer_coords[i_rot][atom_id][dim];
        }
      }
    };

    ++count_pick_passes;

    if (packer_stream == 0) {
      // packer_stream = at::cuda::getStreamFromPool().stream();
      cudaStreamCreate(&packer_stream);
    }
    // mgpu::standard_context_t context(packer_stream);
    mgpu::standard_context_t context;
    // Dispatch<D>::forall(n_contexts, select_rotamer);
    mgpu::transform(select_rotamer, n_contexts, context);

    // Dispatch<D>::forall(n_contexts * 2 * max_n_atoms, copy_rotamer_coords);
    mgpu::transform(copy_rotamer_coords, n_contexts * 2 * max_n_atoms, context);

    // Record an event for the completion of the initialization of new
    // coordinates into the alternate_coords and alternate_block_id tensors so
    // that the score terms can wait until the rotamer coordinates are ready to
    // be evaluated
    if (annealer_event[0] != 0) {
      auto annealer_event_ptr =
          reinterpret_cast<cudaEvent_t>(annealer_event[0]);
      cudaEventRecord(annealer_event_ptr, context.stream());
      // std::cout << "Pick Rots " << count_pick_passes << ": recorded new event
      // " << annealer_event_ptr << " in stream " << context.stream() <<
      // std::endl;
    }
  }
};

void wait_on_score_events(
    cudaStream_t stream, TView<int64_t, 1, tmol::Device::CPU> score_events) {
  int const n_score_terms = score_events.size(0);
  for (int i = 0; i < n_score_terms; ++i) {
    cudaEvent_t event = reinterpret_cast<cudaEvent_t>(score_events[i]);
    if (!event) {
      // not all entries in the score_events_ tensor are
      // non-null
      continue;
    }
    cudaError_t status = cudaEventQuery(event);
    // std::cout << "MC " << count_mc_passes << " Event " << event << " status "
    // << status << " (success =" << cudaSuccess << ", ErrorNotReady=" <<
    // cudaErrorNotReady << ")" << std::endl;
    if (status == cudaSuccess) {
      // no need to wait
    } else if (status == cudaErrorNotReady) {
      // std::cout << "MC AcceptReject " << count_mc_passes << " waiting on
      // event " << event << " in stream " << stream << std::endl;
      cudaStreamWaitEvent(stream, event, 0);
    } else {
      // potential error situation?
    }
  }
}

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct MetropolisAcceptReject {
  static auto f(
      TView<Real, 1, tmol::Device::CPU> temperature,
      TView<Real, 4, D> context_coords,
      TView<Int, 2, D> context_block_type,
      TView<Real, 3, D> alternate_coords,
      TView<Int, 2, D> alternate_ids,
      TView<Real, 2, D> rotamer_component_energies,
      TView<Int, 1, D> accept,
      TView<int64_t, 1, tmol::Device::CPU> score_events) -> void {
    int const n_contexts = context_coords.size(0);
    int const n_terms = rotamer_component_energies.size(0);
    int const max_n_atoms = context_coords.size(2);

    assert(rotamer_component_energies.size(1) == 2 * n_contexts);
    assert(alternate_coords.size(0) == 2 * n_contexts);
    assert(alternate_coords.size(1) == max_n_atoms);
    assert(alternate_coords.size(2) == 3);
    assert(alternate_ids.size(0) == 2 * n_contexts);
    assert(accept.size(0) == n_contexts);
    assert(score_events.size(0) == n_terms);

    // TEMP!!!
    // auto sum_energies_tp = TPack<Real, 1, D>::zeros({1});
    // auto sum_energies = sum_energies_tp.view;

    // auto accept_tp = TPack<Int, 1, D>::zeros({n_contexts});
    // auto accept = accept_tp.view;

    std::pair<uint64_t, uint64_t> rng_engine_inputs;
    auto gen = at::check_generator<at::CUDAGeneratorImpl>(
        at::cuda::detail::getDefaultCUDAGenerator());
    {
      // aquire lock when using random generators
      std::lock_guard<std::mutex> lock(gen->mutex_);
      rng_engine_inputs = gen->philox_engine_inputs(1);
    }
    // auto philox_seed = next_philox_seed(1);

    Real const temp = temperature[0];
    ++count_mc_passes;
    int const n_mc_passes = count_mc_passes;

    auto accept_reject = [=] MGPU_DEVICE(int i) {
      curandStatePhilox4_32_10_t state;
      curand_init(rng_engine_inputs.first, i, rng_engine_inputs.second, &state);

      Real altE = 0;
      Real currE = 0;
      for (int j = 0; j < n_terms; ++j) {
        currE += rotamer_component_energies[j][2 * i];
        altE += rotamer_component_energies[j][2 * i + 1];
        rotamer_component_energies[j][2 * i] = 0;
        rotamer_component_energies[j][2 * i + 1] = 0;
      }
      // Real sumE = altE + currE;
      // score::common::accumulate<D, Real>::add(sum_energies[0], sumE);
      Real deltaE = altE - currE;
      Real rand_unif = curand_uniform(&state);
      Real prob_accept = temp > 0 ? exp(-1 * deltaE / temp) : 0;
      bool i_accept = deltaE < 0 || rand_unif < prob_accept;
      // if (n_mc_passes % 1000 == 1) {
      //   printf(
      //       "accept reject temp=%f tid=%d dE=%f runif=%f proba=%f
      //       iaccept=%d\n", temp, i, deltaE, rand_unif, prob_accept,
      //       i_accept);
      // }
      accept[i] = i_accept;
      if (i_accept) {
        int block_id = alternate_ids[2 * i + 1][1];
        context_block_type[i][block_id] = alternate_ids[2 * i + 1][2];
      }
    };

    auto copy_accepted_coords = [=] MGPU_DEVICE(int i) {
      // if (i == 0) {
      //   printf("n total atoms calc'd: %f\n", sum_energies[0]);
      // }
      int context_id = i / max_n_atoms;
      Int quasi_atom_ind = i % max_n_atoms;

      Int accepted = accept[context_id];
      if (accepted) {
        int block_id = alternate_ids[2 * context_id + 1][1];
        for (int j = 0; j < 3; ++j) {
          int j_count = j * max_n_atoms + quasi_atom_ind;
          int atom_id = j_count / 3;
          int dim = j_count % 3;

          context_coords[context_id][block_id][atom_id][dim] =
              alternate_coords[2 * context_id + 1][atom_id][dim];
        }
      }
    };

    // First we ensure that scoring has completed,
    // then we make the accept/reject decision for each
    // trajectory. Finally, we copy the coordinates
    // from the trajectories that have accepted substitutions
    // into the context_coords tensor.
    if (packer_stream == 0) {
      packer_stream = c10::cuda::getStreamFromPool().stream();
    }

    // mgpu::standard_context_t context(packer_stream);
    mgpu::standard_context_t context;
    wait_on_score_events(context.stream(), score_events);
    mgpu::transform(accept_reject, n_contexts, context);
    gpuErrchk(cudaPeekAtLastError());
    mgpu::transform(copy_accepted_coords, n_contexts * max_n_atoms, context);
    gpuErrchk(cudaPeekAtLastError());

    // if (count_mc_passes % 100 == 0) {
    //   using namespace mgpu;
    //   typedef launch_box_t<
    //       arch_20_cta<32, 1>,
    //       arch_35_cta<32, 1>,
    //       arch_52_cta<32, 1>>
    //       launch_t;
    //   gpuErrchk(cudaPeekAtLastError());
    //
    //   // std::cout << "device sync..." << std::flush;
    //   gpuErrchk(cudaDeviceSynchronize());
    //   // std::cout << "synced" << std::endl;
    //
    //   auto n_accepted_tp = TPack<Int, 1, D>::zeros({1});
    //   auto n_accepted = n_accepted_tp.view;
    //   gpuErrchk(cudaPeekAtLastError());
    //   //std::cout << "n_accepted: " << &n_accepted[0] << std::endl;
    //
    //   auto count_accepted = ([=] MGPU_DEVICE(int tid, int cta) {
    //     typedef typename launch_t::sm_ptx params_t;
    //     enum {
    //       nt = params_t::nt,
    //       vt = params_t::vt,
    //       vt0 = params_t::vt0,
    //       nv = nt * vt
    //     };
    //     typedef mgpu::cta_reduce_t<nt, Int> reduce_t;
    //
    //     __shared__ struct { typename reduce_t::storage_t reduce; } shared;
    //     Int local_sum = 0;
    //
    //     for (int i = tid; i < accept.size(0); i += blockDim.x) {
    //       local_sum += accept[i];
    //     }
    //
    //     Int cta_total = reduce_t().reduce(
    //         tid, local_sum, shared.reduce, nt, mgpu::plus_t<Int>());
    //     if (tid == 0) {
    //       n_accepted[0] = cta_total;
    //       // printf("cta total: %d\n", cta_total);
    //     }
    //   });
    //
    //   // std::cout << "launching kernel..." << std::flush;
    //   mgpu::cta_launch<launch_t>(count_accepted, 1, context);
    //   gpuErrchk(cudaPeekAtLastError());
    //
    //   // std::cout << "launched; memcpy sync..." << std::flush;
    //   Int cpu_n_accepted(0);
    //   gpuErrchk(cudaMemcpy(
    //       &cpu_n_accepted,
    //       &n_accepted[0],
    //       sizeof(Int),
    //       cudaMemcpyDeviceToHost));
    //
    //   // std::cout << "N accepted at temp " << temp << ": " << cpu_n_accepted
    //   //           << std::endl;
    // }

    // Dispatch<D>::forall(n_contexts, accept_reject);
    // Dispatch<D>::forall(n_contexts * max_n_atoms, copy_accepted_coords);
  }
};

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct FinalOp {
  static auto f() -> void {
    cudaDeviceSynchronize();
    // auto gen_ = THCRandom_getGenerator(at::globalContext().getTHCState());
    // gen_->set_current_seed(0);
  }
};

template struct PickRotamers<
    score::common::ForallDispatch,
    tmol::Device::CUDA,
    float,
    int32_t>;
template struct PickRotamers<
    score::common::ForallDispatch,
    tmol::Device::CUDA,
    double,
    int32_t>;
template struct PickRotamers<
    score::common::ForallDispatch,
    tmol::Device::CUDA,
    float,
    int64_t>;
template struct PickRotamers<
    score::common::ForallDispatch,
    tmol::Device::CUDA,
    double,
    int64_t>;

template struct MetropolisAcceptReject<
    score::common::ForallDispatch,
    tmol::Device::CUDA,
    float,
    int32_t>;
template struct MetropolisAcceptReject<
    score::common::ForallDispatch,
    tmol::Device::CUDA,
    double,
    int32_t>;
template struct MetropolisAcceptReject<
    score::common::ForallDispatch,
    tmol::Device::CUDA,
    float,
    int64_t>;
template struct MetropolisAcceptReject<
    score::common::ForallDispatch,
    tmol::Device::CUDA,
    double,
    int64_t>;

template struct FinalOp<
    score::common::ForallDispatch,
    tmol::Device::CUDA,
    float,
    int32_t>;
template struct FinalOp<
    score::common::ForallDispatch,
    tmol::Device::CUDA,
    double,
    int32_t>;
template struct FinalOp<
    score::common::ForallDispatch,
    tmol::Device::CUDA,
    float,
    int64_t>;
template struct FinalOp<
    score::common::ForallDispatch,
    tmol::Device::CUDA,
    double,
    int64_t>;

}  // namespace compiled
}  // namespace sim_anneal
}  // namespace pack
}  // namespace tmol
