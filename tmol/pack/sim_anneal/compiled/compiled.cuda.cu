#include <ATen/CUDAGenerator.h>
#include <ATen/Context.h>
#include <THC/THCTensorRandom.h>
#include <c10/DeviceType.h>
#include <THC/THCGenerator.hpp>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>

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
  // static bool seeded = false;
  // if ( ! seeded ) {
  //   std::cout << "Setting RNG seed" << std::endl;
  //   THCRandom_manualSeed(at::globalContext().getTHCState(), 0);
  //   seeded = true;
  // }
  auto gen_ = THCRandom_getGenerator(at::globalContext().getTHCState());
  uint64_t offset = gen_->state.philox_seed_offset.fetch_add(increment);
  return std::make_pair(gen_->state.initial_seed, offset);
}

namespace tmol {
namespace pack {
namespace sim_anneal {
namespace compiled {

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
      TView<Int, 1, D> random_rots) -> void {
    // This code will work for future versions of the torch/aten libraries, but
    // not this one.
    // // Increment the cuda generator
    // std::pair<uint64_t, uint64_t> rng_engine_inputs;
    // at::CUDAGenerator * gen = at::cuda::detail::getDefaultCUDAGenerator();
    // {
    //   std::lock_guard<std::mutex> lock(gen->mutex_);
    //   rng_engine_inputs = gen->philox_engine_inputs(1);
    // }

    // Increment the seed (and capture the current seed) for the
    // cuda generator. The number of calls to curand must be known
    // by this statement -- there will be only a single call to curand
    //
    auto philox_seed = next_philox_seed(1);

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
      curand_init(philox_seed.first, i, philox_seed.second, &state);

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

    Dispatch<D>::forall(n_contexts, select_rotamer);

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

    Dispatch<D>::forall(n_contexts * 2 * max_n_atoms, copy_rotamer_coords);
  };
};

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct MetropolisAcceptReject {
  static auto f(
      TView<Real, 1, D> temperature,
      TView<Real, 4, D> context_coords,
      TView<Int, 2, D> context_block_type,
      TView<Real, 3, D> alternate_coords,
      TView<Int, 2, D> alternate_ids,
      TView<Real, 2, D> rotamer_component_energies,
      TView<Int, 1, D> accept) -> void {
    int const n_contexts = context_coords.size(0);
    int const n_terms = rotamer_component_energies.size(0);
    int const max_n_atoms = context_coords.size(2);

    assert(rotamer_component_energies.size(1) == 2 * n_contexts);
    assert(alternate_coords.size(0) == 2 * n_contexts);
    assert(alternate_coords.size(1) == max_n_atoms);
    assert(alternate_coords.size(2) == 3);
    assert(alternate_ids.size(0) == 2 * n_contexts);
    assert(accept.size(0) == n_contexts);

    // TEMP!!!
    // auto sum_energies_tp = TPack<Real, 1, D>::zeros({1});
    // auto sum_energies = sum_energies_tp.view;

    // auto accept_tp = TPack<Int, 1, D>::zeros({n_contexts});
    // auto accept = accept_tp.view;

    auto philox_seed = next_philox_seed(1);

    auto accept_reject = [=] MGPU_DEVICE(int i) {
      curandStatePhilox4_32_10_t state;
      curand_init(philox_seed.first, i, philox_seed.second, &state);

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
      Real temp = temperature[0];
      Real prob_accept = exp(-1 * deltaE / temp);
      bool i_accept = deltaE < 0 || rand_unif < prob_accept;
      accept[i] = i_accept;
      if (i_accept) {
        int block_id = alternate_ids[2 * i + 1][1];
        context_block_type[i][block_id] = alternate_ids[2 * i + 1][2];
      }
    };

    Dispatch<D>::forall(n_contexts, accept_reject);

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

    Dispatch<D>::forall(n_contexts * max_n_atoms, copy_accepted_coords);
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
