#pragma once

#include <moderngpu/context.hxx>
#include <c10/cuda/CUDAStream.h>
#include <tmol/utility/tensor/context_manager.hh>

#include <memory>
#include <mutex>

namespace tmol {

struct ContextDeleter {
  static void operator()(void* ctxt) {
    mgpu::standard_context_t* std_ctxt =
        static_cast<mgpu::standard_context_t*>(ctxt);
    delete std_ctxt;
  }
};

inline std::shared_ptr<mgpu::standard_context_t> current_context(
    ContextManager& mgr) {
  c10::cuda::CUDAStream c10_stream = c10::cuda::getCurrentCUDAStream();
  cudaStream_t cuda_stream(c10_stream);
  void* cuda_stream_address = static_cast<void*>(cuda_stream);
  int device_index = c10_stream.device_index();

  std::lock_guard(mgr.get_mutex());
  if (mgr.has(cuda_stream_address)) {
    std::pair<int, std::shared_ptr<void>> dev_ind_and_context_ptr =
        mgr.get(cuda_stream_address);
    if (dev_ind_and_context_ptr.first == device_index) {
      // Assumption:
      // if we have a stream on a particular device, then
      // even if that stream has been deallocated since
      // the standard_context_t was created for its
      // address has been reused to point at a new cuda stream
      // that it doesn't matter from the perspective of the
      // standard_context_t: the context is essentially still valid

      return std::static_pointer_cast<mgpu::standard_context_t>(
          dev_ind_and_context_ptr.second);
    }
  }
  // okay, we need to create a new standard_context_t and that
  // can take a little time, (~1.5ms), so we try not to do this much.
  // Args to standar_context_t:
  // 1. false: do not print the device properties to std::cout
  // 2. the cuda stream we're sending this to
  ContextDeleter dstor_functor_instance;
  std::shared_ptr<mgpu::standard_context_t> new_context(
      new mgpu::standard_context_t(false, cuda_stream), dstor_functor_instance);
  mgr.set(
      cuda_stream_address,
      std::make_pair(
          device_index, std::static_pointer_cast<void>(new_context)));
  return new_context;
}

}  // namespace tmol
