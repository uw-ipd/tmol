#pragma once

#include <moderngpu/context.hxx>
#include <c10/cuda/CUDAStream.h>
#include <tmol/utility/tensor/context_manager.hh>

#include <memory>
#include <mutex>

namespace tmol {

struct ContextDeleter {
  inline void operator()(void* ctxt) {
    mgpu::standard_context_t* std_ctxt =
        static_cast<mgpu::standard_context_t*>(ctxt);
    delete std_ctxt;
  }
};

// Get a pointer to an mgpu::standard_context_t that uses the
// current stream as set by torch; we let the pytorch/c10 libraries
// decide which stream to launch a function within.
inline std::shared_ptr<mgpu::standard_context_t> current_context(
    ContextManager& mgr) {
  c10::cuda::CUDAStream c10_stream = c10::cuda::getCurrentCUDAStream();
  cudaStream_t cuda_stream(c10_stream);
  void* cuda_stream_address = static_cast<void*>(cuda_stream);
  int device_index = c10_stream.device_index();
  std::pair<int, void*> device_index_and_stream_address(
      std::make_pair(device_index, cuda_stream_address));

  // We lock the mutex because writing to a std::map changes it.
  // This code is overly safe; a more complex strategy of obtaining
  // read locks for the general case and write locks for the specific
  // instances when new context objects must be allocated and stored
  // in the manager could be created, but such cases can lead to
  // deadlock if not done carefully, and our use case is basically
  // that a single python thread is calling the C++
  std::lock_guard(mgr.get_mutex());

  // We are will accumulate new standard_context_t objects over the lifetime
  // of execution and none of these objects will be deallocated until
  // the program ends. That means that repeated allocation of cuda stream
  // objects would be problematic. Torch does not do that: it allocates
  // stream pools. Furthermore, we mostly work in stream 0. However, if
  // this code were put to use in a context that for some reason repeatedly
  // allocated and deallocated cudaStream objects instead of holding
  // them in a pool, then this code would appear to leak memory.
  //
  // We allocate a new standard_context_t object for each device/stream pair
  // and will reuse that object over the lifetime of execution.
  // If we have already seen this device/stream pair, then we do not need
  // to allocate a new one: just return the already allocated one.
  if (mgr.has(device_index_and_stream_address)) {
    std::shared_ptr<void> context_ptr =
        mgr.get(device_index_and_stream_address);
    return std::static_pointer_cast<mgpu::standard_context_t>(context_ptr);
  }
  // okay, we need to create a new standard_context_t and that
  // can take a lot of time, (~1.5ms), so we try not to do this much.
  // Args to standar_context_t ctor:
  // 1. false: do not print the device properties to std::cout
  // 2. the cuda stream we're sending this to
  ContextDeleter dstor_functor_instance;
  std::shared_ptr<mgpu::standard_context_t> new_context(
      new mgpu::standard_context_t(false, cuda_stream), dstor_functor_instance);
  mgr.set(
      device_index_and_stream_address,
      std::static_pointer_cast<void>(new_context));
  return new_context;
}

}  // namespace tmol
