#include <tmol/utility/tensor/native_cuda.hh>
#pragma once

#include <moderngpu/context.hxx>
#include <tmol/utility/tensor/context_manager.hh>

#include <memory>
#include <mutex>

namespace tmol {

// Route temporary device storage through Torch's stream-aware allocator.
// Host-pinned allocations retain ModernGPU's allocator.
class TorchCudaContext : public mgpu::standard_context_t {
 public:
  TorchCudaContext(cudaStream_t stream, int device)
      : mgpu::standard_context_t(false, stream), device_(device) {}

  void* alloc(size_t size, mgpu::memory_space_t space) override {
    if (space != mgpu::memory_space_device) {
      return mgpu::standard_context_t::alloc(size, space);
    }
    if (size == 0) return nullptr;
    native::CUDAGuard guard(device_);
    CUDAStreamGuardHandle stream_guard;
    STABLE_TORCH_ERROR_CODE_CHECK(
        aoti_torch_create_cuda_stream_guard(stream(), device_, &stream_guard));
    void* pointer;
    auto status = aoti_torch_cuda_caching_allocator_raw_alloc(size, &pointer);
    STABLE_TORCH_ERROR_CODE_CHECK(
        aoti_torch_delete_cuda_stream_guard(stream_guard));
    STABLE_TORCH_ERROR_CODE_CHECK(status);
    return pointer;
  }

  void free(void* pointer, mgpu::memory_space_t space) override {
    if (space != mgpu::memory_space_device) {
      mgpu::standard_context_t::free(pointer, space);
    } else if (pointer != nullptr) {
      STABLE_TORCH_ERROR_CODE_CHECK(
          aoti_torch_cuda_caching_allocator_raw_delete(pointer));
    }
  }

 private:
  int device_;
};

// Get a pointer to an mgpu::standard_context_t that uses the
// current stream as set by torch; we let the pytorch/c10 libraries
// decide which stream to launch a function within.
inline std::shared_ptr<mgpu::standard_context_t> current_context(
    ContextManager& mgr) {
  auto cuda_stream = native::current_cuda_stream();
  void* cuda_stream_address = static_cast<void*>(cuda_stream);
  int device_index;
  native::cuda_check(cudaGetDevice(&device_index));
  std::pair<int, void*> device_index_and_stream_address(
      std::make_pair(device_index, cuda_stream_address));

  // Hold the mutex through lookup and insertion: calls on different Python
  // threads may create contexts for different streams concurrently.
  std::lock_guard lock(mgr.get_mutex());

  // We accumulate new standard_context_t objects over the lifetime
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
  // The base destructor is nonvirtual; retain the concrete shared deleter.
  std::shared_ptr<mgpu::standard_context_t> new_context =
      std::make_shared<TorchCudaContext>(cuda_stream, device_index);
  mgr.set(
      device_index_and_stream_address,
      std::static_pointer_cast<void>(new_context));
  return new_context;
}

}  // namespace tmol
