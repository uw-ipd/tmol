#pragma once

#include <memory>
#include <mutex>

namespace tmol {

// This class will be used when running on non-CPU devices to manage the
// creation objects that are expensive to allocate and only need to be allocated
// once: and in particular, when running on CUDA gpus (the only non-CPU device
// we support at the time of this writing) will hold the moderngpu
// standard_context_t objects. The ContextManagers are allocated on a
// one-per-compilation-unit basis, which is already more than we ought to need,
// but is a workaround for the fact that each compilation unit we load with
// torch is blind to everything else we load with torch meaning that we don't
// have anything like persistent global (extern) data. So instead, each
// compiled.ops.cpp file holds one of these. Then the CUDA-kernel-launching
// functions that are accessed within those files will be able to access the
// contexts when they are asked to launch kernels.
class ContextManager {
 public:
  ContextManager() {}
  ~ContextManager() = default;

  inline bool has(std::pair<int, void*> device_index_and_stream_address) {
    return contexts_.find(device_index_and_stream_address) != contexts_.end();
  }

  inline void set(
      std::pair<int, void*> device_index_and_stream_address,
      std::shared_ptr<void> context) {
    contexts_[device_index_and_stream_address] = context;
  }

  inline std::shared_ptr<void> get(
      std::pair<int, void*> device_index_and_stream_address) {
    return contexts_[device_index_and_stream_address];
  }

  inline std::mutex& get_mutex() { return mutex_; }

 private:
  std::map<std::pair<int, void*>, std::shared_ptr<void>> contexts_;
  std::mutex mutex_;
};

}  // namespace tmol
