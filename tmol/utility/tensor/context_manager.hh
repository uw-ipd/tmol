#pragma once

#include <memory>
#include <mutex>

namespace tmol {

class ContextManager {
 public:
  ContextManager() {}
  ~ContextManager() = default;

  bool has(void* address) { return contexts_.find(address) != contexts_.end(); }

  void set(
      void* address, std::pair<int, std::shared_ptr<void>> dev_and_context) {
    contexts_[address] = dev_and_context;
  }

  std::pair<int, std::shared_ptr<void>> get(void* address) {
    return contexts_[address];
  }

  std::mutex& get_mutex() { return mutex_; }

 private:
  std::map<void*, std::pair<int, std::shared_ptr<void>>> contexts_;
  std::mutex mutex_;
};

}  // namespace tmol
