#pragma once

// This is a wrapper around ATen's (eventually c10's)
// CUDAEvent class, which produces errors in the
// absence of the NVidia libraries.

#include <memory>
#include "stream.hh"

#ifdef WITH_CUDA

#include <ATen/cuda/CUDAEvent.h>

#else

namespace at {
namespace cuda {

// If cuda is absent, then, define an empty class
// so that the tmol::utility::cuda::CUDAEvent class
// can hold a null pointer to it.
class CUDAEvent {};

}
}

#endif

namespace tmol {
namespace utility {
namespace cuda {

struct CUDAEvent{
  CUDAEvent():
#ifdef __NVCC__    
    event_(std::make_shared<at::cuda::CUDAEvent>())
#else
    event_(nullptr)
#endif
  {}

  CUDAEvent(std::shared_ptr<at::cuda::CUDAEvent> event) :
    event_(event)
  {}

  void record() {
#ifdef __NVCC__
    if (event_) {
      event_->record();
    }
#endif

  }

  void block(CUDAStream stream)
  {
#ifdef __NVCC__
    if (event_ && stream.stream_) {
      event_->block(*stream.stream_);
    }
#endif
  }

  void synchronize() {
#ifdef __NVCC__
    if (event_) {
      cudaEventSynchronize(*event_);
    }
#endif
  }

  std::shared_ptr<at::cuda::CUDAEvent> event_;

};

}
}
}
