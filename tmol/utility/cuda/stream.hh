#pragma once

#include <memory>

// This is a wrapper around ATen's (eventually c10's)
// CUDAStream class, which produces errors in the
// absence of the NVidia libraries.


#ifdef WITH_CUDA

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAStream.h>

#else

namespace at {
namespace cuda {

// If cuda is absent, then, define an empty class
// so that the tmol::utility::cuda::CUDAStream class
// can hold a null pointer to it.
class CUDAStream {};

}
}

#endif


namespace tmol {
namespace utility {
namespace cuda {

// The class to wrap an ATen cuda stream
// or hold a null pointer in the absence
// of CUDA.
struct CUDAStream {
  CUDAStream(std::shared_ptr<at::cuda::CUDAStream> stream) :
    stream_(stream)
  {}
  std::shared_ptr<at::cuda::CUDAStream> stream_;
};

// Request one of the pre-allocated streams from
// the ATen library for the current device (-1),
// and wrap that stream in a
// tmol::utility::cuda::CUDAStream object
inline
CUDAStream
get_cuda_stream_from_pool() {
#ifdef __NVCC__
  auto stream_ptr = std::make_shared<at::cuda::CUDAStream>(
    at::cuda::CUDAStream::UNCHECKED,
    at::cuda::getStreamFromPool().unwrap()
  );
  return tmol::utility::cuda::CUDAStream(stream_ptr);
#else
  // Return a class wrapping a null pointer
  return tmol::utility::cuda::CUDAStream(nullptr);
#endif
}

inline
CUDAStream
get_current_cuda_stream() {
#ifdef __NVCC__
  auto stream_ptr = std::make_shared<at::cuda::CUDAStream>(
    at::cuda::CUDAStream::UNCHECKED,
    at::cuda::getCurrentCUDAStream().unwrap()
  );
  return tmol::utility::cuda::CUDAStream(stream_ptr);
#else
  // Return a class wrapping a null pointer
  return tmol::utility::cuda::CUDAStream(nullptr);
#endif
}


inline
CUDAStream
get_default_stream() {
#ifdef __NVCC__
  // Request the default stream from the ATen library
  auto stream_ptr = std::make_shared<at::cuda::CUDAStream>(
    at::cuda::CUDAStream::UNCHECKED,
    at::cuda::getDefaultCUDAStream().unwrap()
  );
  return tmol::utility::cuda::CUDAStream(stream_ptr);
#else
  return tmol::utility::cuda::CUDAStream(nullptr);
#endif
}

inline
void
set_current_cuda_stream(
#ifdef __NVCC__
  CUDAStream const & stream
#else
  CUDAStream const &
#endif
)
{
#ifdef __NVCC__
  if (stream.stream_) {
    at::cuda::setCurrentCUDAStream(*stream.stream_);
  }
#endif
}

inline
void
set_default_cuda_stream()
{
#ifdef __NVCC__
  // Request the default stream from the ATen library
  auto stream = at::cuda::getDefaultCUDAStream(-1);
  at::cuda::setCurrentCUDAStream(stream);
#endif
}

}
}
}
