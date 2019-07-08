#pragma once

#include "stream.hh"

#ifdef __NVCC__
#include <moderngpu/context.hxx>
#endif

namespace tmol {
namespace utility {
namespace cuda {


#ifdef __NVCC__
// Create an MGPU context from a given stream
inline
mgpu::standard_context_t
context_from_stream(
  tmol::utility::cuda::CUDAStream stream) {
  if (stream.stream_) {
    return mgpu::standard_context_t(stream.stream_->stream());
  } else {
    return mgpu::standard_context_t();
  }
}
#endif



}
}
}
