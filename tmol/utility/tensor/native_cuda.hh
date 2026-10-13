#pragma once
#include <tmol/utility/tensor/native.hh>
#include <cuda_runtime_api.h>

namespace tmol::native {
inline void cuda_check(cudaError_t error) {
  TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
}
struct CUDAGuard {
  int previous;
  explicit CUDAGuard(Device device) : CUDAGuard(device.index()) {}
  explicit CUDAGuard(int device) {
    cuda_check(cudaGetDevice(&previous));
    if (device != previous) cuda_check(cudaSetDevice(device));
  }
  ~CUDAGuard() { cudaSetDevice(previous); }
};
inline cudaStream_t current_cuda_stream() {
  int device;
  cuda_check(cudaGetDevice(&device));
  void* stream;
  STABLE_TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(device, &stream));
  return static_cast<cudaStream_t>(stream);
}
}  // namespace tmol::native
#define C10_CUDA_KERNEL_LAUNCH_CHECK() \
  tmol::native::cuda_check(cudaGetLastError())
