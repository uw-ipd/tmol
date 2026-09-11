#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <tmol/utility/tensor/torch_context.hh>

at::Tensor copy_via_context(at::Tensor const& input) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous());
  c10::cuda::CUDAGuard guard(input.device());
  static tmol::ContextManager manager;
  auto context = tmol::current_context(manager);
  mgpu::mem_t<unsigned char> scratch(input.nbytes(), *context);
  auto output = at::empty_like(input);
  if (input.nbytes() != 0) {
    C10_CUDA_CHECK(cudaMemcpyAsync(
        scratch.data(),
        input.const_data_ptr(),
        input.nbytes(),
        cudaMemcpyDeviceToDevice,
        context->stream()));
    C10_CUDA_CHECK(cudaMemcpyAsync(
        output.mutable_data_ptr(),
        scratch.data(),
        input.nbytes(),
        cudaMemcpyDeviceToDevice,
        context->stream()));
  }
  return output;
}
