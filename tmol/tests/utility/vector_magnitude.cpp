#include <torch/torch.h>
#include <cmath>

at::Tensor vector_magnitude_aten(at::Tensor input) {
  return (input * input).sum(-1).sqrt();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "aten", &vector_magnitude_aten, "ATen-based vector_magnitude function.");
}
