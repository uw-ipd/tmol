#include <torch/torch.h>
#include <cmath>
#include <Eigen/Core>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>

at::Tensor vector_magnitude_aten(at::Tensor input) {
  return (input * input).sum(-1).sqrt();
}

at::Tensor vector_magnitude_accessor(at::Tensor input_t) {
  auto output_t = at::empty(input_t.type(), input_t.size(0));

  auto input = input_t.accessor<float, 2>();
  auto output = output_t.accessor<float, 1>();

  for(int64_t i = 0; i < input.size(0); ++i) {
    auto v = input[i];
    output[i] = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  }

  return output_t;
}

at::Tensor vector_magnitude_eigen(at::Tensor input_t) {
  auto output_t = at::empty(input_t.type(), input_t.size(0));

  auto input = tmol::reinterpret_tensor<Eigen::Vector3f, float, 2>(input_t);
  auto output = output_t.accessor<float, 1>();

  for(int64_t i = 0; i < input.size(0); ++i) {
    auto v = input[i][0];
    output[i] = std::sqrt(v.dot(v));
  }

  return output_t;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("aten", &vector_magnitude_aten, "ATen-based vector_magnitude function.");
  m.def("accessor", &vector_magnitude_accessor, "TensorAccessor-based vector_magnitude function.");
  m.def("eigen", &vector_magnitude_eigen, "Eigen-based vector_magnitude function.");
}
