#include <torch/extension.h>

#include <tmol/utility/tensor/TensorCollection.h>
#include <tmol/utility/tensor/pybind.h>

auto sum_tensor_collection(tmol::TCollection<float, 2, tmol::Device::CPU> input)
    -> tmol::TPack<float, 2, tmol::Device::CPU> {
  auto output_t =
      tmol::TPack<float, 2, tmol::Device::CPU>::zeros_like(input.view[0]);
  auto output = output_t.view;

  for (int i = 0; i < input.tensors.size(); ++i) {
    for (int j = 0; j < input.view[i].size(0); ++j) {
      for (int k = 0; k < input.view[i].size(1); ++k) {
        output[j][k] += input.view[i][j][k];
      }
    }
  }
  return output_t;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sum_tensor_collection", &sum_tensor_collection, "Sum a TCollection.");
}
