#include <torch/extension.h>

at::Tensor copy_via_context(at::Tensor const& input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "copy",
      &copy_via_context,
      pybind11::call_guard<pybind11::gil_scoped_release>());
}
