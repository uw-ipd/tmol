#include <tmol/utility/tensor/native_library.hh>

namespace tmol::native {
Tensor Tensor::reshape(IntList shape) const {
  if (needs_grad && GradMode::is_enabled()) {
    py::gil_scoped_acquire gil;
    return from_python(to_python(*this).attr("reshape")(
        py::cast(std::vector<int64_t>(shape.begin(), shape.end()))));
  }
  return stable::reshape(handle(), shape);
}
Tensor Tensor::narrow(int64_t d, int64_t start, int64_t length) const {
  if (needs_grad && GradMode::is_enabled()) {
    py::gil_scoped_acquire gil;
    return from_python(to_python(*this).attr("narrow")(d, start, length));
  }
  auto input = handle();
  return stable::narrow(input, d, start, length);
}
Tensor Tensor::select(int64_t d, int64_t i) const {
  if (needs_grad && GradMode::is_enabled()) {
    py::gil_scoped_acquire gil;
    return from_python(to_python(*this).attr("select")(d, i));
  }
  return stable::select(handle(), d, i);
}
Tensor Tensor::contiguous() const {
  if (needs_grad && GradMode::is_enabled()) {
    py::gil_scoped_acquire gil;
    return from_python(to_python(*this).attr("contiguous")());
  }
  return stable::contiguous(handle());
}
Tensor Tensor::unsqueeze(int64_t d) const {
  if (needs_grad && GradMode::is_enabled()) {
    py::gil_scoped_acquire gil;
    return from_python(to_python(*this).attr("unsqueeze")(d));
  }
  return stable::unsqueeze(handle(), d);
}
Tensor Tensor::squeeze(int64_t d) const {
  if (needs_grad && GradMode::is_enabled()) {
    py::gil_scoped_acquire gil;
    return from_python(to_python(*this).attr("squeeze")(d));
  }
  return stable::squeeze(handle(), d);
}
Tensor Tensor::sum(IntList dims) const {
  if (needs_grad && GradMode::is_enabled()) {
    py::gil_scoped_acquire gil;
    return from_python(to_python(*this).attr("sum")(
        py::cast(std::vector<int64_t>(dims.begin(), dims.end()))));
  }
  return stable::sum(handle(), dims);
}
Tensor Tensor::index_select(int64_t d, const Tensor& index) const {
  if (needs_grad && GradMode::is_enabled()) {
    py::gil_scoped_acquire gil;
    return from_python(
        to_python(*this).attr("index_select")(d, to_python(index)));
  }
  return stable::index_select(handle(), d, index.handle());
}
Tensor operator*(const Tensor& a, const Tensor& b) {
  if ((a.needs_grad || b.needs_grad) && GradMode::is_enabled()) {
    py::gil_scoped_acquire gil;
    return from_python(to_python(a).attr("__mul__")(to_python(b)));
  }
  std::array<StableIValue, 2> stack{
      stable::detail::from(a.handle()), stable::detail::from(b.handle())};
  STABLE_TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::mul", "Tensor", stack.data(), TORCH_ABI_VERSION));
  return stable::detail::to<stable::Tensor>(stack[0]);
}
}  // namespace tmol::native
PYBIND11_MODULE(_C, m) {
  tmol::native::bind_libraries(m);
#ifdef WITH_CUDA
  m.attr("has_cuda") = true;
#else
  m.attr("has_cuda") = false;
#endif
}
