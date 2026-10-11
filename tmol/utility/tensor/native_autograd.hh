#pragma once
#include "native_pybind.hh"
#ifdef TORCH_TARGET_VERSION
namespace tmol::native::autograd {
using tensor_list = std::vector<Tensor>;
// Borrowed handles are valid for the synchronous Python forward/backward call.
// Only context access needs the GIL; kernels and stable C API calls run without
// it.
struct SavedValue {
  py::handle data;
  const char* key;
  template <class T>
  void operator=(T value) {
    py::gil_scoped_acquire gil;
    data[py::str(key)] = py::cast(value);
  }
  int64_t toInt() const {
    py::gil_scoped_acquire gil;
    return py::cast<int64_t>(data[py::str(key)]);
  }
  bool toBool() const {
    py::gil_scoped_acquire gil;
    return py::cast<bool>(data[py::str(key)]);
  }
};
struct SavedData {
  py::handle data;
  SavedValue operator[](const char* key) { return {data, key}; }
};
struct AutogradContext {
  py::handle context;
  SavedData saved_data;
  explicit AutogradContext(py::handle c)
      : context(c), saved_data{c.attr("_tmol_saved_data")} {}
  void save_for_backward(tensor_list tensors) {
    py::gil_scoped_acquire gil;
    context.attr("save_for_backward")(*py::tuple(py::cast(tensors)));
  }
  tensor_list get_saved_variables() {
    py::gil_scoped_acquire gil;
    return py::cast<tensor_list>(context.attr("saved_tensors"));
  }
  void set_materialize_grads(bool value) {
    py::gil_scoped_acquire gil;
    context.attr("set_materialize_grads")(value);
  }
  void mark_non_differentiable(tensor_list tensors) {
    py::gil_scoped_acquire gil;
    context.attr("mark_non_differentiable")(*py::tuple(py::cast(tensors)));
  }
};
template <class F>
struct Function {
  template <class... Args>
  static auto apply(Args... args) {
    py::gil_scoped_acquire gil;
    using Result =
        decltype(F::forward(static_cast<AutogradContext*>(nullptr), args...));
    PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object>
        storage;
    auto& apply =
        storage
            .call_once_and_store_result([] {
              auto support = py::module_::import("tmol._native_autograd");
              auto forward = py::cpp_function([](py::object context,
                                                 py::capsule arguments) {
                AutogradContext ctx(context);
                auto& values = *static_cast<std::tuple<std::decay_t<Args>...>*>(
                    arguments.get_pointer());
                py::gil_scoped_release release;
                return std::apply(
                    [&](auto&... values) {
                      return F::forward(&ctx, values...);
                    },
                    values);
              });
              auto backward =
                  py::cpp_function([](py::object context, tensor_list grads) {
                    AutogradContext ctx(context);
                    py::gil_scoped_release release;
                    return F::backward(&ctx, grads);
                  });
              return support.attr("_bind")(forward, backward);
            })
            .get_stored();
    // Function.apply needs Python tensors to attach autograd edges. Reuse the
    // already imported native arguments during its synchronous forward call.
    auto values = std::make_tuple(args...);
    return py::cast<Result>(
        apply(py::capsule(&values, "tmol.native_arguments"), args...));
  }
};
}  // namespace tmol::native::autograd

#endif
