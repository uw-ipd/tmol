#pragma once
#include "native_pybind.hh"
#ifdef TORCH_TARGET_VERSION
namespace tmol::native::autograd {
using tensor_list = std::vector<Tensor>;
struct ScoreCapture {
  ScoreCapture* previous = nullptr;
  tensor_list derivatives;
  std::vector<int64_t> offsets;
  int64_t offset = 0;
  bool enabled = true;
};
inline thread_local ScoreCapture* score_capture = nullptr;
template <class F, class = void>
struct Capturable : std::false_type {};
template <class F>
struct Capturable<F, std::void_t<decltype(F::capture_pose_gradient)>>
    : std::true_type {};

inline void attach_autograd_result(Tensor& tensor, py::handle result) {
  if (!tensor.defined()) return;
  tensor.original.reset();
  if (tensor_api().import_tensor) {
    AtenTensorHandle handle = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(
        tensor_api().import_tensor(result.ptr(), &handle));
    tensor.value.emplace(handle);
  } else {
    tensor.value = import_tensor_python(result);
  }
  tensor.cache_metadata();
  tensor.needs_grad = py::cast<bool>(result.attr(tensor_api().requires_grad));
}
inline void attach_autograd_result(tensor_list& tensors, py::handle results) {
  auto sequence = py::reinterpret_borrow<py::sequence>(results);
  for (size_t i = 0; i < tensors.size(); ++i)
    attach_autograd_result(tensors[i], sequence[i]);
}
// Borrowed handles are valid for the synchronous Python forward/backward call.
// Only context access needs the GIL; kernels and stable C API calls run without
// it.
struct SavedValue {
  py::handle data;
  const char* key;
  template <class T>
  void operator=(T value) {
    if (!data) return;
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
  tensor_list* captured = nullptr;
  SavedData saved_data;
  AutogradContext() : saved_data{} {}
  explicit AutogradContext(py::handle c)
      : context(c), saved_data{c.attr("_tmol_saved_data")} {}
  void save_for_backward(tensor_list tensors) {
    if (captured) {
      *captured = std::move(tensors);
      return;
    }
    if (!context) return;
    py::gil_scoped_acquire gil;
    context.attr("save_for_backward")(*to_python_tensors(tensors));
  }
  tensor_list get_saved_variables() {
    py::gil_scoped_acquire gil;
    return from_python_tensors(context.attr("saved_tensors"));
  }
  void set_materialize_grads(bool value) {
    if (!context) return;
    py::gil_scoped_acquire gil;
    context.attr("set_materialize_grads")(value);
  }
  void mark_non_differentiable(tensor_list tensors) {
    if (!context) return;
    py::gil_scoped_acquire gil;
    context.attr("mark_non_differentiable")(*py::tuple(py::cast(tensors)));
  }
};
template <class F>
struct Function {
  template <class... Args>
  static auto apply(const Args&... args) {
    using Result =
        decltype(F::forward(static_cast<AutogradContext*>(nullptr), args...));
    struct Invocation {
      std::tuple<const Args&...> arguments;
      Result result;
    };
    auto differentiable = [](const auto& arg) {
      if constexpr (std::is_same_v<std::decay_t<decltype(arg)>, Tensor>)
        return arg.requires_grad();
      else
        return false;
    };
    if (!GradMode::is_enabled() || !(differentiable(args) || ...)) {
      AutogradContext ctx;
      return F::forward(&ctx, args...);
    }
    if constexpr (Capturable<F>::value) {
      if (score_capture && score_capture->enabled) {
        tensor_list saved;
        AutogradContext ctx;
        ctx.captured = &saved;
        struct NoGrad {
          NoGrad() { aoti_torch_grad_mode_set_enabled(false); }
          ~NoGrad() { aoti_torch_grad_mode_set_enabled(true); }
        } no_grad;
        auto result = F::forward(&ctx, args...);
        STD_TORCH_CHECK(saved.size() == 1, "Expected whole-pose derivatives");
        score_capture->derivatives.push_back(std::move(saved[0]));
        score_capture->offsets.push_back(score_capture->offset);
        return result;
      }
    }
    py::gil_scoped_acquire gil;
    PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object>
        storage;
    auto& apply =
        storage
            .call_once_and_store_result([] {
              auto support = py::module_::import("tmol._native_autograd");
              auto forward = py::cpp_function([](py::object context,
                                                 py::capsule arguments,
                                                 py::object indices,
                                                 py::args) {
                context.attr("_tmol_gradient_indices") = indices;
                context.attr("_tmol_saved_data") = py::dict();
                AutogradContext ctx(context);
                auto& invocation =
                    *static_cast<Invocation*>(arguments.get_pointer());
                {
                  py::gil_scoped_release release;
                  invocation.result = std::apply(
                      [&](const auto&... values) {
                        return F::forward(&ctx, values...);
                      },
                      invocation.arguments);
                }
                if constexpr (std::is_same_v<Result, tensor_list>)
                  return py::object(to_python_tensors(invocation.result));
                else
                  return py::cast(invocation.result);
              });
              auto backward =
                  py::cpp_function([](py::object context, py::args args) {
                    AutogradContext ctx(context);
                    auto grads = from_python_tensors(args);
                    tensor_list gradients;
                    {
                      py::gil_scoped_release release;
                      gradients = F::backward(&ctx, grads);
                    }
                    auto indices = context.attr("_tmol_gradient_indices");
                    py::tuple result(2 + py::len(indices));
                    result[0] = py::none();
                    result[1] = py::none();
                    size_t i = 2;
                    for (auto index : indices)
                      result[i++] =
                          to_python(gradients.at(py::cast<size_t>(index)));
                    return result;
                  });
              return support.attr("_bind")(forward, backward);
            })
            .get_stored();
    // Only differentiable tensors need autograd edges. Static topology and
    // parameter tensors stay in the native argument tuple; save_for_backward
    // still gives PyTorch ownership and version checks for saved tensors.
    py::list inputs;
    std::vector<size_t> indices;
    size_t position = 0;
    auto add_input = [&](const auto& arg) {
      if constexpr (std::is_same_v<std::decay_t<decltype(arg)>, Tensor>) {
        if (arg.requires_grad()) {
          inputs.append(to_python(arg));
          indices.push_back(position);
        }
      }
      ++position;
    };
    if (GradMode::is_enabled()) (add_input(args), ...);
    Invocation invocation{std::tie(args...), {}};
    auto result = apply(
        py::capsule(&invocation, "tmol.native_arguments"),
        py::cast(indices),
        *py::tuple(inputs));
    // Autograd may wrap aliased outputs. Keep its returned handles so both
    // dispatcher boxing and subsequent native calls retain the gradient edges.
    attach_autograd_result(invocation.result, result);
    return std::move(invocation.result);
  }
};
}  // namespace tmol::native::autograd

#endif
