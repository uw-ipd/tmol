#pragma once
#include "native_autograd.hh"
#ifdef TORCH_TARGET_VERSION
#include <functional>
#include <torch/csrc/stable/library.h>
namespace tmol::native {
template <class T>
std::string schema_type();
template <class... T>
std::string tuple_schema(std::tuple<T...>*) {
  std::string result = "(";
  size_t n = 0;
  ((result += (n++ ? ", " : "") + schema_type<T>()), ...);
  return result + ")";
}
template <class T>
std::string schema_type() {
  if constexpr (std::is_same_v<T, Tensor>)
    return "Tensor";
  else if constexpr (std::is_same_v<T, std::vector<Tensor>>)
    return "Tensor[]";
  else if constexpr (std::is_same_v<T, std::vector<int64_t>>)
    return "int[]";
  else if constexpr (std::is_same_v<T, bool>)
    return "bool";
  else if constexpr (std::is_integral_v<T>)
    return "int";
  else if constexpr (std::is_floating_point_v<T>)
    return "float";
  else if constexpr (std::is_void_v<T>)
    return "()";
  else
    return tuple_schema(static_cast<T*>(nullptr));
}
// Box directly at the stable dispatcher boundary, without a Python kernel.
template <class T>
T unbox(StableIValue value) {
  if constexpr (std::is_same_v<T, Tensor>)
    return Tensor(stable::detail::to<stable::Tensor>(value));
  else if constexpr (std::is_same_v<T, std::vector<Tensor>>) {
    auto inputs = stable::detail::to<std::vector<stable::Tensor>>(value);
    std::vector<Tensor> result;
    result.reserve(inputs.size());
    for (auto& input : inputs) result.emplace_back(std::move(input));
    return result;
  } else
    return stable::detail::to<T>(value);
}
inline StableIValue box(const Tensor& value) {
  return stable::detail::from(
      value.defined() ? value.handle() : stable::Tensor());
}
inline StableIValue box(const std::vector<Tensor>& values) {
  std::vector<stable::Tensor> tensors;
  tensors.reserve(values.size());
  for (auto& value : values)
    tensors.push_back(value.defined() ? value.handle() : stable::Tensor());
  return stable::detail::from(tensors);
}
template <class T>
void box_result(StableIValue* stack, const T& value) {
  stack[0] = box(value);
}
template <class... T>
void box_result(StableIValue* stack, const std::tuple<T...>& values) {
  size_t i = 0;
  std::apply(
      [&](const auto&... value) { ((stack[i++] = box(value)), ...); }, values);
}
template <auto fn, class Result, class... Args, size_t... I>
void invoke_boxed(
    StableIValue* stack, Result (*)(Args...), std::index_sequence<I...>) {
  auto unpack = [&] {
    return std::tuple<std::decay_t<Args>...>{
        unbox<std::decay_t<Args>>(stack[I])...};
  };
  auto args = [&] {
    // Whole-pose accumulation checks that every static parameter is frozen.
    // Its direct term calls always pass coordinates first, so no per-tensor
    // Python export or GIL acquisition is needed to discover gradient edges.
    if (GradMode::is_enabled() && autograd::score_capture
        && autograd::score_capture->enabled) {
      auto values = unpack();
      if constexpr (sizeof...(Args) > 0) {
        auto& first = std::get<0>(values);
        if constexpr (std::is_same_v<std::decay_t<decltype(first)>, Tensor>)
          first.needs_grad = true;
      }
      return values;
    }
    if (GradMode::is_enabled()) {
      py::gil_scoped_acquire gil;
      auto values = unpack();
      if (GradMode::is_enabled()) {
        std::vector<Tensor*> targets;
        std::vector<Tensor> tensors;
        auto collect = [&](auto& value) {
          if constexpr (std::is_same_v<std::decay_t<decltype(value)>, Tensor>) {
            if (value.defined()
                && (value.scalar_type() == ScalarType::Float
                    || value.scalar_type() == ScalarType::Double)) {
              if (tensor_api().export_tensor) {
                value.needs_grad = py::cast<bool>(
                    to_python(value).attr(tensor_api().requires_grad));
              } else {
                targets.push_back(&value);
                tensors.push_back(value);
              }
            }
          }
        };
        std::apply([&](auto&... value) { (collect(value), ...); }, values);
        if (!targets.empty()) {
          auto objects = to_python_tensors(tensors);
          for (size_t j = 0; j < targets.size(); ++j) {
            targets[j]->original = tensors[j].original;
            targets[j]->needs_grad =
                py::cast<bool>(objects[j].attr(tensor_api().requires_grad));
          }
        }
      }
      return values;
    }
    return unpack();
  }();
  if constexpr (std::is_void_v<Result>)
    std::apply(fn, args);
  else
    box_result(stack, std::apply(fn, args));
}
struct Library {
  stable::detail::StableLibrary definitions, implementations;
  std::vector<std::string> operators;
  explicit Library(const char* name)
      : definitions(
            stable::detail::StableLibrary::Kind::FRAGMENT,
            name,
            nullptr,
            __FILE__,
            __LINE__),
        implementations(
            stable::detail::StableLibrary::Kind::IMPL,
            name,
            "CompositeExplicitAutograd",
            __FILE__,
            __LINE__) {}
  void def(const char* schema) { definitions.def(schema); }
  template <class Result, class... Args, Result (*fn)(Args...)>
  void def(
      const char* schema, std::integral_constant<Result (*)(Args...), fn>) {
    std::string name(schema), signature(schema);
    auto paren = name.find('(');
    if (paren != std::string::npos)
      name.resize(paren);
    else {
      signature += "(";
      size_t n = 0;
      ((signature += (n ? ", " : "") + schema_type<std::decay_t<Args>>() + " _"
                     + std::to_string(n),
        ++n),
       ...);
      signature += ") -> " + schema_type<Result>();
    }
    definitions.def(signature.c_str());
    operators.push_back(name);
    implementations.impl(
        name.c_str(), [](StableIValue* stack, uint64_t, uint64_t) {
          invoke_boxed<fn>(stack, fn, std::index_sequence_for<Args...>{});
        });
  }
};
using Registration = std::pair<const char*, void (*)(Library&)>;
inline auto& registrations() {
  static std::vector<Registration> registry;
  return registry;
}
struct Register {
  Register(const char* name, void (*fn)(Library&)) {
    registrations().emplace_back(name, fn);
  }
};
inline void bind_libraries(py::module_& m) {
  // Registered kernels must outlive every invocation.
  static std::vector<std::unique_ptr<Library>> libraries;
  py::dict operators;
  for (auto [name, fn] : registrations()) {
    auto library = std::make_unique<Library>(name);
    fn(*library);
    operators[py::str(name)] = py::cast(library->operators);
    libraries.push_back(std::move(library));
  }
  m.attr("_operators") = operators;
}
}  // namespace tmol::native
#define TMOL_FN(...) \
  std::integral_constant<decltype(__VA_ARGS__), __VA_ARGS__> {}
#define TMOL_LIBRARY(NAME, VAR)                                      \
  static void bind_##NAME(tmol::native::Library&);                   \
  static tmol::native::Register register_##NAME(#NAME, bind_##NAME); \
  static void bind_##NAME(tmol::native::Library& VAR)
#else
#define TMOL_FN(...) __VA_ARGS__
#define TMOL_LIBRARY TORCH_LIBRARY
#endif
