#pragma once
#include "native_autograd.hh"
#ifdef TORCH_TARGET_VERSION
#include <functional>
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
struct Library {
  py::module_ module;
  void def(const char* schema) {
    module.attr("_schemas").cast<py::list>().append(schema);
  }
  template <class Result, class... Args>
  void def(const char* schema, Result (*fn)(Args...)) {
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
    module.def(name.c_str(), fn, py::call_guard<py::gil_scoped_release>());
    module.attr("_schemas").cast<py::list>().append(signature);
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
  for (auto [name, fn] : registrations()) {
    auto submodule = m.def_submodule(name);
    submodule.attr("_schemas") = py::list();
    Library library{submodule};
    fn(library);
  }
}
}  // namespace tmol::native
#define TMOL_LIBRARY(NAME, VAR)                                      \
  static void bind_##NAME(tmol::native::Library&);                   \
  static tmol::native::Register register_##NAME(#NAME, bind_##NAME); \
  static void bind_##NAME(tmol::native::Library& VAR)

#else
#define TMOL_LIBRARY TORCH_LIBRARY
#endif
