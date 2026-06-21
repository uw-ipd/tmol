#pragma once

#include <tmol/utility/tensor/pybind.h>
#include <torch/extension.h>

namespace tmol {
namespace utility {
namespace function_dispatch {

template <typename Scalar>
struct TypeInfo {};

#define FORALL_DTYPES_EXCEPT_HALF(_) \
  _(uint8_t, uint8)                  \
  _(int8_t, int8)                    \
  _(int16_t, int16)                  \
  _(int, int32)                      \
  _(int64_t, int64)                  \
  _(float, float32)                  \
  _(double, float64)

#define TYPE_INFO(CTYPE, NAME)                                              \
  template <>                                                               \
  struct TypeInfo<CTYPE> {                                                  \
    static std::string name() { return #NAME; }                             \
    static auto dtype() { return py::module::import("torch").attr(#NAME); } \
  };

FORALL_DTYPES_EXCEPT_HALF(TYPE_INFO)
#undef TYPE_INFO

template <tmol::Device Dev>
struct DevInfo {};

template <>
struct DevInfo<tmol::Device::CPU> {
  static std::string name() { return "cpu"; }
  static auto type() { return "cpu"; }
};

template <>
struct DevInfo<tmol::Device::CUDA> {
  static std::string name() { return "cuda"; }
  static auto type() { return "cuda"; }
};

template <tmol::Device Dev, typename Scalar, typename Func, typename... Extra>
auto add_dispatch_impl(
    pybind11::module& m,
    std::string dispatch_name,
    Func&& f,
    const Extra&... extra) {
  if (!py::hasattr(m, dispatch_name.c_str())) {
    py::dict dispatch;
    m.add_object(dispatch_name.c_str(), dispatch);
  }

  py::dict dispatch = py::getattr(m, dispatch_name.c_str());

  std::string fname = "_" + dispatch_name + "_" + DevInfo<Dev>::name() + "_"
                      + TypeInfo<Scalar>::name();

  m.def(fname.c_str(), f, extra...);

  dispatch[py::make_tuple(DevInfo<Dev>::type(), TypeInfo<Scalar>::dtype())] =
      m.attr(fname.c_str());
}

}  // namespace function_dispatch
}  // namespace utility
}  // namespace tmol
