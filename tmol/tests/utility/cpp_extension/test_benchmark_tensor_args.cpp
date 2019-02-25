#include <pybind11/pybind11.h>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/tensor/pybind.h>

namespace tmol {
namespace tests {
namespace utility {
namespace cpp_extension {

struct vec_args {
  typedef Eigen::Vector3f Real3;

  static const tmol::Device D = tmol::Device::CPU;

  static auto v1(TView<Real3, 1, D> t1) -> void { return; }

  static auto v2(TView<Real3, 1, D> t1, TView<Real3, 1, D> t2) -> void {
    return;
  }

  static auto v4(
      TView<Real3, 1, D> t1,
      TView<Real3, 1, D> t2,
      TView<Real3, 1, D> t3,
      TView<Real3, 1, D> t4) -> void {
    return;
  }

  static auto v8(
      TView<Real3, 1, D> t1,
      TView<Real3, 1, D> t2,
      TView<Real3, 1, D> t3,
      TView<Real3, 1, D> t4,
      TView<Real3, 1, D> t5,
      TView<Real3, 1, D> t6,
      TView<Real3, 1, D> t7,
      TView<Real3, 1, D> t8) -> void {
    return;
  }

  static auto v16(
      TView<Real3, 1, D> t1,
      TView<Real3, 1, D> t2,
      TView<Real3, 1, D> t3,
      TView<Real3, 1, D> t4,
      TView<Real3, 1, D> t5,
      TView<Real3, 1, D> t6,
      TView<Real3, 1, D> t7,
      TView<Real3, 1, D> t8,
      TView<Real3, 1, D> t9,
      TView<Real3, 1, D> t10,
      TView<Real3, 1, D> t11,
      TView<Real3, 1, D> t12,
      TView<Real3, 1, D> t13,
      TView<Real3, 1, D> t14,
      TView<Real3, 1, D> t15,
      TView<Real3, 1, D> t16) -> void {
    return;
  }
};

struct scalar_args {
  static const tmol::Device D = tmol::Device::CPU;

  static auto v1(TView<float, 1, D> t1) -> void { return; }

  static auto v2(TView<float, 1, D> t1, TView<float, 1, D> t2) -> void {
    return;
  }

  static auto v4(
      TView<float, 1, D> t1,
      TView<float, 1, D> t2,
      TView<float, 1, D> t3,
      TView<float, 1, D> t4) -> void {
    return;
  }

  static auto v8(
      TView<float, 1, D> t1,
      TView<float, 1, D> t2,
      TView<float, 1, D> t3,
      TView<float, 1, D> t4,
      TView<float, 1, D> t5,
      TView<float, 1, D> t6,
      TView<float, 1, D> t7,
      TView<float, 1, D> t8) -> void {
    return;
  }

  static auto v16(
      TView<float, 1, D> t1,
      TView<float, 1, D> t2,
      TView<float, 1, D> t3,
      TView<float, 1, D> t4,
      TView<float, 1, D> t5,
      TView<float, 1, D> t6,
      TView<float, 1, D> t7,
      TView<float, 1, D> t8,
      TView<float, 1, D> t9,
      TView<float, 1, D> t10,
      TView<float, 1, D> t11,
      TView<float, 1, D> t12,
      TView<float, 1, D> t13,
      TView<float, 1, D> t14,
      TView<float, 1, D> t15,
      TView<float, 1, D> t16) -> void {
    return;
  }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;
  py::dict vec_funcs;
  m.add_object("vec_args", vec_funcs);

  m.def("_vec_args_v1", &vec_args::v1, "t0"_a);
  vec_funcs[py::cast(1)] = m.attr("_vec_args_v1");

  m.def("_vec_args_v2", &vec_args::v2, "t0"_a, "t1"_a);
  vec_funcs[py::cast(2)] = m.attr("_vec_args_v2");

  m.def("_vec_args_v4", &vec_args::v4, "t0"_a, "t1"_a, "t2"_a, "t3"_a);
  vec_funcs[py::cast(4)] = m.attr("_vec_args_v4");

  m.def(
      "_vec_args_v8",
      &vec_args::v8,
      "t0"_a,
      "t1"_a,
      "t2"_a,
      "t3"_a,
      "t4"_a,
      "t5"_a,
      "t6"_a,
      "t7"_a);
  vec_funcs[py::cast(8)] = m.attr("_vec_args_v8");

  m.def(
      "_vec_args_v16",
      &vec_args::v16,
      "t0"_a,
      "t1"_a,
      "t2"_a,
      "t3"_a,
      "t4"_a,
      "t5"_a,
      "t6"_a,
      "t7"_a,
      "t8"_a,
      "t9"_a,
      "t10"_a,
      "t11"_a,
      "t12"_a,
      "t13"_a,
      "t14"_a,
      "t15"_a);
  vec_funcs[py::cast(16)] = m.attr("_vec_args_v16");

  py::dict scalar_funcs;
  m.add_object("scalar_args", scalar_funcs);

  m.def("_scalar_args_v1", &scalar_args::v1, "t0"_a);
  scalar_funcs[py::cast(1)] = m.attr("_scalar_args_v1");

  m.def("_scalar_args_v2", &scalar_args::v2, "t0"_a, "t1"_a);
  scalar_funcs[py::cast(2)] = m.attr("_scalar_args_v2");

  m.def("_scalar_args_v4", &scalar_args::v4, "t0"_a, "t1"_a, "t2"_a, "t3"_a);
  scalar_funcs[py::cast(4)] = m.attr("_scalar_args_v4");

  m.def(
      "_scalar_args_v8",
      &scalar_args::v8,
      "t0"_a,
      "t1"_a,
      "t2"_a,
      "t3"_a,
      "t4"_a,
      "t5"_a,
      "t6"_a,
      "t7"_a);
  scalar_funcs[py::cast(8)] = m.attr("_scalar_args_v8");

  m.def(
      "_scalar_args_v16",
      &scalar_args::v16,
      "t0"_a,
      "t1"_a,
      "t2"_a,
      "t3"_a,
      "t4"_a,
      "t5"_a,
      "t6"_a,
      "t7"_a,
      "t8"_a,
      "t9"_a,
      "t10"_a,
      "t11"_a,
      "t12"_a,
      "t13"_a,
      "t14"_a,
      "t15"_a);
  scalar_funcs[py::cast(16)] = m.attr("_scalar_args_v16");
}

}  // namespace cpp_extension
}  // namespace utility
}  // namespace tests
}  // namespace tmol
