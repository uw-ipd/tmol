#pragma once
#include "native.hh"
#ifdef TORCH_TARGET_VERSION
#include <pybind11/pybind11.h>
#include <pybind11/gil_safe_call_once.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace tmol::native {
namespace py = pybind11;
inline py::object call_one(py::handle function, py::handle argument) {
  auto* result = PyObject_CallOneArg(function.ptr(), argument.ptr());
  if (!result) throw py::error_already_set();
  return py::reinterpret_steal<py::object>(result);
}
inline void own_python(PythonOwner& owner, py::handle obj) {
  Py_INCREF(obj.ptr());
  owner.object = obj.ptr();
  owner.release = [](void* p) {
    py::gil_scoped_acquire gil;
    Py_DECREF(static_cast<PyObject*>(p));
  };
}
struct TensorAPI {
  py::object is_neg, resolve_neg, library;
  py::str requires_grad{"requires_grad"};
  using Import = AOTITorchError (*)(void*, AtenTensorHandle*);
  using Export = AOTITorchError (*)(AtenTensorHandle, void*, void**);
  Import import_tensor = nullptr;
  Export export_tensor = nullptr;
};
inline const TensorAPI& tensor_api() {
  PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<TensorAPI> storage;
  return storage
      .call_once_and_store_result([] {
        auto torch = py::module_::import("torch");
        auto tensor = torch.attr("Tensor");
        auto ctypes = py::module_::import("ctypes");
        // These public stable C functions arrived in Torch 2.14. Resolve them
        // optionally; older runtimes exchange tensors through a stable
        // operator.
        auto library = ctypes.attr("CDLL")(torch.attr("_C").attr("__file__"));
        TensorAPI api{
            tensor.attr("is_neg"), tensor.attr("resolve_neg"), library};
        if (py::hasattr(library, "torch_tensor_from_pyobject")
            && py::hasattr(library, "torch_tensor_to_pyobject")) {
          auto address = [&](const char* name) {
            return py::cast<uintptr_t>(
                ctypes.attr("cast")(library.attr(name), ctypes.attr("c_void_p"))
                    .attr("value"));
          };
          api.import_tensor = reinterpret_cast<TensorAPI::Import>(
              address("torch_tensor_from_pyobject"));
          api.export_tensor = reinterpret_cast<TensorAPI::Export>(
              address("torch_tensor_to_pyobject"));
        }
        return api;
      })
      .get_stored();
}
std::vector<Tensor> from_python_tensors(py::handle objects);
py::tuple to_python_tensors(const std::vector<Tensor>& tensors);
stable::Tensor import_tensor_python(py::handle obj);
py::object export_tensor_python(const stable::Tensor& tensor);
inline Tensor from_python(py::handle obj) {
  if (obj.is_none()) return {};
  const auto& api = tensor_api();
  auto input = py::reinterpret_borrow<py::object>(obj);
  bool negative = py::cast<bool>(call_one(api.is_neg, input));
  if (negative) input = call_one(api.resolve_neg, input);
  Tensor result;
  if (api.import_tensor) {
    AtenTensorHandle handle = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(api.import_tensor(input.ptr(), &handle));
    result.value.emplace(handle);
  } else {
    result.value = import_tensor_python(input);
  }
  result.cache_metadata();
  if (negative) {
    result.original = std::make_shared<PythonOwner>();
    own_python(*result.original, obj);
  }
  result.needs_grad = GradMode::is_enabled()
                      && (result.scalar_type() == ScalarType::Float
                          || result.scalar_type() == ScalarType::Double)
                      && py::cast<bool>(obj.attr(api.requires_grad));
  return result;
}
inline py::object to_python(const Tensor& tensor) {
  if (!tensor.defined()) return py::none();
  if (tensor.original && tensor.original->object)
    return py::reinterpret_borrow<py::object>(
        static_cast<PyObject*>(tensor.original->object));
  const auto& api = tensor_api();
  py::object result;
  if (api.export_tensor) {
    void* object = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(
        api.export_tensor(tensor.handle().get(), nullptr, &object));
    result = py::reinterpret_steal<py::object>(static_cast<PyObject*>(object));
  } else {
    result = export_tensor_python(tensor.handle());
  }
  return result;
}
}  // namespace tmol::native
namespace pybind11::detail {
template <>
struct type_caster<tmol::native::Tensor> {
  PYBIND11_TYPE_CASTER(tmol::native::Tensor, const_name("torch.Tensor"));
  bool load(handle obj, bool) {
    value = tmol::native::from_python(obj);
    return true;
  }
  static handle cast(tmol::native::Tensor value, return_value_policy, handle) {
    return tmol::native::to_python(std::move(value)).release();
  }
};
}  // namespace pybind11::detail

#else
#include <torch/csrc/utils/pybind.h>
namespace py = pybind11;
#endif
