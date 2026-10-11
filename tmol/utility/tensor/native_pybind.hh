#pragma once
#include "native.hh"
#ifdef TORCH_TARGET_VERSION
#include <pybind11/pybind11.h>
#include <pybind11/gil_safe_call_once.h>
#include <pybind11/stl.h>
#include <dlpack/dlpack.h>

namespace py = pybind11;
namespace tmol::native {
namespace py = pybind11;
inline std::shared_ptr<void> own_python(py::handle obj) {
  Py_INCREF(obj.ptr());
  return {obj.ptr(), [](void* p) {
            py::gil_scoped_acquire gil;
            Py_DECREF(static_cast<PyObject*>(p));
          }};
}
inline ScalarType scalar_type(DLDataType type) {
  if (type.lanes != 1)
    throw std::invalid_argument("TMol needs scalar DLPack elements");
  if (type.code == kDLFloat && type.bits == 32) return ScalarType::Float;
  if (type.code == kDLFloat && type.bits == 64) return ScalarType::Double;
  if (type.code == kDLInt && type.bits == 64) return ScalarType::Long;
  if (type.code == kDLInt && type.bits == 32) return ScalarType::Int;
  if (type.code == kDLInt && type.bits == 16) return ScalarType::Short;
  if (type.code == kDLInt && type.bits == 8) return ScalarType::Char;
  if (type.code == kDLUInt && type.bits == 8) return ScalarType::Byte;
  if (type.code == kDLBool && type.bits == 8) return ScalarType::Bool;
  throw std::invalid_argument("Unsupported TMol tensor dtype");
}
inline DLDataType dl_type(ScalarType type) {
  switch (type) {
    case ScalarType::Float:
      return {kDLFloat, 32, 1};
    case ScalarType::Double:
      return {kDLFloat, 64, 1};
    case ScalarType::Long:
      return {kDLInt, 64, 1};
    case ScalarType::Int:
      return {kDLInt, 32, 1};
    case ScalarType::Short:
      return {kDLInt, 16, 1};
    case ScalarType::Char:
      return {kDLInt, 8, 1};
    case ScalarType::Byte:
      return {kDLUInt, 8, 1};
    case ScalarType::Bool:
      return {kDLBool, 8, 1};
    default:
      throw std::invalid_argument("Unsupported TMol tensor dtype");
  }
}
struct TensorAPI {
  py::object to, from, is_neg, resolve_neg, library;
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
        auto dlpack = py::module_::import("torch.utils.dlpack");
        auto tensor = torch.attr("Tensor");
        auto ctypes = py::module_::import("ctypes");
        // These public stable C functions arrived in Torch 2.14. Resolve them
        // optionally so the same binary still works on 2.11 through DLPack.
        auto library = ctypes.attr("CDLL")(torch.attr("_C").attr("__file__"));
        TensorAPI api{
            dlpack.attr("to_dlpack"),
            dlpack.attr("from_dlpack"),
            tensor.attr("is_neg"),
            tensor.attr("resolve_neg"),
            library};
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
inline Tensor from_python(py::handle obj) {
  if (obj.is_none()) return {};
  const auto& api = tensor_api();
  auto input = py::reinterpret_borrow<py::object>(obj);
  if (py::cast<bool>(api.is_neg(input))) input = api.resolve_neg(input);
  Tensor result;
  if (api.import_tensor) {
    AtenTensorHandle handle = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(api.import_tensor(input.ptr(), &handle));
    result.value.emplace(handle);
  } else {
    auto capsule = api.to(input);
    auto* managed = static_cast<DLManagedTensor*>(
        PyCapsule_GetPointer(capsule.ptr(), "dltensor"));
    if (!managed) throw py::error_already_set();
    auto& dl = managed->dl_tensor;
    if (dl.device.device_type != kDLCPU && dl.device.device_type != kDLCUDA)
      throw std::invalid_argument("TMol supports CPU and CUDA tensors");
    std::vector<int64_t> contiguous;
    if (!dl.strides) {
      contiguous.assign(dl.ndim, 1);
      for (int d = dl.ndim - 2; d >= 0; --d)
        contiguous[d] = contiguous[d + 1] * dl.shape[d + 1];
    }
    auto dtype = scalar_type(dl.dtype);
    PyCapsule_SetName(capsule.ptr(), "used_dltensor");
    result.external = std::make_shared<ExternalTensor>(ExternalTensor{
        {managed,
         [](DLManagedTensor* value) {
           if (value->deleter) value->deleter(value);
         }},
        std::move(contiguous),
        dtype,
        Device(
            dl.device.device_type == kDLCPU ? kCPU : kCUDA,
            dl.device.device_type == kDLCPU ? -1 : dl.device.device_id)});
  }
  result.original = std::make_shared<std::shared_ptr<void>>(own_python(obj));
  result.needs_grad = py::cast<bool>(obj.attr(api.requires_grad));
  return result;
}
struct ExportedTensor {
  DLManagedTensor managed{};
  Tensor tensor;
  std::vector<int64_t> shape, strides;
  explicit ExportedTensor(Tensor value)
      : tensor(std::move(value)),
        shape(tensor.sizes().begin(), tensor.sizes().end()),
        strides(tensor.strides().begin(), tensor.strides().end()) {
    // The exported storage must not own its Python wrapper (a reference cycle).
    tensor.original.reset();
    auto device = tensor.device();
    managed.dl_tensor = {
        tensor.data_ptr(),
        {device.is_cpu() ? kDLCPU : kDLCUDA,
         device.is_cpu() ? 0 : device.index()},
        static_cast<int>(shape.size()),
        dl_type(tensor.scalar_type()),
        shape.data(),
        strides.data(),
        0};
    managed.manager_ctx = this;
    managed.deleter = [](DLManagedTensor* m) {
      delete static_cast<ExportedTensor*>(m->manager_ctx);
    };
  }
};
inline py::object to_python(Tensor tensor) {
  if (!tensor.defined()) return py::none();
  if (tensor.original && *tensor.original)
    return py::reinterpret_borrow<py::object>(
        static_cast<PyObject*>(tensor.original->get()));
  auto original = tensor.original;
  const auto& api = tensor_api();
  py::object result;
  if (api.export_tensor) {
    void* object = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(
        api.export_tensor(tensor.handle().get(), nullptr, &object));
    result = py::reinterpret_steal<py::object>(static_cast<PyObject*>(object));
  } else {
    auto* exported = new ExportedTensor(std::move(tensor));
    py::capsule capsule(&exported->managed, "dltensor", [](PyObject* obj) {
      if (PyCapsule_IsValid(obj, "dltensor")) {
        auto* m = static_cast<DLManagedTensor*>(
            PyCapsule_GetPointer(obj, "dltensor"));
        m->deleter(m);
      }
    });
    result = api.from(capsule);
  }
  *original = own_python(result);
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
