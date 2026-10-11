#pragma once
#include "native.hh"
#ifdef TORCH_TARGET_VERSION
#include <pybind11/pybind11.h>
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
inline Tensor from_python(py::handle obj) {
  if (obj.is_none()) return {};
  auto capsule =
      py::module_::import("torch.utils.dlpack")
          .attr("to_dlpack")(obj.attr("resolve_conj")().attr("resolve_neg")());
  auto* managed = static_cast<DLManagedTensor*>(
      PyCapsule_GetPointer(capsule.ptr(), "dltensor"));
  if (!managed) throw py::error_already_set();
  auto& dl = managed->dl_tensor;
  if (dl.device.device_type != kDLCPU && dl.device.device_type != kDLCUDA)
    throw std::invalid_argument("TMol supports CPU and CUDA tensors");
  std::vector<int64_t> contiguous(dl.ndim, 1);
  if (!dl.strides)
    for (int d = dl.ndim - 2; d >= 0; --d)
      contiguous[d] = contiguous[d + 1] * dl.shape[d + 1];
  auto dtype = scalar_type(dl.dtype);
  PyCapsule_SetName(capsule.ptr(), "used_dltensor");
  Tensor result(
      stable::from_blob(
          static_cast<char*>(dl.data) + dl.byte_offset,
          IntList(dl.shape, dl.ndim),
          IntList(dl.strides ? dl.strides : contiguous.data(), dl.ndim),
          Device(
              dl.device.device_type == kDLCPU ? kCPU : kCUDA,
              dl.device.device_type == kDLCPU ? -1 : dl.device.device_id),
          dtype,
          [managed](void*) {
            if (managed->deleter) managed->deleter(managed);
          }));
  *result.original = own_python(obj);
  result.needs_grad = py::cast<bool>(obj.attr("requires_grad"));
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
    tensor.original = std::make_shared<std::shared_ptr<void>>();
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
  if (*tensor.original)
    return py::reinterpret_borrow<py::object>(
        static_cast<PyObject*>(tensor.original->get()));
  auto original = tensor.original;
  auto* exported = new ExportedTensor(std::move(tensor));
  py::capsule capsule(&exported->managed, "dltensor", [](PyObject* obj) {
    if (PyCapsule_IsValid(obj, "dltensor")) {
      auto* m =
          static_cast<DLManagedTensor*>(PyCapsule_GetPointer(obj, "dltensor"));
      m->deleter(m);
    }
  });
  auto result =
      py::module_::import("torch.utils.dlpack").attr("from_dlpack")(capsule);
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
