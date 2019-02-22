#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <torch/csrc/utils/pybind.h>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorUtil.h>

namespace pybind11 {
namespace detail {

template <tmol::Device D>
struct device_name {};

template <>
struct device_name<tmol::Device::CPU> {
  static constexpr auto name = _("\'cpu\'");
};

template <>
struct device_name<tmol::Device::CUDA> {
  static constexpr auto name = _("\'cuda\'");
};

template <typename T, int N>
struct npy_format_descriptor_name<Eigen::Matrix<T, N, 1>> {
  static constexpr auto name =
      npy_format_descriptor_name<T>::name + _("[") + _<N>() + _("]");
};

template <typename T, int N>
struct npy_format_descriptor_name<Eigen::AlignedBox<T, N>> {
  static constexpr auto name = _("AlignedBox(")
                               + npy_format_descriptor_name<T>::name + _("[")
                               + _<N * 2>() + _("])");
};

template <typename T, size_t N, tmol::Device D, tmol::PtrTag P>
struct handle_type_name<tmol::TView<T, N, D, P>> {
  static constexpr auto name =
      _("torch.Tensor[") + npy_format_descriptor_name<T>::name + _(", ")
      + _<N>() + _(", ") + device_name<D>::name + _("]");
};

template <typename T, size_t N, tmol::Device D, tmol::PtrTag P>
struct handle_type_name<tmol::TPack<T, N, D, P>> {
  static constexpr auto name =
      _("torch.Tensor[") + npy_format_descriptor_name<T>::name + _(", ")
      + _<N>() + _(", ") + device_name<D>::name + _("]");
};

template <typename T, size_t N, tmol::Device D, tmol::PtrTag P>
struct handle_type_name<tmol::TViewCollection<T, N, D, P>> {
  static constexpr auto name =
      _("torch.TensorCollection[") + npy_format_descriptor_name<T>::name
      + _(", ") + _<N>() + _(", ") + device_name<D>::name + _("]");
};

template <typename T, size_t N, tmol::Device D, tmol::PtrTag P>
struct type_caster<tmol::TView<T, N, D, P>> {
 public:
  typedef tmol::TView<T, N, D, P> ViewType;
  PYBIND11_TYPE_CASTER(ViewType, handle_type_name<ViewType>::name);

  bool load(handle src, bool convert) {
    using pybind11::print;

    type_caster<at::Tensor> conv;

    if (!conv.load(src, convert)) {
#ifdef DEBUG
      print("Error casting to tensor: ", src);
#endif
      return false;
    }

    try {
      value = tmol::view_tensor<T, N, D, P>(conv);
      return true;
    } catch (at::Error err) {
#ifdef DEBUG
      print(
          "Error casting to TView type: ",
          type_id<ViewType>(),
          " value: ",
          src);
#endif
      return false;
    }
  }

  // C++ -> Python cast operation not supported.
};

template <typename T, size_t N, tmol::Device D, tmol::PtrTag P>
struct type_caster<tmol::TPack<T, N, D, P>> {
 public:
  typedef tmol::TPack<T, N, D, P> PackType;
  PYBIND11_TYPE_CASTER(PackType, handle_type_name<PackType>::name);

  bool load(handle src, bool convert) {
    using pybind11::print;

    type_caster<at::Tensor> conv;

    if (!conv.load(src, convert)) {
#ifdef DEBUG
      print("Error casting to tensor: ", src);
#endif
      return false;
    }

    try {
      value = tmol::TPack<T, N, D, P>(conv);
      return true;
    } catch (at::Error err) {
#ifdef DEBUG
      print(
          "Error casting to TPack type: ",
          type_id<PackType>(),
          " value: ",
          src);
#endif
      return false;
    }
  }

  static handle cast(PackType src, return_value_policy policy, handle parent) {
    return type_caster<at::Tensor>::cast(src.tensor, policy, parent);
  }

  // C++ -> Python cast operation not supported.
};

template <typename T, size_t N, tmol::Device D, tmol::PtrTag P>
struct type_caster<tmol::TViewCollection<T, N, D, P>> {
 public:
  typedef tmol::TViewCollection<T, N, D, P> ViewCollType;
  PYBIND11_TYPE_CASTER(ViewCollType, handle_type_name<ViewCollType>::name);

  bool load(handle src, bool convert) {
    PyObject* source = src.ptr();
    auto tuple = PyTuple_Check(source);
    if (tuple || PyList_Check(source)) {
      auto size = tuple ? PyTuple_GET_SIZE(source) : PyList_GET_SIZE(source);
      value_v.resize(size);

      for (int i = 0; i < size; ++i) {
        PyObject* obj =
            tuple ? PyTuple_GET_ITEM(source, i) : PyList_GET_ITEM(source, i);
        type_caster<tmol::TView<T, N, D, P>> conv;
        if (conv.load(obj, convert)) {
          value_v[i] = conv;
        } else {
          return false;
        }
      }
      value = value_v;
      return true;
    }
  }

 private:
  tmol::TViewCollection<T, N, D, P> value_v;
};

}  // namespace detail
}  // namespace pybind11
