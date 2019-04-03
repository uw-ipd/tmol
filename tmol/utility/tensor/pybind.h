#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <torch/csrc/utils/pybind.h>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorCollection.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

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

template <typename T, int M, int N>
struct npy_format_descriptor_name<Eigen::Matrix<T, M, N>> {
  static constexpr auto name = npy_format_descriptor_name<T>::name + _("[")
                               + _<M>() + _(", ") + _<N>() + _("]");
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
struct handle_type_name<tmol::TCollection<T, N, D, P>> {
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
    nvtx_range_function();

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
      print("err.what()");
      print(err.what());

      print(
          "Error casting to TView:",
          "\ntype:\n",
          type_id<ViewType>(),
          "\nvalue:\n",
          src,
          "\nerror:\n",
          err.what_without_backtrace());
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
    nvtx_range_function();

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
      print("err.what()");
      print(err.what());
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
    nvtx_range_function();

    return type_caster<at::Tensor>::cast(src.tensor, policy, parent);
  }

  // C++ -> Python cast operation not supported.
};

template <typename T, size_t N, tmol::Device D, tmol::PtrTag P>
struct type_caster<tmol::TCollection<T, N, D, P>> {
 public:
  typedef tmol::TCollection<T, N, D, P> ViewCollType;
  PYBIND11_TYPE_CASTER(ViewCollType, handle_type_name<ViewCollType>::name);

  bool load(handle src, bool convert) {
    type_caster<std::vector<at::Tensor>> conv;

    if (conv.load(src, convert)) {
      value = tmol::TCollection<T, N, D, P>(conv);
      return true;
    }
    return false;
  }
};

}  // namespace detail
}  // namespace pybind11
