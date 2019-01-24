#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <torch/csrc/utils/pybind.h>

#include <tmol/utility/tensor/TensorAccessor.h>
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
  static constexpr auto name = _("Vec(") + npy_format_descriptor_name<T>::name
                               + _(", ") + _<N>() + _(")");
};

template <typename T, int N>
struct npy_format_descriptor_name<Eigen::AlignedBox<T, N>> {
  static constexpr auto name = _("AlignedBox(")
                               + npy_format_descriptor_name<T>::name + _(", ")
                               + _<N * 2>() + _(")");
};

template <typename T, size_t N, tmol::Device D, tmol::PtrTag P>
struct handle_type_name<tmol::TView<T, N, D, P>> {
  static constexpr auto name =
      _("torch.Tensor[") + npy_format_descriptor_name<T>::name + _(", ")
      + _<N>() + _(", ") + device_name<D>::name + _("]");
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
      print("Error casting to tensor: ", src);
      return false;
    }

    try {
      value = tmol::view_tensor<T, N, D, P>(conv);
      return true;
    } catch (at::Error err) {
      print("Error casting to type: ", type_id<ViewType>(), " value: ", src);
      return false;
    }
  }

  // C++ -> Python cast operation not supported.
};

}  // namespace detail
}  // namespace pybind11
