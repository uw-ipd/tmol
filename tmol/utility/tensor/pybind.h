#pragma once

#include <pybind11/pybind11.h>

#include <torch/csrc/utils/pybind.h>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>

namespace pybind11 {
namespace detail {

template <typename T, size_t N, template <typename U> class P>
struct type_caster<tmol::TView<T, N, P>> {
 public:
  typedef tmol::TView<T, N, P> ViewType;
  PYBIND11_TYPE_CASTER(ViewType, _("TENSOR"));

  bool load(handle src, bool convert) {
    using pybind11::print;

    type_caster<at::Tensor> conv;

    if (!conv.load(src, convert)) {
      print("Error casting to tensor: ", src);
      return false;
    }

    try {
      value = tmol::view_tensor<T, N, P>(conv);
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
