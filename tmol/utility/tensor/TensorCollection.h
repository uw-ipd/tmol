#pragma once

#include <array>

#include <ATen/Error.h>
#include <ATen/Functions.h>
#include <ATen/ScalarType.h>
#include <ATen/Tensor.h>

#include <torch/torch.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>

#include <pybind11/pybind11.h>

namespace tmol {

// a container for an arbitrary sized set of TViews
// TCollection::view can be passed through device lambda capture
template <typename T, size_t N, Device D, PtrTag P = PtrTag::Restricted>
class TCollection {
 public:
  AT_HOST TCollection() {}

  AT_HOST TCollection(std::vector<at::Tensor> &tviews) {
    int n = tviews.size();

    tensors.resize(n);
    for (int i = 0; i < n; ++i) {
      tensors[i] = tmol::TPack<T, N, D, P>(tviews[i]);
    }

    auto data_cpu =
        tmol::TPack<tmol::TView<T, N, D, P>, 1, tmol::Device::CPU, P>::empty(n);

    for (int i = 0; i < n; ++i) {
      data_cpu.view[i] = tensors[i].view;
    }

    // push view data to CUDA device (if necessary)
    if (D == Device::CUDA) {
      data = decltype(data)(data_cpu.tensor.to(at::kCUDA));
    } else {
      data = decltype(data)(data_cpu.tensor);
    }
    view = data.view;
  }

  static constexpr size_t blocksize = sizeof(tmol::TView<T, N, D, P>);
  std::vector<tmol::TPack<T, N, D, P>> tensors;
  tmol::TPack<tmol::TView<T, N, D, P>, 1, D, P> data;
  tmol::TView<tmol::TView<T, N, D, P>, 1, D, P> view;
};

// TCollection conversion as a tensor of bytes
template <typename T, size_t N, Device D, PtrTag P>
struct enable_tensor_view<tmol::TView<T, N, D, P>> {
  static const bool enabled = enable_tensor_view<uint8_t>::enabled;
  static const at::ScalarType scalar_type =
      enable_tensor_view<uint8_t>::scalar_type;
  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0) ? tmol::TCollection<T, N, D, P>::blocksize : 0;
  }
  typedef typename enable_tensor_view<uint8_t>::PrimitiveType PrimitiveType;
};

}  // namespace tmol
