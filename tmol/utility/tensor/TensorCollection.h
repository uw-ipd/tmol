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

//#include <tmol/utility/nvtx.hh>

namespace tmol {

// a container for an arbitrary sized set of TViews
// TCollection::view can be passed through device lambda capture
template <typename T, size_t N, Device D, PtrTag P = PtrTag::Restricted>
class TCollection {
 public:
  AT_HOST TCollection() {}

  AT_HOST TCollection(std::vector<at::Tensor> &tviews) {
    // nvtx_range_function();

    int n = tviews.size();

    // nvtx_range_push("TCollecion::tensors construction");
    tensors.resize(n);
    for (int i = 0; i < n; ++i) {
      tensors[i] = tmol::TPack<T, N, D, P>(tviews[i]);
    }
    // nvtx_range_pop();

    // nvtx_range_push("TCollecion::data_cpu construction");
    auto data_cpu =
        tmol::TPack<tmol::TView<T, N, D, P>, 1, tmol::Device::CPU, P>::empty(n);

    for (int i = 0; i < n; ++i) {
      data_cpu.view[i] = tensors[i].view;
    }
    // nvtx_range_pop();

    // push view data to CUDA device (if necessary)
    if (D == Device::CUDA) {
      // nvtx_range_push("TCollecion::to gpu");
      data = decltype(data)(data_cpu.tensor.to(at::kCUDA));
      // nvtx_range_pop();
    } else {
      data = decltype(data)(data_cpu.tensor);
    }
    view = data.view;
  }

  at::Device device() const {
    if (tensors.size() > 0) {
      return tensors[0].tensor.device();
    } else if (D == tmol::Device::CPU) {
      return at::Device(at::Device::Type::CPU);
    } else if (D == tmol::Device::CUDA) {
      return at::Device(at::Device::Type::CUDA);
    }
  }

  size_t size() const { return tensors.size(); }

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
