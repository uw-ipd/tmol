#include <torch/extension.h>
#include <cmath>
#include <map>
#include <string>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>

template <int N, tmol::Device D>
struct TData {
  tmol::TView<int64_t, N, D> a;
  tmol::TView<int64_t, N, D> b;

  template <typename TMap>
  static TData view_tensor_map(TMap& map) {
    return {
        tmol::view_tensor_item<int64_t, N, D>(map, "a"),
        tmol::view_tensor_item<int64_t, N, D>(map, "b")};
  }
};

namespace pybind11 {
namespace detail {

template <int N, tmol::Device D>
struct type_caster<TData<N, D>> {
 public:
  // typedef template specialization so it can be passed into macro.
  typedef TData<N, D> TDataT;
  PYBIND11_TYPE_CASTER(TDataT, _<TDataT>());

  bool load(handle src, bool convert) {
    try {
      type_caster<std::map<std::string, at::Tensor>> map_conv;
      if (map_conv.load(src, convert)) {
        value = TData<N, D>::view_tensor_map(*map_conv);
        return true;
      }
    } catch (at::Error err) {
    }

    return false;
  }
};

}  // namespace detail
}  // namespace pybind11

int64_t sum_a(std::map<std::string, at::Tensor> tmap) {
  TData<1, tmol::Device::CPU> tdata =
      TData<1, tmol::Device::CPU>::view_tensor_map(tmap);

  int64_t v = 0;
  for (int i = 0; i < tdata.a.size(0); ++i) {
    v += tdata.a[i];
  }

  return v;
}

int64_t sum_a_map(TData<1, tmol::Device::CPU> tdata) {
  int64_t v = 0;
  for (int i = 0; i < tdata.a.size(0); ++i) {
    v += tdata.a[i];
  }

  return v;
}

int64_t sum(at::Tensor tensor_data_t) {
  auto tensor_data = tmol::view_tensor<int64_t, 1, tmol::Device::CPU>(
      tensor_data_t, "tensor_data");

  int64_t v = 0;
  for (int i = 0; i < tensor_data.size(0); ++i) {
    v += tensor_data[i];
  }

  return v;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sum_a", &sum_a, "Sum tensor member a values.");
  m.def("sum_a_map", &sum_a_map, "Sum tensor member a values.");
  m.def("sum", &sum, "Sum tensor values.");
}
