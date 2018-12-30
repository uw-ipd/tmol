#include <torch/torch.h>
#include <cmath>
#include <map>
#include <string>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>

template <int N>
struct TData {
  tmol::TView<int64_t, N> a;
  tmol::TView<int64_t, N> b;

  template <typename TMap>
  static TData view_tensor_map(TMap& map) {
    return {tmol::view_tensor_item<int64_t, N>(map, "a"),
            tmol::view_tensor_item<int64_t, N>(map, "b")};
  }
};

namespace pybind11 {
namespace detail {

template <int N>
struct type_caster<TData<N>> {
 public:
  PYBIND11_TYPE_CASTER(TData<N>, _<TData<N>>());

  bool load(handle src, bool convert) {
    try {
      type_caster<std::map<std::string, at::Tensor>> map_conv;
      if (map_conv.load(src, convert)) {
        value = TData<N>::view_tensor_map(*map_conv);
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
  TData<1> tdata = TData<1>::view_tensor_map(tmap);

  int64_t v = 0;
  for (int i = 0; i < tdata.a.size(0); ++i) {
    v += tdata.a[i];
  }

  return v;
}

int64_t sum_a_map(TData<1> tdata) {
  int64_t v = 0;
  for (int i = 0; i < tdata.a.size(0); ++i) {
    v += tdata.a[i];
  }

  return v;
}

int64_t sum(at::Tensor dat_t) {
  auto dat = tmol::view_tensor<int64_t, 1>(dat_t, "dat");

  int64_t v = 0;
  for (int i = 0; i < dat.size(0); ++i) {
    v += dat[i];
  }

  return v;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sum_a", &sum_a, "Sum tensor member a values.");
  m.def("sum_a_map", &sum_a_map, "Sum tensor member a values.");
  m.def("sum", &sum, "Sum tensor values.");
}
