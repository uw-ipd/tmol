#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <torch/torch.h>
#include <cmath>
#include <map>
#include <string>

struct TData {
  tmol::TView<int64_t, 1> a;
  tmol::TView<int64_t, 1> b;

  static TData from_map(std::map<std::string, at::Tensor> map) {
    TData result = {tmol::view_tensor<int64_t, 1>(map, "a"),
                    tmol::view_tensor<int64_t, 1>(map, "b")};

    return result;
  }
};

int64_t sum_a(std::map<std::string, at::Tensor> tmap) {
  TData tdata = TData::from_map(tmap);

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
  m.def("sum", &sum, "Sum tensor values.");
}
