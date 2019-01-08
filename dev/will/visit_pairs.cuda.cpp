#include <torch/torch.h>
#include "lj.hh"

namespace tmol {
namespace willtest {

template <typename Real>
int visit_pairs_cpu_naive(at::Tensor pts, Real dis);

extern template int visit_pairs_cpu_naive<float>(at::Tensor pts, float dis);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;

  m.def("visit_pairs_cpu_naive", &visit_pairs_cpu_naive<float>);
}

}  // namespace willtest
}  // namespace tmol
