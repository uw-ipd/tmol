#include <torch/torch.h>
#include "lj.hh"

namespace tmol {
namespace score {
namespace lj {

template <int BLOCK_SIZE, typename Real, typename Int, typename PathLength>
std::tuple<at::Tensor, at::Tensor, at::Tensor> lj_intra_block(
    at::Tensor coords_t, at::Tensor types_t, LJ_PARAM_ARGS);

extern template std::tuple<at::Tensor, at::Tensor, at::Tensor>
lj_intra_block<8, float, int64_t, uint8_t>(at::Tensor coords_t,
                                           at::Tensor types_t, LJ_PARAM_ARGS);

int foo(int a) { return 2 * a; }

extern int foocuda(int a);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;

  m.def("lj_intra_block", &lj_intra_block<8, float, int64_t, uint8_t>,
        "LJ intra-coordinate score.", "coords"_a, "types"_a, LJ_PARAM_PYARGS);

  m.def("foo", &foo);
  m.def("foocuda", &foocuda);
}
}  // namespace lj
}  // namespace score
}  // namespace tmol
