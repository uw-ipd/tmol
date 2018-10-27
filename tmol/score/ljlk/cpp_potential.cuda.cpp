#include <torch/torch.h>
#include "lj.hh"

namespace tmol {
namespace score {
namespace lj {

template <typename Real, typename Int>
at::Tensor block_interaction_lists(
    at::Tensor coords_t, Real max_dis, Int block_size);

extern template at::Tensor block_interaction_lists<float, int64_t>(
    at::Tensor coords_t, float max_dis, int64_t block_size);

template <typename Real, typename Int, typename PathLength>
at::Tensor lj_intra_block(
    at::Tensor coords_t,
    at::Tensor block_interactions_t,
    Int block_size,
    at::Tensor types_t,
    LJ_PARAM_ARGS);

extern template at::Tensor lj_intra_block<float, int64_t, uint8_t>(
    at::Tensor coords_t,
    at::Tensor block_interactions_t,
    int64_t block_size,
    at::Tensor types_t,
    LJ_PARAM_ARGS);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;

  m.def(
      "block_interaction_lists",
      &block_interaction_lists<float, int64_t>,
      "Calculate coordinate-block interaction lists.",
      "coords"_a,
      "max_dis"_a,
      "block_size"_a);

  m.def(
      "lj_intra_block",
      &lj_intra_block<float, int64_t, uint8_t>,
      "LJ intra-coordinate score.",
      "coords"_a,
      "block_iteractions"_a,
      "block_size"_a,
      "types"_a,
      LJ_PARAM_PYARGS);
}
}  // namespace lj
}  // namespace score
}  // namespace tmol
