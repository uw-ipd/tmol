#include <torch/torch.h>

namespace tmol {
namespace score {
namespace blocked {

template <typename Real, typename Int, int BLOCK_SIZE>
at::Tensor block_interaction_table(at::Tensor coords_t, Real max_dis);

extern template at::Tensor block_interaction_table<float, int32_t, 8>(
    at::Tensor coords_t, float max_dis);

template <typename Real, typename Int, int BLOCK_SIZE>
std::tuple<at::Tensor, at::Tensor> block_interaction_list(
    at::Tensor coords_t, Real max_dis);

extern template std::tuple<at::Tensor, at::Tensor>
block_interaction_list<float, int32_t, 8>(at::Tensor coords_t, float max_dis);

template <typename Real>
at::Tensor calc_block_aabb(at::Tensor coords_t);

extern template at::Tensor calc_block_aabb<float>(at::Tensor coords_t);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;

  m.def(
      "block_interaction_table",
      &block_interaction_table<float, int32_t, 8>,
      "Calculate coordinate-block interaction table.",
      "coords"_a,
      "max_dis"_a);

  m.def(
      "block_interaction_list",
      &block_interaction_list<float, int32_t, 8>,
      "Calculate coordinate-block interaction list.",
      "coords"_a,
      "max_dis"_a);

  m.def(
      "calc_block_aabb",
      &calc_block_aabb<float>,
      "Calculate coordinate-block aabb.",
      "coords"_a);
}
}  // namespace blocked
}  // namespace score
}  // namespace tmol
