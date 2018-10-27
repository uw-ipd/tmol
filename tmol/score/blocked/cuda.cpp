#include <torch/torch.h>

namespace tmol {
namespace score {
namespace blocked {

template <typename Real, typename Int, int BLOCK_SIZE>
at::Tensor block_interaction_table(at::Tensor coords_t, Real max_dis);

extern template at::Tensor block_interaction_table<float, int32_t, 8>(
    at::Tensor coords_t, float max_dis);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;

  m.def(
      "block_interaction_table",
      &block_interaction_table<float, int32_t, 8>,
      "Calculate coordinate-block interaction table.",
      "coords"_a,
      "max_dis"_a);
}
}  // namespace blocked
}  // namespace score
}  // namespace tmol
