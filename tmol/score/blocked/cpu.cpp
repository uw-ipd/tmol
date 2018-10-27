#include <ATen/ScalarTypeUtils.h>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <torch/torch.h>
#include "Eigen/Geometry"

namespace tmol {
namespace score {
namespace blocked {

template <typename Real, typename Int>
at::Tensor coord_interaction_table(at::Tensor coords_t, Real max_dis) {
  typedef Eigen::AlignedBox<Real, 3> Box;
  typedef Eigen::Matrix<Real, 3, 1> Vector;

  TView<Vector, 2> coords = tmol::view_tensor<Vector, 2>(coords_t);

  at::Tensor result_t = at::empty(
      {coords.size(0), coords.size(0)},
      at::TensorOptions(coords_t).dtype(at::CTypeToScalarType<Int>::to()));
  TView<Int, 2> result = tmol::view_tensor<Int, 2>(result_t);

  for (int i = 0; i < coords.size(0); ++i) {
    Box box(coords[i][0]);
    box.extend(box.max() + Vector(max_dis, max_dis, max_dis));
    box.extend(box.min() - Vector(max_dis, max_dis, max_dis));

    for (int j = 0; j < coords.size(0); ++j) {
      result[i][j] = box.contains(coords[j][0]) ? 1 : 0;
    }
  }

  return result_t;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;

  m.def(
      "coord_interaction_table",
      &coord_interaction_table<float, int64_t>,
      "Calculate coordinate-coordinate aabb interaction table.",
      "coords"_a,
      "max_dis"_a);
}

}  // namespace blocked
}  // namespace score
}  // namespace tmol
