#include <torch/torch.h>

namespace tmol {
namespace score {
namespace lj {

at::Tensor lj_intra_cuda(
    at::Tensor coords_t,
    at::Tensor types_t,
    at::Tensor bonded_path_length_t,
    at::Tensor lj_sigma_t,
    at::Tensor lj_switch_slope_t,
    at::Tensor lj_switch_intercept_t,
    at::Tensor lj_coeff_sigma12_t,
    at::Tensor lj_coeff_sigma6_t,
    at::Tensor lj_spline_y0_t,
    at::Tensor lj_spline_dy0_t,
    at::Tensor lj_switch_dis2sigma_t,
    at::Tensor spline_start_t,
    at::Tensor max_dis_t);

#define CHECK_CUDA(x) \
  AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x);

at::Tensor lj_intra(
    at::Tensor coords_t,
    at::Tensor types_t,
    at::Tensor bonded_path_length_t,
    at::Tensor lj_sigma_t,
    at::Tensor lj_switch_slope_t,
    at::Tensor lj_switch_intercept_t,
    at::Tensor lj_coeff_sigma12_t,
    at::Tensor lj_coeff_sigma6_t,
    at::Tensor lj_spline_y0_t,
    at::Tensor lj_spline_dy0_t,
    at::Tensor lj_switch_dis2sigma_t,
    at::Tensor spline_start_t,
    at::Tensor max_dis_t) {
  CHECK_INPUT(coords_t);
  CHECK_INPUT(types_t);
  CHECK_INPUT(bonded_path_length_t);
  CHECK_INPUT(lj_sigma_t);
  CHECK_INPUT(lj_switch_slope_t);
  CHECK_INPUT(lj_switch_intercept_t);
  CHECK_INPUT(lj_coeff_sigma12_t);
  CHECK_INPUT(lj_coeff_sigma6_t);
  CHECK_INPUT(lj_spline_y0_t);
  CHECK_INPUT(lj_spline_dy0_t);
  CHECK_INPUT(lj_switch_dis2sigma_t);
  CHECK_INPUT(spline_start_t);
  CHECK_INPUT(max_dis_t);

  return lj_intra_cuda(
      coords_t,
      types_t,
      bonded_path_length_t,
      lj_sigma_t,
      lj_switch_slope_t,
      lj_switch_intercept_t,
      lj_coeff_sigma12_t,
      lj_coeff_sigma6_t,
      lj_spline_y0_t,
      lj_spline_dy0_t,
      lj_switch_dis2sigma_t,
      spline_start_t,
      max_dis_t);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;

  m.def(
      "lj_intra",
      &lj_intra,
      "LJ intra-coordinate score.",
      "coords"_a,
      "types"_a,
      "bonded_path_length"_a,
      "lj_sigma"_a,
      "lj_switch_slope"_a,
      "lj_switch_intercept"_a,
      "lj_coeff_sigma12"_a,
      "lj_coeff_sigma6"_a,
      "lj_spline_y0"_a,
      "lj_spline_dy0"_a,
      "lj_switch_dis2sigma"_a,
      "spline_start"_a,
      "max_dis"_a);
}
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
