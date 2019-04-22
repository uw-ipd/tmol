#include <ATen/Dispatch.h>
#include <torch/script.h>

#include "lj.dispatch.hh"

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

#ifdef WITH_CUDA

#define TMOL_DISPATCH_REAL_AND_DEV(TYPE, NAME, ...)          \
  [&] {                                                      \
    if (TYPE.device_type() == at::DeviceType::CPU) {         \
      constexpr tmol::Device device_t = tmol::Device::CPU;   \
                                                             \
      AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, __VA_ARGS__);   \
    } else if (TYPE.device_type() == at::DeviceType::CUDA) { \
      constexpr tmol::Device device_t = tmol::Device::CUDA;  \
                                                             \
      AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, __VA_ARGS__);   \
    } else {                                                 \
      AT_ERROR("Unsupported tensor device type.");           \
    }                                                        \
  }();

#else

#define TMOL_DISPATCH_REAL_AND_DEV(TYPE, NAME, ...)          \
  [&] {                                                      \
    if (TYPE.device_type() == at::DeviceType::CPU) {         \
      constexpr tmol::Device device_t = tmol::Device::CPU;   \
                                                             \
      AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, __VA_ARGS__);   \
    } else if (TYPE.device_type() == at::DeviceType::CUDA) { \
      AT_ERROR("Unsupported cuda tensor, non-cuda build.");  \
    } else {                                                 \
      AT_ERROR("Unsupported device type.");                  \
    }                                                        \
  }();
#endif

using torch::Tensor;

template <typename Real, tmol::Device D>
auto view_type_params(std::vector<Tensor> tensors)
    -> LJTypeParamTensors<Real, D> {
  AT_CHECK(tensors.size() == 6, "Invalid type_params tensor list length.");

  return {view_tensor<Real, 1, D>(tensors.at(0), "lj_radius"),
          view_tensor<Real, 1, D>(tensors.at(1), "lj_wdepth"),
          view_tensor<bool, 1, D>(tensors.at(2), "is_donor"),
          view_tensor<bool, 1, D>(tensors.at(3), "is_hydroxyl"),
          view_tensor<bool, 1, D>(tensors.at(4), "is_polarh"),
          view_tensor<bool, 1, D>(tensors.at(5), "is_acceptor")};
}

template <typename Real, tmol::Device D>
auto view_global_params(std::vector<Tensor> tensors)
    -> LJGlobalParamTensors<Real, D> {
  AT_CHECK(tensors.size() == 3, "Invalid global_params tensor list length.");

  return {view_tensor<Real, 1, D>(tensors.at(0), "lj_hbond_dis"),
          view_tensor<Real, 1, D>(tensors.at(1), "lj_hbond_OH_donor_dis"),
          view_tensor<Real, 1, D>(tensors.at(2), "lj_hbond_hdis")};
};

Tensor lj_triu(
    Tensor I,
    Tensor atom_type_I,
    Tensor J,
    Tensor atom_type_J,
    Tensor bonded_path_lengths,
    std::vector<Tensor> type_params,
    std::vector<Tensor> global_params) {
  at::Tensor score;

  using Int = int64_t;

  TMOL_DISPATCH_REAL_AND_DEV(
      I.type(), "lj_triu", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        auto result = LJDispatch<AABBTriuDispatch, Dev, Real, Int>::f(
            view_tensor<Vec<Real, 3>, 1, Dev>(I, "I"),
            view_tensor<Int, 1, Dev>(atom_type_I, "atom_type_I"),
            view_tensor<Vec<Real, 3>, 1, Dev>(J, "J"),
            view_tensor<Int, 1, Dev>(atom_type_I, "atom_type_J"),
            view_tensor<Real, 2, Dev>(
                bonded_path_lengths, "bonded_path_lengths"),
            view_type_params<Real, Dev>(type_params),
            view_global_params<Real, Dev>(global_params));

        score = std::get<0>(result).tensor;
      }));

  return score;
};

static auto registry =
    torch::jit::RegisterOperators("tmol::score_ljlk_lj_triu", &lj_triu);
}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
