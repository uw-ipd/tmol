#include <torch/torch.h>
#include <torch/script.h>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/device_operations.hh>
#include <tmol/score/common/forall_dispatch.hh>

#include <pybind11/pybind11.h>

#include "constraint_score.hh"

namespace tmol {
namespace score {
namespace constraint {
namespace potentials {

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::tensor_list;

using namespace tmol::score::common;

template <template <tmol::Device> class DispatchMethod>
class GetTorsionAngleOp
    : public torch::autograd::Function<GetTorsionAngleOp<DispatchMethod>> {
 public:
  static Tensor forward(AutogradContext* ctx, Tensor coords) {
    at::Tensor angle;
    at::Tensor dangle_dcoords;

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        coords.type(), "get_torsion_angle_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result =
              GetTorsionAngleDispatch<DispatchMethod, Dev, Real>::forward(
                  TCAST(coords));

          angle = std::get<0>(result).tensor;
          dangle_dcoords = std::get<1>(result).tensor;
        }));

    ctx->save_for_backward({dangle_dcoords});

    return angle;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();

    at::Tensor dV_d_pose_coords;

    // single-score mode
    auto saved_grads = ctx->get_saved_variables();

    tensor_list result;

    for (auto& saved_grad : saved_grads) {
      auto ingrad = grad_outputs[0];
      while (ingrad.dim() < saved_grad.dim()) {
        ingrad = ingrad.unsqueeze(-1);
      }

      result.emplace_back(saved_grad * ingrad);
    }

    int i = 0;
    dV_d_pose_coords = result[i++];

    return {
        dV_d_pose_coords,
    };
  }
};

template <template <tmol::Device> class DispatchMethod>
Tensor get_torsion_angle_op(Tensor coords) {
  return GetTorsionAngleOp<DispatchMethod>::apply(coords);
}

// Macro indirection to force TORCH_EXTENSION_NAME macro expansion
// See https://stackoverflow.com/a/3221914
#define TORCH_LIBRARY_(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_(TORCH_EXTENSION_NAME, m) {
  m.def("get_torsion_angle", &get_torsion_angle_op<DeviceOperations>);
}

}  // namespace potentials
}  // namespace constraint
}  // namespace score
}  // namespace tmol
