#include <tmol/utility/tensor/native_library.hh>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/utility/function_dispatch/aten.hh>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/device_operations.hh>

#include <pybind11/pybind11.h>

#include "constraint_score.hh"

namespace tmol {
namespace score {
namespace constraint {
namespace potentials {

ContextManager mgr;

using tmol::native::Tensor;
using tmol::native::autograd::AutogradContext;
using tmol::native::autograd::Function;
using tmol::native::autograd::tensor_list;

using namespace tmol::score::common;

template <template <tmol::Device> class DispatchMethod>
class GetTorsionAngleOp : public tmol::native::autograd::Function<
                              GetTorsionAngleOp<DispatchMethod>> {
 public:
  static Tensor forward(AutogradContext* ctx, Tensor coords) {
    tmol::native::Tensor angle;
    tmol::native::Tensor dangle_dcoords;

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        coords.options(), "get_torsion_angle_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result =
              GetTorsionAngleDispatch<DispatchMethod, Dev, Real>::forward(
                  mgr, TCAST(coords));

          angle = std::get<0>(result).tensor;
          dangle_dcoords = std::get<1>(result).tensor;
        }));

    ctx->save_for_backward({dangle_dcoords});

    return angle;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();

    tmol::native::Tensor dV_d_pose_coords;

    // single-score mode

    tensor_list result;

    for (auto& saved_grad : saved) {
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

// See https://stackoverflow.com/a/3221914
TMOL_LIBRARY(tmol_constraint, m) {
  m.def("get_torsion_angle", TMOL_FN(&get_torsion_angle_op<DeviceOperations>));
}

}  // namespace potentials
}  // namespace constraint
}  // namespace score
}  // namespace tmol
