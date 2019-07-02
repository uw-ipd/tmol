#include <torch/script.h>
#include <tmol/utility/autograd.hh>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/forall_dispatch.hh>

#include "params.hh"
#include "dispatch.hh"

namespace tmol {
namespace score {
namespace omega {
namespace potentials {

using torch::Tensor;

template < template <tmol::Device> class DispatchMethod >
Tensor omega_op(
      Tensor coords,
      Tensor params
) {
  using tmol::utility::connect_backward_pass;
  using tmol::utility::SavedGradsBackward;
  nvtx_range_push("omega_op");

  at::Tensor score;
  at::Tensor dScore;

  using Int = int32_t;

  TMOL_DISPATCH_FLOATING_DEVICE(
      coords.type(), "omega_op", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        auto result = OmegaDispatch<DispatchMethod, Dev, Real, Int>::f(
            TCAST(coords),
            TCAST(params));

        score = std::get<0>(result).tensor;
        dScore = std::get<1>(result).tensor;
      }));

  auto backward_op = connect_backward_pass({coords}, score, [&]() {
    return SavedGradsBackward::create({dScore});
  });

  nvtx_range_pop();
  return backward_op;
};


static auto registry =
    torch::jit::RegisterOperators()
        .op("tmol::score_omega", &omega_op<common::ForallDispatch>);

}  // namespace potentials
}  // namespace omega
}  // namespace score
}  // namespace tmol
