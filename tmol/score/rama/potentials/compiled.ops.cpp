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
namespace rama {
namespace potentials {

using torch::Tensor;

template < template <tmol::Device> class DispatchMethod >
Tensor rama_op(
      Tensor coords,
      Tensor params,
      Tensor tables,
      Tensor table_params
) {
  using tmol::utility::connect_backward_pass;
  using tmol::utility::SavedGradsBackward;
  nvtx_range_push("rama_op");

  at::Tensor score;
  at::Tensor dScore;

  using Int = int32_t;

  TMOL_DISPATCH_FLOATING_DEVICE(
      coords.type(), "rama_op", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        auto result = RamaDispatch<DispatchMethod, Dev, Real, Int>::f(
            TCAST(coords),
            TCAST(params),
            TCAST(tables),
            TCAST(table_params));

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
        .op("tmol::score_rama", &rama_op<common::ForallDispatch>);

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
