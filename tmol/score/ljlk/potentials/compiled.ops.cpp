#include <torch/script.h>
#include <tmol/utility/autograd.hh>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include "lj.dispatch.hh"
#include "params.hh"

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

using torch::Tensor;

template <template <tmol::Device> class DispatchType>
Tensor lj_op(
    Tensor I,
    Tensor atom_type_I,
    Tensor J,
    Tensor atom_type_J,
    Tensor bonded_path_lengths,
    Tensor type_params,
    Tensor global_params) {
  using tmol::utility::connect_backward_pass;
  using tmol::utility::SavedGradsBackward;

  at::Tensor score;
  at::Tensor dScore_dI;
  at::Tensor dScore_dJ;

  using Int = int64_t;

  TMOL_DISPATCH_FLOATING_DEVICE(
      I.type(), "lj_op", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        auto result = LJDispatch<DispatchType, Dev, Real, Int>::f(
            TCAST(I),
            TCAST(atom_type_I),
            TCAST(J),
            TCAST(atom_type_J),
            TCAST(bonded_path_lengths),
            TCAST(type_params),
            TCAST(global_params));

        score = std::get<0>(result).tensor;
        dScore_dI = std::get<1>(result).tensor;
        dScore_dJ = std::get<2>(result).tensor;
      }));

  return connect_backward_pass({I, J}, score, [&]() {
    return SavedGradsBackward::create({dScore_dI, dScore_dJ});
  });
};

static auto registry =
    torch::jit::RegisterOperators()
        .op("tmol::score_ljlk_lj", &lj_op<AABBDispatch>)
        .op("tmol::score_ljlk_lj_triu", &lj_op<AABBTriuDispatch>);
}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
