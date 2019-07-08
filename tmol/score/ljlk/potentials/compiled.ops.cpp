#include <torch/script.h>
#include <tmol/utility/autograd.hh>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/forall_dispatch.hh>
#include <tmol/score/common/simple_dispatch.hh>

#include "lj.dispatch.hh"
#include "lk_isotropic.dispatch.hh"

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

using torch::Tensor;

template <
    template <
        template <tmol::Device>
        class SingleDispatch,
        template <tmol::Device>
        class PairDispatch,
        tmol::Device D,
        typename Real,
        typename Int>
    class ScoreDispatch,
    template <tmol::Device>
    class SingleDispatchMethod,
    template <tmol::Device>
    class PairDispatchMethod>
Tensor score_op(
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
      I.type(), "score_op", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        auto result = ScoreDispatch<SingleDispatchMethod, PairDispatchMethod, Dev, Real, Int>::f(
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
  .op("tmol::score_ljlk_lj", &score_op<LJDispatch, common::ForallDispatch, common::AABBDispatch>)
  .op("tmol::score_ljlk_lj_triu", &score_op<LJDispatch, common::ForallDispatch, common::AABBTriuDispatch>)
        .op("tmol::score_ljlk_lk_isotropic",
	  &score_op<LKIsotropicDispatch, common::ForallDispatch, common::AABBDispatch>)
        .op("tmol::score_ljlk_lk_isotropic_triu",
	  &score_op<LKIsotropicDispatch, common::ForallDispatch, common::AABBTriuDispatch>);

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
