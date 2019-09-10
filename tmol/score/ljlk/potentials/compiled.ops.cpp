#include <torch/script.h>
#include <tmol/utility/autograd.hh>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

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
        class Dispatch,
        tmol::Device D,
        typename Real,
        typename Int>
    class ScoreDispatch,
    template <tmol::Device>
    class DispatchMethod>
Tensor lj_score_op(
    Tensor I,
    Tensor atom_type_I,
    Tensor J,
    Tensor atom_type_J,
    Tensor bonded_path_lengths,
    Tensor type_params,
    Tensor global_params) {
  using tmol::utility::connect_backward_pass;
  using tmol::utility::StackedSavedGradsBackward;

  at::Tensor score;
  at::Tensor dScore_dI;
  at::Tensor dScore_dJ;

  using Int = int64_t;

  TMOL_DISPATCH_FLOATING_DEVICE(
      I.type(), "score_op", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        auto result = ScoreDispatch<DispatchMethod, Dev, Real, Int>::f(
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
    return StackedSavedGradsBackward::create({dScore_dI, dScore_dJ});
  });
};

template <
    template <
        template <tmol::Device>
        class Dispatch,
        tmol::Device D,
        typename Real,
        typename Int>
    class ScoreDispatch,
    template <tmol::Device>
    class DispatchMethod>
Tensor lk_score_op(
    Tensor I,
    Tensor atom_type_I,
    Tensor heavyatom_inds_I,
    Tensor J,
    Tensor atom_type_J,
    Tensor heavyatom_inds_J,
    Tensor bonded_path_lengths,
    Tensor type_params,
    Tensor global_params) {
  using tmol::utility::connect_backward_pass;
  using tmol::utility::StackedSavedGradsBackward;

  at::Tensor score;
  at::Tensor dScore_dI;
  at::Tensor dScore_dJ;

  using Int = int64_t;

  TMOL_DISPATCH_FLOATING_DEVICE(
      I.type(), "score_op", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        auto result = ScoreDispatch<DispatchMethod, Dev, Real, Int>::f(
            TCAST(I),
            TCAST(atom_type_I),
            TCAST(heavyatom_inds_I),
            TCAST(J),
            TCAST(atom_type_J),
            TCAST(heavyatom_inds_J),
            TCAST(bonded_path_lengths),
            TCAST(type_params),
            TCAST(global_params));

        score = std::get<0>(result).tensor;
        dScore_dI = std::get<1>(result).tensor;
        dScore_dJ = std::get<2>(result).tensor;
      }));

  return connect_backward_pass({I, J}, score, [&]() {
    return StackedSavedGradsBackward::create({dScore_dI, dScore_dJ});
  });
};

static auto registry =
    torch::jit::RegisterOperators()
        .op("tmol::score_ljlk_lj", &lj_score_op<LJDispatch, AABBDispatch>)
        .op("tmol::score_ljlk_lj_triu", &lj_score_op<LJDispatch, AABBTriuDispatch>)
        .op("tmol::score_ljlk_lk_isotropic",
            &lk_score_op<LKIsotropicDispatch, AABBDispatch>)
        .op("tmol::score_ljlk_lk_isotropic_triu",
            &lk_score_op<LKIsotropicDispatch, AABBTriuDispatch>);

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
