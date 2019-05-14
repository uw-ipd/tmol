#include <torch/script.h>
#include <tmol/utility/autograd.hh>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/simple_dispatch.hh>

#include "params.hh"
#include "compiled.hh"

namespace tmol {
namespace score {
namespace cartbonded {
namespace potentials {

using torch::Tensor;

Tensor cbl_score_op(
      Tensor coords,
      Tensor atom_indices,
      Tensor param_table
) {
  using tmol::utility::connect_backward_pass;
  using tmol::utility::SavedGradsBackward;

  at::Tensor score;
  at::Tensor dScore;

  using Int = int64_t;

  TMOL_DISPATCH_FLOATING_DEVICE(
      coords.type(), "cbl_score_op", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        auto result = CartBondedLengthDispatch<Dev, Real, Int>::f(
            TCAST(coords),
            TCAST(atom_indices),
            TCAST(param_table));

        score = std::get<0>(result).tensor;
        dScore = std::get<1>(result).tensor;
      }));

  return connect_backward_pass({coords}, score, [&]() {
    return SavedGradsBackward::create({dScore});
  });
};

Tensor cba_score_op(
      Tensor coords,
      Tensor atom_indices,
      Tensor param_table
) {
  using tmol::utility::connect_backward_pass;
  using tmol::utility::SavedGradsBackward;

  at::Tensor score;
  at::Tensor dScore;

  using Int = int64_t;

  TMOL_DISPATCH_FLOATING_DEVICE(
      coords.type(), "cba_score_op", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        auto result = CartBondedAngleDispatch<Dev, Real, Int>::f(
            TCAST(coords),
            TCAST(atom_indices),
            TCAST(param_table));

        score = std::get<0>(result).tensor;
        dScore = std::get<1>(result).tensor;
      }));

  return connect_backward_pass({coords}, score, [&]() {
    return SavedGradsBackward::create({dScore});
  });
};

Tensor cbt_score_op(
      Tensor coords,
      Tensor atom_indices,
      Tensor param_table
) {
  using tmol::utility::connect_backward_pass;
  using tmol::utility::SavedGradsBackward;

  at::Tensor score;
  at::Tensor dScore;

  using Int = int64_t;

  TMOL_DISPATCH_FLOATING_DEVICE(
      coords.type(), "cbt_score_op", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        auto result = CartBondedTorsionDispatch<Dev, Real, Int>::f(
            TCAST(coords),
            TCAST(atom_indices),
            TCAST(param_table));

        score = std::get<0>(result).tensor;
        dScore = std::get<1>(result).tensor;
      }));

  return connect_backward_pass({coords}, score, [&]() {
    return SavedGradsBackward::create({dScore});
  });
};

Tensor cbht_score_op(
      Tensor coords,
      Tensor atom_indices,
      Tensor param_table
) {
  using tmol::utility::connect_backward_pass;
  using tmol::utility::SavedGradsBackward;

  at::Tensor score;
  at::Tensor dScore;

  using Int = int64_t;

  TMOL_DISPATCH_FLOATING_DEVICE(
      coords.type(), "cbht_score_op", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        auto result = CartBondedHxlTorsionDispatch<Dev, Real, Int>::f(
            TCAST(coords),
            TCAST(atom_indices),
            TCAST(param_table));

        score = std::get<0>(result).tensor;
        dScore = std::get<1>(result).tensor;
      }));

  return connect_backward_pass({coords}, score, [&]() {
    return SavedGradsBackward::create({dScore});
  });
};

static auto registry =
    torch::jit::RegisterOperators()
        .op("tmol::score_cartbonded_length", &cbl_score_op)
        .op("tmol::score_cartbonded_angle", &cba_score_op)
        .op("tmol::score_cartbonded_torsion", &cbt_score_op)
        .op("tmol::score_cartbonded_hxltorsion", &cbht_score_op);

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
