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
namespace cartbonded {
namespace potentials {

using torch::Tensor;

template < template <tmol::Device> class DispatchMethod >
Tensor cbl_score_op(
      Tensor coords,
      Tensor atom_indices,
      Tensor param_table
) {
  using tmol::utility::connect_backward_pass;
  using tmol::utility::SavedGradsBackward;
  nvtx_range_push("cbl_score_op");

  at::Tensor score;
  at::Tensor dScore;

  using Int = int64_t;

  TMOL_DISPATCH_FLOATING_DEVICE(
      coords.type(), "cbl_score_op", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        auto result = CartBondedLengthDispatch<DispatchMethod, Dev, Real, Int>::f(
            TCAST(coords),
            TCAST(atom_indices),
            TCAST(param_table));

        score = std::get<0>(result).tensor;
        dScore = std::get<1>(result).tensor;
      }));

  auto backward_op = connect_backward_pass({coords}, score, [&]() {
    return SavedGradsBackward::create({dScore});
  });

  nvtx_range_pop();
  return backward_op;
};


template < template <tmol::Device> class DispatchMethod >
Tensor cba_score_op(
      Tensor coords,
      Tensor atom_indices,
      Tensor param_table
) {
  using tmol::utility::connect_backward_pass;
  using tmol::utility::SavedGradsBackward;
  nvtx_range_push("cba_score_op");

  at::Tensor score;
  at::Tensor dScore;

  using Int = int64_t;

  TMOL_DISPATCH_FLOATING_DEVICE(
      coords.type(), "cba_score_op", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        auto result = CartBondedAngleDispatch<DispatchMethod, Dev, Real, Int>::f(
            TCAST(coords),
            TCAST(atom_indices),
            TCAST(param_table));

        score = std::get<0>(result).tensor;
        dScore = std::get<1>(result).tensor;
      }));

  nvtx_range_pop();
  return connect_backward_pass({coords}, score, [&]() {
    return SavedGradsBackward::create({dScore});
  });
};


template < template <tmol::Device> class DispatchMethod >
Tensor cbt_score_op(
      Tensor coords,
      Tensor atom_indices,
      Tensor param_table
) {
  using tmol::utility::connect_backward_pass;
  using tmol::utility::SavedGradsBackward;
  nvtx_range_push("cbt_score_op");

  at::Tensor score;
  at::Tensor dScore;

  using Int = int64_t;

  TMOL_DISPATCH_FLOATING_DEVICE(
      coords.type(), "cbt_score_op", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        auto result = CartBondedTorsionDispatch<DispatchMethod, Dev, Real, Int>::f(
            TCAST(coords),
            TCAST(atom_indices),
            TCAST(param_table));

        score = std::get<0>(result).tensor;
        dScore = std::get<1>(result).tensor;
      }));

  auto backward_op = connect_backward_pass({coords}, score, [&]() {
    return SavedGradsBackward::create({dScore});
  });

  nvtx_range_pop();
  return backward_op;
};


template < template <tmol::Device> class DispatchMethod >
Tensor cbht_score_op(
      Tensor coords,
      Tensor atom_indices,
      Tensor param_table
) {
  using tmol::utility::connect_backward_pass;
  using tmol::utility::SavedGradsBackward;
  nvtx_range_push("cbht_score_op");

  at::Tensor score;
  at::Tensor dScore;

  using Int = int64_t;

  TMOL_DISPATCH_FLOATING_DEVICE(
      coords.type(), "cbht_score_op", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        auto result = CartBondedHxlTorsionDispatch<DispatchMethod, Dev, Real, Int>::f(
            TCAST(coords),
            TCAST(atom_indices),
            TCAST(param_table));

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
        .op("tmol::score_cartbonded_length", &cbl_score_op<common::ForallDispatch>)
        .op("tmol::score_cartbonded_angle", &cba_score_op<common::ForallDispatch>)
        .op("tmol::score_cartbonded_torsion", &cbt_score_op<common::ForallDispatch>)
        .op("tmol::score_cartbonded_hxltorsion", &cbht_score_op<common::ForallDispatch>);

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
