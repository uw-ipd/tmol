#include <torch/script.h>
#include <tmol/utility/autograd.hh>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/simple_dispatch.hh>
#include "dispatch.hh"
#include "gen_waters.hh"

namespace tmol {
namespace score {
namespace lk_ball {
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
Tensor score_op(
    Tensor I,
    Tensor atom_type_I,
    Tensor waters_I,
    Tensor J,
    Tensor atom_type_J,
    Tensor waters_J,
    Tensor bonded_path_lengths,
    Tensor type_params,
    Tensor global_params) {
  using tmol::utility::connect_backward_pass;
  using tmol::utility::SavedGradsBackward;

  at::Tensor score;

  using Int = int64_t;

  TMOL_DISPATCH_FLOATING_DEVICE(
      I.type(), "score_op", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        auto result = ScoreDispatch<DispatchMethod, Dev, Real, Int>::forward(
            TCAST(I),
            TCAST(atom_type_I),
            TCAST(waters_I),
            TCAST(J),
            TCAST(atom_type_J),
            TCAST(waters_J),
            TCAST(bonded_path_lengths),
            TCAST(type_params),
            TCAST(global_params));

        score = result.tensor;
      }));

  return connect_backward_pass({I, J}, score, [&]() {
    return SavedGradsBackward::create({});
  });
};

template <
    template <
        tmol::Device D,
        typename Real,
        typename Int,
        int MAX_WATER>
    class WaterGenDispatch>
Tensor watergen_op(
    Tensor coords,
    Tensor atom_types,
    Tensor indexed_bonds,
    Tensor indexed_bond_spans,
    Tensor type_params,
    Tensor global_params,
    Tensor sp2_water_tors,
    Tensor sp3_water_tors,
    Tensor ring_water_tors ) {
  using tmol::utility::connect_backward_pass;
  using tmol::utility::SavedGradsBackward;

  at::Tensor score;

  using Int = int64_t;
  constexpr int MAX_WATER = 4;

  TMOL_DISPATCH_FLOATING_DEVICE(
      coords.type(), "score_op", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        auto result = WaterGenDispatch<Dev, Real, Int, MAX_WATER>::forward(
            TCAST(coords),
            TCAST(atom_types),
            TCAST(indexed_bonds),
            TCAST(indexed_bond_spans),
            TCAST(type_params),
            TCAST(global_params),
            TCAST(sp2_water_tors),
            TCAST(sp3_water_tors),
            TCAST(ring_water_tors));

        score = result.tensor;
      }));

  return connect_backward_pass({coords}, score, [&]() {
    return SavedGradsBackward::create({});
  });
};

static auto registry =
    torch::jit::RegisterOperators()
        .op("tmol::score_lkball", &score_op<LKBallDispatch, common::AABBDispatch>)
        .op("tmol::watergen_lkball", &watergen_op<GenerateWaters>);

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
