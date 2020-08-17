#include <torch/torch.h>
#include <torch/script.h>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/simple_dispatch.hh>
#include <tmol/score/common/forall_dispatch.hh>
#include "dispatch.hh"
#include "gen_waters.hh"

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::tensor_list;

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
class ScoreOp
    : public torch::autograd::Function<ScoreOp<ScoreDispatch, DispatchMethod>> {
 public:
  static Tensor forward(
      AutogradContext* ctx,
      Tensor I,
      Tensor polars_I,
      Tensor atom_type_I,
      Tensor waters_I,
      Tensor J,
      Tensor occluders_J,
      Tensor atom_type_J,
      Tensor waters_J,
      Tensor bonded_path_lengths,
      Tensor type_params,
      Tensor global_params) {
    at::Tensor score;

    using Int = int64_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        I.type(), "score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result = ScoreDispatch<DispatchMethod, Dev, Real, Int>::forward(
              TCAST(I),
              TCAST(polars_I),
              TCAST(atom_type_I),
              TCAST(waters_I),
              TCAST(J),
              TCAST(occluders_J),
              TCAST(atom_type_J),
              TCAST(waters_J),
              TCAST(bonded_path_lengths),
              TCAST(type_params),
              TCAST(global_params));

          score = result.tensor;
        }));

    ctx->save_for_backward({I,
                            polars_I,
                            atom_type_I,
                            waters_I,
                            J,
                            occluders_J,
                            atom_type_J,
                            waters_J,
                            bonded_path_lengths,
                            type_params,
                            global_params});

    return score;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();

    int i = 0;
    auto I = saved[i++];
    auto polars_I = saved[i++];
    auto atom_type_I = saved[i++];
    auto waters_I = saved[i++];
    auto J = saved[i++];
    auto occluders_J = saved[i++];
    auto atom_type_J = saved[i++];
    auto waters_J = saved[i++];
    auto bonded_path_lengths = saved[i++];
    auto type_params = saved[i++];
    auto global_params = saved[i++];

    at::Tensor dV_dI, dV_dJ, dV_dwaters_I, dV_dwaters_J;
    using Int = int64_t;

    auto dTdV = grad_outputs[0];

    TMOL_DISPATCH_FLOATING_DEVICE(
        I.type(), "ScoreOpBackward", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result = ScoreDispatch<DispatchMethod, Dev, Real, Int>::backward(
              TCAST(dTdV),
              TCAST(I),
              TCAST(polars_I),
              TCAST(atom_type_I),
              TCAST(waters_I),
              TCAST(J),
              TCAST(occluders_J),
              TCAST(atom_type_J),
              TCAST(waters_J),
              TCAST(bonded_path_lengths),
              TCAST(type_params),
              TCAST(global_params));

          dV_dI = std::get<0>(result).tensor;
          dV_dJ = std::get<1>(result).tensor;
          dV_dwaters_I = std::get<2>(result).tensor;
          dV_dwaters_J = std::get<3>(result).tensor;
        }));

    return {dV_dI,
            torch::Tensor(),
            torch::Tensor(),
            dV_dwaters_I,
            dV_dJ,
            torch::Tensor(),
            torch::Tensor(),
            dV_dwaters_J,
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor()};
  }
};

template <
    template <
        template <tmol::Device>
        class Dispatch,
        tmol::Device D,
        typename Real,
        typename Int,
        int MAX_WATER>
    class WaterGenDispatch,
    template <tmol::Device>
    class DispatchMethod>
class WaterGen : public Function<WaterGen<WaterGenDispatch, DispatchMethod>> {
 public:
  static Tensor forward(
      AutogradContext* ctx,
      Tensor coords,
      Tensor atom_types,
      Tensor indexed_bonds,
      Tensor indexed_bond_spans,
      Tensor type_params,
      Tensor global_params,
      Tensor sp2_water_tors,
      Tensor sp3_water_tors,
      Tensor ring_water_tors) {
    at::Tensor waters;

    using Int = int64_t;
    constexpr int MAX_WATER = 4;

    TMOL_DISPATCH_FLOATING_DEVICE(
        coords.type(), "watergen_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result =
              WaterGenDispatch<DispatchMethod, Dev, Real, Int, MAX_WATER>::
                  forward(
                      TCAST(coords),
                      TCAST(atom_types),
                      TCAST(indexed_bonds),
                      TCAST(indexed_bond_spans),
                      TCAST(type_params),
                      TCAST(global_params),
                      TCAST(sp2_water_tors),
                      TCAST(sp3_water_tors),
                      TCAST(ring_water_tors));

          waters = result.tensor;
        }));

    ctx->save_for_backward({coords,
                            atom_types,
                            indexed_bonds,
                            indexed_bond_spans,
                            type_params,
                            global_params,
                            sp2_water_tors,
                            sp3_water_tors,
                            ring_water_tors});

    return waters;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();

    int i = 0;

    auto coords = saved[i++];
    auto atom_types = saved[i++];
    auto indexed_bonds = saved[i++];
    auto indexed_bond_spans = saved[i++];
    auto type_params = saved[i++];
    auto global_params = saved[i++];
    auto sp2_water_tors = saved[i++];
    auto sp3_water_tors = saved[i++];
    auto ring_water_tors = saved[i++];

    at::Tensor dT_d_coords;
    using Int = int64_t;

    constexpr int MAX_WATER = 4;
    auto dT_d_waters = grad_outputs[0];

    TMOL_DISPATCH_FLOATING_DEVICE(
        coords.type(), "WaterGenOpBackward", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result =
              WaterGenDispatch<DispatchMethod, Dev, Real, Int, MAX_WATER>::
                  backward(
                      TCAST(dT_d_waters),
                      TCAST(coords),
                      TCAST(atom_types),
                      TCAST(indexed_bonds),
                      TCAST(indexed_bond_spans),
                      TCAST(type_params),
                      TCAST(global_params),
                      TCAST(sp2_water_tors),
                      TCAST(sp3_water_tors),
                      TCAST(ring_water_tors));

          dT_d_coords = result.tensor;
        }));

    return {dT_d_coords,
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor()};
  };
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
Tensor score_op(
    Tensor I,
    Tensor polars_I,
    Tensor atom_type_I,
    Tensor waters_I,
    Tensor J,
    Tensor occluders_J,
    Tensor atom_type_J,
    Tensor waters_J,
    Tensor bonded_path_lengths,
    Tensor type_params,
    Tensor global_params) {
  return ScoreOp<ScoreDispatch, DispatchMethod>::apply(
      I,
      polars_I,
      atom_type_I,
      waters_I,
      J,
      occluders_J,
      atom_type_J,
      waters_J,
      bonded_path_lengths,
      type_params,
      global_params);
}

template <
    template <
        template <tmol::Device>
        class Dispatch,
        tmol::Device D,
        typename Real,
        typename Int,
        int MAX_WATER>
    class WaterGenDispatch,
    template <tmol::Device>
    class DispatchMethod>
Tensor watergen_op(
    Tensor coords,
    Tensor atom_types,
    Tensor indexed_bonds,
    Tensor indexed_bond_spans,
    Tensor type_params,
    Tensor global_params,
    Tensor sp2_water_tors,
    Tensor sp3_water_tors,
    Tensor ring_water_tors) {
  return WaterGen<WaterGenDispatch, DispatchMethod>::apply(
      coords,
      atom_types,
      indexed_bonds,
      indexed_bond_spans,
      type_params,
      global_params,
      sp2_water_tors,
      sp3_water_tors,
      ring_water_tors);
};

// Macro indirection to force TORCH_EXTENSION_NAME macro expansion
// See https://stackoverflow.com/a/3221914
#define TORCH_LIBRARY_(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_(TORCH_EXTENSION_NAME, m) {
  m.def("score_lkball", &score_op<LKBallDispatch, common::AABBDispatch>);
  m.def(
      "watergen_lkball", &watergen_op<GenerateWaters, common::ForallDispatch>);
}
}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
