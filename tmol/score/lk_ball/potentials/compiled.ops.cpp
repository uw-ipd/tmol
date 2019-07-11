#include <torch/script.h>
#include <tmol/utility/autograd.hh>

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
struct ScoreOpBackward : public torch::autograd::Function {
  torch::autograd::SavedVariable saved_I;
  torch::autograd::SavedVariable saved_polars_I;
  torch::autograd::SavedVariable saved_atom_type_I;
  torch::autograd::SavedVariable saved_waters_I;
  torch::autograd::SavedVariable saved_J;
  torch::autograd::SavedVariable saved_occulders_J;
  torch::autograd::SavedVariable saved_atom_type_J;
  torch::autograd::SavedVariable saved_waters_J;
  torch::autograd::SavedVariable saved_bonded_path_lengths;
  torch::autograd::SavedVariable saved_type_params;
  torch::autograd::SavedVariable saved_global_params;

  void release_variables() override {
    saved_I.reset_data();
    saved_I.reset_grad_function();
    saved_polars_I.reset_data();
    saved_polars_I.reset_grad_function();
    saved_atom_type_I.reset_data();
    saved_atom_type_I.reset_grad_function();
    saved_waters_I.reset_data();
    saved_waters_I.reset_grad_function();
    saved_J.reset_data();
    saved_J.reset_grad_function();
    saved_occulders_J.reset_data();
    saved_occulders_J.reset_grad_function();
    saved_atom_type_J.reset_data();
    saved_atom_type_J.reset_grad_function();
    saved_waters_J.reset_data();
    saved_waters_J.reset_grad_function();
    saved_bonded_path_lengths.reset_data();
    saved_bonded_path_lengths.reset_grad_function();
    saved_type_params.reset_data();
    saved_type_params.reset_grad_function();
    saved_global_params.reset_data();
    saved_global_params.reset_grad_function();
  }

  ScoreOpBackward(
    torch::autograd::Variable I,
    torch::autograd::Variable polars_I,
    torch::autograd::Variable atom_type_I,
    torch::autograd::Variable waters_I,
    torch::autograd::Variable J,
    torch::autograd::Variable occulders_J,
    torch::autograd::Variable atom_type_J,
    torch::autograd::Variable waters_J,
    torch::autograd::Variable bonded_path_lengths,
    torch::autograd::Variable type_params,
    torch::autograd::Variable global_params
  )   : 
    saved_I(I, false), 
    saved_polars_I(polars_I, false), 
    saved_atom_type_I(atom_type_I, false), 
    saved_waters_I(waters_I, false), 
    saved_J(J, false), 
    saved_occulders_J(occulders_J, false), 
    saved_atom_type_J(atom_type_J, false), 
    saved_waters_J(waters_J, false), 
    saved_bonded_path_lengths(bonded_path_lengths, false),
    saved_type_params(type_params, false),
    saved_global_params(global_params, false) {
  }

  torch::autograd::variable_list apply(
      torch::autograd::variable_list&& grads) override {
    auto I = saved_I.unpack();
    auto polars_I = saved_polars_I.unpack();
    auto atom_type_I = saved_atom_type_I.unpack();
    auto waters_I = saved_waters_I.unpack();
    auto J = saved_J.unpack();
    auto occulders_J = saved_occulders_J.unpack();
    auto atom_type_J = saved_atom_type_J.unpack();
    auto waters_J = saved_waters_J.unpack();
    auto bonded_path_lengths = saved_bonded_path_lengths.unpack();
    auto type_params = saved_type_params.unpack();
    auto global_params = saved_global_params.unpack();

    at::Tensor dV_dI, dV_dJ, dW_dI, dW_dJ;
    using Int = int64_t;

    auto dTdV = grads[0];

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
            TCAST(occulders_J),
            TCAST(atom_type_J),
            TCAST(waters_J),
            TCAST(bonded_path_lengths),
            TCAST(type_params),
            TCAST(global_params));

        dV_dI = std::get<0>(result).tensor;
        dV_dJ = std::get<1>(result).tensor;
        dW_dI = std::get<2>(result).tensor;
        dW_dJ = std::get<3>(result).tensor;
      }));


    return {dV_dI,dV_dJ,dW_dI,dW_dJ};
  }
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

  return connect_backward_pass({I, J, waters_I, waters_J}, score, [&]() {
    return std::shared_ptr<ScoreOpBackward<LKBallDispatch, common::AABBDispatch>>(
        new ScoreOpBackward<LKBallDispatch, common::AABBDispatch>( 
            I, polars_I, atom_type_I, waters_I, 
            J, occluders_J, atom_type_J, waters_J, 
            bonded_path_lengths,
            type_params,
            global_params), 
        torch::autograd::deleteFunction);
  });
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
struct WaterGenOpBackward : public torch::autograd::Function {
  torch::autograd::SavedVariable saved_coords;
  torch::autograd::SavedVariable saved_atom_types;
  torch::autograd::SavedVariable saved_indexed_bonds;
  torch::autograd::SavedVariable saved_indexed_bond_spans;
  torch::autograd::SavedVariable saved_type_params;
  torch::autograd::SavedVariable saved_global_params;
  torch::autograd::SavedVariable saved_sp2_water_tors;
  torch::autograd::SavedVariable saved_sp3_water_tors;
  torch::autograd::SavedVariable saved_ring_water_tors;

  void release_variables() override {
    saved_coords.reset_data();
    saved_coords.reset_grad_function();
    saved_atom_types.reset_data();
    saved_atom_types.reset_grad_function();
    saved_indexed_bonds.reset_data();
    saved_indexed_bonds.reset_grad_function();
    saved_indexed_bond_spans.reset_data();
    saved_indexed_bond_spans.reset_grad_function();
    saved_type_params.reset_data();
    saved_type_params.reset_grad_function();
    saved_global_params.reset_data();
    saved_global_params.reset_grad_function();
    saved_sp2_water_tors.reset_data();
    saved_sp2_water_tors.reset_grad_function();
    saved_sp3_water_tors.reset_data();
    saved_sp3_water_tors.reset_grad_function();
    saved_ring_water_tors.reset_data();
    saved_ring_water_tors.reset_grad_function();
  }

  WaterGenOpBackward(
    torch::autograd::Variable coords,
    torch::autograd::Variable atom_types,
    torch::autograd::Variable indexed_bonds,
    torch::autograd::Variable indexed_bond_spans,
    torch::autograd::Variable type_params,
    torch::autograd::Variable global_params,
    torch::autograd::Variable sp2_water_tors,
    torch::autograd::Variable sp3_water_tors,
    torch::autograd::Variable ring_water_tors
  )   : 
    saved_coords(coords, false), 
    saved_atom_types(atom_types, false), 
    saved_indexed_bonds(indexed_bonds, false), 
    saved_indexed_bond_spans(indexed_bond_spans, false), 
    saved_type_params(type_params, false), 
    saved_global_params(global_params, false), 
    saved_sp2_water_tors(sp2_water_tors, false),
    saved_sp3_water_tors(sp3_water_tors, false),
    saved_ring_water_tors(ring_water_tors, false) {

  }

  torch::autograd::variable_list apply(
      torch::autograd::variable_list&& grads) override {
    auto coords = saved_coords.unpack();
    auto atom_types = saved_atom_types.unpack();
    auto indexed_bonds = saved_indexed_bonds.unpack();
    auto indexed_bond_spans = saved_indexed_bond_spans.unpack();
    auto type_params = saved_type_params.unpack();
    auto global_params = saved_global_params.unpack();
    auto sp2_water_tors = saved_sp2_water_tors.unpack();
    auto sp3_water_tors = saved_sp3_water_tors.unpack();
    auto ring_water_tors = saved_ring_water_tors.unpack();

    at::Tensor derivs;
    using Int = int64_t;

    constexpr int MAX_WATER = 4;
    auto dTdV = grads[0];

    TMOL_DISPATCH_FLOATING_DEVICE(
      coords.type(), "WaterGenOpBackward", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        auto result = WaterGenDispatch<DispatchMethod, Dev, Real, Int, MAX_WATER>::backward(
            TCAST(dTdV),
            TCAST(coords),
            TCAST(atom_types),
            TCAST(indexed_bonds),
            TCAST(indexed_bond_spans),
            TCAST(type_params),
            TCAST(global_params),
            TCAST(sp2_water_tors),
            TCAST(sp3_water_tors),
            TCAST(ring_water_tors));

        derivs = result.tensor;
      }));

    return {derivs};
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
      coords.type(), "watergen_op", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        auto result = WaterGenDispatch<DispatchMethod, Dev, Real, Int, MAX_WATER>::forward(
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
    return std::shared_ptr<WaterGenOpBackward<GenerateWaters, common::ForallDispatch>>(
        new WaterGenOpBackward<GenerateWaters, common::ForallDispatch>( 
            coords, 
            atom_types, 
            indexed_bonds,
            indexed_bond_spans,
            type_params,
            global_params,
            sp2_water_tors,
            sp3_water_tors,
            ring_water_tors),
        torch::autograd::deleteFunction);
  });
};

static auto registry =
    torch::jit::RegisterOperators()
        .op("tmol::score_lkball", &score_op<LKBallDispatch, common::AABBDispatch>)
        .op("tmol::watergen_lkball", &watergen_op<GenerateWaters, common::ForallDispatch>);

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
