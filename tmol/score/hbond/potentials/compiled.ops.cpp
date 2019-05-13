#include <torch/script.h>
#include <tmol/utility/autograd.hh>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/simple_dispatch.hh>
#include "dispatch.hh"


namespace tmol {
namespace score {
namespace hbond {
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
struct HBScoreOpBackward : public torch::autograd::Function {
  torch::autograd::SavedVariable saved_dv_d_don;
  torch::autograd::SavedVariable saved_dv_d_acc

  void release_variables() override {
    saved_dv_d_don.reset_data();
    saved_dv_d_don.reset_grad_function();
    saved_dv_d_acc.reset_data();
    saved_dv_d_acc.reset_grad_function();
  }

  HBScoreOpBackward(
    torch::autograd::Variable dv_d_don,
    torch::autograd::Variable dv_d_acc,
  )   :
    saved_dv_d_don(dv_d_don, false),
    saved_dv_d_acc(dv_d_acc, false)
  {}

  torch::autograd::variable_list apply(
      torch::autograd::variable_list&& grads) override {
    // Currently, it just returns the variables that were
    // computed on the forward pass, but, I'm keeping this lkball code
    // here for the time being

    auto dv_d_don = saved_dv_d_don.unpack();
    auto dv_d_acc = saved_dv_d_acc.unpack();

    //at::Tensor dV_dI, dV_dJ, dW_dI, dW_dJ;
    //using Int = int64_t;
    //
    //auto dTdV = grads[0];
    //
    //TMOL_DISPATCH_FLOATING_DEVICE(
    //  I.type(), "ScoreOpBackward", ([&] {
    //    using Real = scalar_t;
    //    constexpr tmol::Device Dev = device_t;
    //
    //    auto result = ScoreDispatch<DispatchMethod, Dev, Real, Int>::backward(
    //        TCAST(dTdV),
    //        TCAST(I),
    //        TCAST(atom_type_I),
    //        TCAST(waters_I),
    //        TCAST(J),
    //        TCAST(atom_type_J),
    //        TCAST(waters_J),
    //        TCAST(bonded_path_lengths),
    //        TCAST(type_params),
    //        TCAST(global_params));
    //
    //    dV_dI = std::get<0>(result).tensor;
    //    dV_dJ = std::get<1>(result).tensor;
    //    dW_dI = std::get<2>(result).tensor;
    //    dW_dJ = std::get<3>(result).tensor;
    //  }));
    //
    //
    return {dv_d_don, dv_d_acc};
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
    Tensor donor_coords,
    Tensor acceptor_coords
    Tensor D,
    Tensor H,
    Tensor donor_type,
    Tensor A,
    Tensor B,
    Tensor B0,
    Tensor acceptor_type,
    Tensor donor_weight,
    Tensor acceptor_weight,
    Tensor acceptor_hybridization,
    Tensor AHdist_range,
    Tensor AHdist_bound,
    Tensor AHdist_coeffs,
    Tensor cosBAH_range,
    Tensor cosBAH_bound,
    Tensor cosBAH_coeffs,
    Tensor cosAHD_range,
    Tensor cosAHD_bound,
    Tensor cosAHD_coeffs
) {
  using tmol::utility::connect_backward_pass;
  using tmol::utility::SavedGradsBackward;

  at::Tensor score;
  at::Tensor dV_d_don;
  at::Tensor dV_d_acc;

  using Int = int64_t;

  TMOL_DISPATCH_FLOATING_DEVICE(
      I.type(), "score_op", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        auto result = HBondDispatch<DispatchMethod, Dev, Real, Int>::forward(
            TCAST(donor_coords),
            TCAST(acceptor_coords),
            TCAST(D),
            TCAST(H),
            TCAST(donor_type),
            TCAST(A),
            TCAST(B),
            TCAST(B0),
            TCAST(acceptor_type),
            TCAST(donor_weight),
            TCAST(acceptor_weight),
            TCAST(acceptor_hybridization),
            TCAST(AHdist_range),
            TCAST(AHdist_bound),
            TCAST(AHdist_coeffs),
            TCAST(cosBAH_range),
            TCAST(cosBAH_bound),
            TCAST(cosBAH_coeffs),
            TCAST(cosAHD_range),
            TCAST(cosAHD_bound),
            TCAST(cosAHD_coeffs));

        score = std::get<0>(result).tensor;
	dv_d_don = std::get<1>(result).tensor;
	dv_d_acc = std::get<2>(result).tensor;
      }));

  return connect_backward_pass({dv_d_don, dv_d_acc}, score, [&]() {
    return std::shared_ptr<HBScoreOpBackward<HBondDispatch, common::AABBDispatch>>(
        new ScoreOpBackward<HBondDispatch, common::AABBDispatch>(
	  dv_d_don, dv_d_acc
	),
        torch::autograd::deleteFunction);
  });
};



static auto registry =
    torch::jit::RegisterOperators()
        .op("tmol::score_hbond", &score_op<HBondDispatch, common::AABBDispatch>);


}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
