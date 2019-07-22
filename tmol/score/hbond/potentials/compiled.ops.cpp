#include <torch/script.h>
#include <tmol/utility/autograd.hh>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/simple_dispatch.hh>
#include "dispatch.hh"
#include <tmol/utility/nvtx.hh>

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
Tensor score_op(
    Tensor donor_coords,
    Tensor acceptor_coords,
    Tensor Dinds,
    Tensor H,
    Tensor donor_type,
    Tensor A,
    Tensor B,
    Tensor B0,
    Tensor acceptor_type,
    Tensor pair_params,
    Tensor pair_polynomials,
    Tensor global_params
) {
  using tmol::utility::connect_backward_pass;
  using tmol::utility::StackedSavedGradsBackward;

  at::Tensor score;
  at::Tensor dV_d_don;
  at::Tensor dV_d_acc;

  using Int = int32_t;

  TMOL_DISPATCH_FLOATING_DEVICE(
      donor_coords.type(), "score_op", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        auto result = HBondDispatch<DispatchMethod, Dev, Real, Int>::f(
            TCAST(donor_coords),
            TCAST(acceptor_coords),
            TCAST(Dinds),
            TCAST(H),
            TCAST(donor_type),
            TCAST(A),
            TCAST(B),
            TCAST(B0),
            TCAST(acceptor_type),
	    TCAST(pair_params),
	    TCAST(pair_polynomials),
	    TCAST(global_params)	    
	);

        score = std::get<0>(result).tensor;
	dV_d_don = std::get<1>(result).tensor;
	dV_d_acc = std::get<2>(result).tensor;
      }));

  return connect_backward_pass({donor_coords, acceptor_coords}, score, [&]() {
      return StackedSavedGradsBackward::create({dV_d_don, dV_d_acc});});
};



static auto registry =
    torch::jit::RegisterOperators()
        .op("tmol::score_hbond", &score_op<HBondDispatch, common::AABBDispatch>);


}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
