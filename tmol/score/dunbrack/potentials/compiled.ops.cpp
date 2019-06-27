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
namespace dunbrack {
namespace potentials {

using torch::Tensor;

template < 
  template <tmol::Device> class DispatchMethod,
  int MAXBB,
  int MAXCHI>
Tensor dun_op(
      Tensor coords,
      Tensor rotameric_tables,
      Tensor rotameric_table_params,
      Tensor semirotameric_tables,
      Tensor semirotameric_table_params,
      Tensor residue_params,
      Tensor residue_lookup_params
) {
  using tmol::utility::connect_backward_pass;
  using tmol::utility::SavedGradsBackward;
  nvtx_range_push("dun_op");

  at::Tensor score;
  at::Tensor dScore;

  using Int = int32_t;

  TMOL_DISPATCH_FLOATING_DEVICE(
      coords.type(), "dun_op", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        auto result = DunbrackDispatch<DispatchMethod, Dev, Real, Int, MAXBB, MAXCHI>::f(
            TCAST(coords),
            TCAST(rotameric_tables),
            TCAST(rotameric_table_params),
            TCAST(semirotameric_tables),
            TCAST(semirotameric_table_params),
            TCAST(residue_params),
            TCAST(residue_lookup_params));

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
        .op("tmol::score_dun", &dun_op<common::ForallDispatch, 2, 4>);

}  // namespace potentials
}  // namespace dunbrack
}  // namespace score
}  // namespace tmol
