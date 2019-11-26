#include <torch/script.h>
#include <tmol/utility/autograd.hh>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/forall_dispatch.hh>

#include "annealer.hh"

namespace tmol {
namespace pack {
namespace compiled {


template < template <tmol::Device> class DispatchMethod >
std::tuple<Tensor, Tensor>
anneal(
  Tensor nrotamers_for_res,
  Tensor oneb_offsets,
  Tensor res_for_rot,
  Tensor nenergies,
  Tensor twob_offsets,
  Tensor energy1b,
  Tensor energy2b
)
{
  nvtx_range_push("pack_anneal");
  at::Tensor scores;
  at::Tensor rotamer_assignments;

  TMOL_DISPATCH_FLOATING_DEVICE(
    coords.type(), "pack_anneal", ([&] {
      using Real = scalar_t;
      constexptr tmol::Device Dev = device_t;

      auto result = AnnealerDispatch<DispatchMethod, Dev, Real, Int>::forward(
	TCAST(nrotamers_for_res),
	TCAST(oneb_offsets),
	TCAST(res_for_rot),
	TCAST(nenergies),
	TCAST(twob_offsets),
	TCAST(energy1b),
	TCAST(energy2b));
      scores = std::get<0>(result).tensor;
      rotamer_assignments = std::get<0>(result).tensor;
      }));
  return {scores, rotamer_assignments};
}

static auto registry =
  torch::jit::RegisterOperators()
  .op("tmol::pack_anneal", &anneal<common::ForallDispatch>);

}
}
}
