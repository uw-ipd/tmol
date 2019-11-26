#include <torch/script.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/autograd/function.h> // ??
#include <torch/csrc/autograd/saved_variable.h> // ??
#include <torch/types.h>

#include <tmol/utility/autograd.hh>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/forall_dispatch.hh>

#include "annealer.hh"

namespace tmol {
namespace pack {
namespace compiled {

using torch::Tensor;

std::vector< Tensor >
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
    energy1b.type(), "pack_anneal", ([&] {
      constexpr tmol::Device Dev = device_t;

      auto result = AnnealerDispatch<Dev>::forward(
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

  std::vector< torch::Tensor > result({scores, rotamer_assignments});
  return result;
}

static auto registry =
  torch::jit::RegisterOperators()
  .op("tmol::pack_anneal", &anneal);

}
}
}
