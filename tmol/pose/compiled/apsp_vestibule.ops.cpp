#include <torch/torch.h>
#include <torch/script.h>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/simple_dispatch.hh>

#include "apsp.hh"

namespace tmol {
namespace pose {

using torch::Tensor;

void apsp_op(Tensor stacked_distances) {
  // using tmol::utility::connect_backward_pass;

  at::Tensor coords;
  at::Tensor HTs;

  using Int = int32_t;

  TMOL_DISPATCH_INDEX_DEVICE(
      stacked_distances.type(), "stacked_apsp_op", ([&] {
        using Int = index_t;
        constexpr tmol::Device Dev = device_t;

        AllPairsShortestPathsDispatch<Dev, Int>::f(TCAST(stacked_distances));
      }));
};

// static auto registry = torch::jit::RegisterOperators()
//   .op("tmol::apsp_op", &apsp_op);

// Macro indirection to force TORCH_EXTENSION_NAME macro expansion
// See https://stackoverflow.com/a/3221914
#define TORCH_LIBRARY_(ns, m) TORCH_LIBRARY(ns, m)

TORCH_LIBRARY_(TORCH_EXTENSION_NAME, m) { m.def("apsp_op", &apsp_op); }

}  // namespace pose
}  // namespace tmol
