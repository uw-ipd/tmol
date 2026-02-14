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

void apsp_op(Tensor stacked_distances, int64_t cutoff) {
  TMOL_DISPATCH_INDEX_DEVICE(
      stacked_distances.options(), "stacked_apsp_op", ([&] {
        using Int = index_t;
        constexpr tmol::Device Dev = device_t;

        AllPairsShortestPathsDispatch<Dev, Int>::f(
            TCAST(stacked_distances), int(cutoff));
      }));
};

// See https://stackoverflow.com/a/3221914

TORCH_LIBRARY(tmol_apsp, m) { m.def("apsp_op", &apsp_op); }

}  // namespace pose
}  // namespace tmol
