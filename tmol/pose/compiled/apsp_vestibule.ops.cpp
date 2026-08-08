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
  // MPS (Phase 1): run on a CPU copy and write back.
  // Direct data_ptr() writes to MPS unified memory are invisible to Metal
  // pipelines, causing coherency issues when Python later uses the tensor
  // through Metal ops.  A CPU round-trip is correct and inexpensive.
  bool is_mps = stacked_distances.device().is_mps();
  Tensor work = is_mps ? stacked_distances.cpu().contiguous()
                       : stacked_distances;

  TMOL_DISPATCH_INDEX_DEVICE(
      work.options(), "stacked_apsp_op", ([&] {
        using Int = index_t;
        constexpr tmol::Device Dev = device_t;

        AllPairsShortestPathsDispatch<Dev, Int>::f(
            TCAST(work), int(cutoff));
      }));

  if (is_mps) {
    stacked_distances.copy_(work);
  }
};

// See https://stackoverflow.com/a/3221914

TORCH_LIBRARY(tmol_apsp, m) { m.def("apsp_op", &apsp_op); }

}  // namespace pose
}  // namespace tmol
