#include <torch/torch.h>
#include <torch/script.h>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include "apsp.hh"
#include "bondsep.hh"

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

Tensor block_bondsep_op(
    Tensor distances,
    Tensor offsets,
    Tensor counts,
    int64_t ports,
    int64_t sentinel) {
  TORCH_CHECK(distances.dim() == 3 && offsets.dim() == 2 && counts.dim() == 2);
  TORCH_CHECK(distances.size(1) == distances.size(2));
  TORCH_CHECK(
      offsets.sizes() == counts.sizes() && counts.size(0) == distances.size(0));
  TORCH_CHECK(
      distances.scalar_type() == torch::kInt32
      && counts.scalar_type() == torch::kInt32);
  TORCH_CHECK(offsets.scalar_type() == torch::kInt64 && ports >= 0);
  TORCH_CHECK(
      distances.device() == offsets.device()
      && distances.device() == counts.device());
  TORCH_CHECK(
      distances.is_contiguous() && offsets.is_contiguous()
      && counts.is_contiguous());
  auto output = torch::empty(
      {counts.size(0), counts.size(1), counts.size(1), ports, ports},
      distances.options());
  TMOL_DISPATCH_INDEX_DEVICE(
      distances.options(), "block_bondsep_op", ([&] {
        gather_block_bondsep<device_t>(
            distances, offsets, counts, output, int32_t(sentinel));
      }));
  return output;
}

TORCH_LIBRARY(tmol_apsp, m) {
  m.def("apsp_op", &apsp_op);
  m.def("block_bondsep_op", &block_bondsep_op);
}

}  // namespace pose
}  // namespace tmol
