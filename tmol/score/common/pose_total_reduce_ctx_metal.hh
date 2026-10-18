#pragma once

#include <tmol/score/common/mps_ctx_builder.hh>
#include <tmol/score/common/pose_total_reduce_ctx.hh>

namespace tmol {
namespace score {
namespace common {

template <typename Real>
using PoseTotalReduceCtxMetal =
    PoseTotalReduceCtxT<tmol::portable::MetalView, Real>;

template <typename Ctx>
Ctx pose_total_reduce_build_metal_ctx(
    device void* const* ptrs, device const int64_t* meta) {
  Ctx ctx = mps_build_ctx<Ctx>(ptrs, meta);
  device const int64_t* scalars = mps_scalars<Ctx>(meta);
  ctx.n_cells = int(scalars[Ctx::scalar_n_cells]);
  return ctx;
}

}  // namespace common
}  // namespace score
}  // namespace tmol
