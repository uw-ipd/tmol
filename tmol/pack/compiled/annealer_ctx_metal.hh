#pragma once

// The Metal instantiation of the annealer context.
//
// A Metal translation unit cannot include the torch tensor layer, so it binds
// the same field list to MetalView instead of TView. See annealer_ctx.hh.

#include <tmol/pack/compiled/annealer_ctx.hh>
#include <tmol/score/common/mps_ctx_builder.hh>
#include <tmol/score/common/mps_float_scalar.hh>
#include <tmol/utility/portable/metal_view.hh>

namespace tmol {
namespace pack {
namespace compiled {

template <typename Real, typename Int>
using AnnealerCtxMetal =
    AnnealerCtxT<tmol::portable::MetalView, Real, Int>;

// Build a context from the bound buffers, then unpack the launch scalars that
// follow the view metadata. They are carried in the metadata buffer rather
// than as buffers of their own so the view count stays inside Metal's
// binding limit.
template <typename Ctx>
Ctx build_annealer_metal_ctx(
    device void* const* ptrs, device const int64_t* meta) {
  Ctx ctx = tmol::score::common::mps_build_ctx<Ctx>(ptrs, meta);
  device const int64_t* s = tmol::score::common::mps_scalars<Ctx>(meta);
  ctx.chunk_size = int(s[Ctx::scalar_chunk_size]);
  ctx.max_n_res = int(s[Ctx::scalar_max_n_res]);
  ctx.n_traj = int(s[Ctx::scalar_n_traj]);
  ctx.n_outer_iterations = int(s[Ctx::scalar_n_outer_iterations]);
  ctx.n_inner_iterations = int(s[Ctx::scalar_n_inner_iterations]);
  ctx.seed_lo = int(s[Ctx::scalar_seed_lo]);
  ctx.seed_hi = int(s[Ctx::scalar_seed_hi]);
  ctx.phase = int(s[Ctx::scalar_phase]);
  // Temperatures travel as scaled integers: the metadata buffer is int64 and
  // mps_float_scalar is the project's convention for getting a float through
  // it without a second buffer.
  ctx.high_temp = tmol::score::common::tmol_unpack_float(s[Ctx::scalar_high_temp]);
  ctx.low_temp = tmol::score::common::tmol_unpack_float(s[Ctx::scalar_low_temp]);
  ctx.cooling_factor =
      tmol::score::common::tmol_unpack_float(s[Ctx::scalar_cooling_factor]);
  return ctx;
}

}  // namespace compiled
}  // namespace pack
}  // namespace tmol
