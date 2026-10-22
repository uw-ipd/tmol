#pragma once

// The calls the streaming interaction-graph builder makes into the Metal
// backend: one thread per table entry, through a named entry point. Compiles
// to nothing unless D is MPS, and to nothing at all without the Metal build.

#include <tmol/utility/portable/device.hh>

#include <cstdint>
#include <stdexcept>

#ifdef WITH_MPS
#include <tmol/pack/compiled/streaming_interaction_graph_mps_kernels_source.h>
#include <tmol/score/common/mps_launch.hh>
#endif

namespace tmol {
namespace pack {
namespace compiled {

template <tmol::Device D, typename Ctx>
void ig_mps_launch_if(char const* kernel_name, Ctx& ctx, int64_t n_entries) {
  if constexpr (D == tmol::Device::MPS) {
#ifdef WITH_MPS
    namespace sc = tmol::score::common;
    if (n_entries == 0) return;
    auto args = sc::mps_collect(ctx);
    args.set_scalar(Ctx::n_views, Ctx::scalar_chunk_size, ctx.chunk_size);
    args.set_scalar(
        Ctx::n_views, Ctx::scalar_max_n_molten_blocks, ctx.max_n_molten_blocks);
    int64_t const group = 256;
    sc::mps_launch(
        kStreamingInteractionGraphMetalShaderSource,
        kernel_name,
        args,
        ((n_entries + group - 1) / group) * group,
        group);
#else
    throw std::runtime_error(
        "tmol: an MPS tensor reached the interaction-graph builder, but this "
        "build has no Metal backend");
#endif
  } else {
    (void)kernel_name;
    (void)ctx;
    (void)n_entries;
  }
}

}  // namespace compiled
}  // namespace pack
}  // namespace tmol
