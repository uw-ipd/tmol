#pragma once

// Host side of the Metal backend.
//
// Compiles a shader library once per source, binds a context's views to
// buffer slots in its visit_views order, and encodes a dispatch onto torch's
// MPS command buffer.
//
// Everything here goes through torch's stable ABI (shim_mps.h). That matters:
// torch/mps.h and all of at::mps:: are compiled out under
// TORCH_TARGET_VERSION, so the usual route -- bit-casting a tensor's storage
// pointer to an id<MTLBuffer> and encoding onto
// torch::mps::get_command_buffer() -- is not available to a module built the
// way tmol ships. The stable shim does the buffer extraction and the stream
// ordering itself.

#include <tmol/score/common/mps_float_scalar.hh>
#include <tmol/score/common/mps_layout.hh>
#include <tmol/score/common/workgroup_lane.hh>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/native.hh>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace tmol {
namespace score {
namespace common {

// One view, reduced to what a launch needs: where its buffer starts and how
// many bytes of it the kernel may touch.
struct MpsBuffer {
  void* data = nullptr;
  int64_t n_bytes = 0;
};

// Everything the host hands the GPU for one launch.
struct MpsArgs {
  std::vector<MpsBuffer> buffers;
  std::vector<int64_t> meta;
};

namespace detail {

// Records one view's buffer and writes its shape into the metadata slot.
//
// Applied by a context's visit_views, so the slot order matches the order the
// Metal entry rebuilds the views in -- the field list is never repeated.
//
// The shape comes from the view rather than from a tensor on purpose. A
// view's rank is not always its tensor's rank (a TView<Vec<Real,3>,1> is
// backed by a 2-D tensor) and its strides count elements of the view's own
// type, which is what the kernel indexes by.
struct MpsViewRecorder {
  MpsArgs* args;
  int next = 0;

  // TView's rank parameter is a size_t, unlike MetalView's int, so the
  // recorder has to spell it the same way to deduce.
  template <typename T, size_t N, tmol::Device D, tmol::PtrTag P>
  void operator()(TView<T, N, D, P> const& view) {
    static_assert(
        N <= size_t(TMOL_MAX_VIEW_RANK),
        "a view's rank exceeds its metadata slot; raise TMOL_MAX_VIEW_RANK");
    int const i = next++;
    int64_t* sizes = args->meta.data() + size_t(i) * TMOL_VIEW_META_STRIDE;
    int64_t* strides = sizes + TMOL_MAX_VIEW_RANK;

    // Highest element index the kernel can reach, so the wrapper tensor
    // covers the whole region the view spans rather than assuming it is
    // contiguous.
    int64_t extent = 1;
    for (size_t d = 0; d < N; ++d) {
      sizes[d] = view.size(int64_t(d));
      strides[d] = view.stride(int64_t(d));
      if (sizes[d] > 0) {
        extent += (sizes[d] - 1) * strides[d];
      }
    }

    args->buffers[i] = MpsBuffer{
        const_cast<void*>(static_cast<void const*>(view.data())),
        view.data() == nullptr ? 0 : extent * int64_t(sizeof(T))};
  }
};

// Records a context's scalars into the slots after the views, in
// visit_scalars order. Floating-point values travel as float bit patterns.
struct MpsScalarRecorder {
  MpsArgs* args;
  size_t next;

  template <typename T>
  void operator()(T const& value) {
    args->meta.at(next++) = std::is_floating_point_v<T>
                                ? tmol_pack_float(float(value))
                                : int64_t(value);
  }
};

}  // namespace detail

// Collect the launch arguments for a context.
template <typename Ctx>
MpsArgs mps_collect(Ctx& ctx) {
  MpsArgs args;
  args.buffers.assign(Ctx::n_views, MpsBuffer{});
  args.meta.assign(
      size_t(Ctx::n_views) * TMOL_VIEW_META_STRIDE + TMOL_MAX_SCALARS, 0);

  detail::MpsViewRecorder recorder{&args};
  ctx.visit_views(recorder);
  ctx.visit_scalars(
      detail::MpsScalarRecorder{
          &args, size_t(Ctx::n_views) * TMOL_VIEW_META_STRIDE});
  return args;
}

// Launch a named entry point from a shader library.
//
// shader_source is the flattened Metal source; it is compiled once and cached
// by address, so pass the same generated constant every time. n_items threads
// are dispatched in groups of threadgroup_size -- for a workgroup kernel that
// is n_workgroups * threadgroup_size, for an element-wise kernel it is the
// element count.
void mps_launch(
    char const* shader_source,
    char const* kernel_name,
    MpsArgs const& args,
    int64_t n_items,
    int64_t threadgroup_size);

// The embedded Metal source for a shader, or null in a build without the
// Metal backend, where the generated source header does not exist.
#ifdef WITH_MPS
#define TMOL_MPS_SOURCE(symbol) (symbol)
#else
#define TMOL_MPS_SOURCE(symbol) static_cast<char const*>(nullptr)
#endif

// Launch a named entry point over a context, or do nothing unless D is MPS.
// The generic dispatch bodies are compiled for every backend, so this is the
// one call they make into Metal. A null source means the build has no Metal
// backend.
//
// workgroups: n threadgroups. threads: n threads, rounded up to whole groups
// (the kernel drops the tail).
template <tmol::Device D, typename Ctx>
void mps_launch_if(
    char const* source,
    char const* kernel,
    Ctx& ctx,
    int64_t n,
    bool one_group_per_item,
    int64_t group = TMOL_MPS_THREADGROUP_SIZE) {
  if constexpr (D == tmol::Device::MPS) {
    if (source == nullptr) {
      throw std::runtime_error(
          "tmol: an MPS tensor reached a Metal kernel, but this build has no "
          "Metal backend");
    }
    if (n <= 0) {
      return;
    }
    int64_t const n_items =
        one_group_per_item ? n * group : ((n + group - 1) / group) * group;
    mps_launch(source, kernel, mps_collect(ctx), n_items, group);
  }
}

}  // namespace common
}  // namespace score
}  // namespace tmol
