#pragma once

// Rebuilds a kernel context's views from the buffers bound to a Metal entry.
//
// Applied by the context's visit_views, so it sees the fields in the same
// order the host bound them and in the same order it wrote their shapes into
// the metadata buffer -- which is why neither side ever repeats the field
// list.

#include <tmol/score/common/mps_layout.hh>
#include <tmol/utility/portable/metal_view.hh>

#if !TMOL_PORTABLE_METAL
#error "mps_ctx_builder.hh is for Metal translation units only"
#endif

namespace tmol {
namespace score {
namespace common {

struct MpsCtxBuilder {
  device void* const* ptrs;
  device const int64_t* meta;
  thread int* next;

  // The element type and rank of each view come from the view's own type,
  // which is why the host can bind everything as untyped buffers and why this
  // has to be a template.
  template <typename T, int N>
  void operator()(thread tmol::portable::MetalView<T, N>& view) const {
    int const i = (*next)++;
    view.data_ = reinterpret_cast<device T*>(ptrs[i]);
    view.sizes_ = meta + i * TMOL_VIEW_META_STRIDE;
    view.strides_ = view.sizes_ + TMOL_MAX_VIEW_RANK;
  }
};

// Build a context's views from the bound buffers. Scalars are the caller's
// business, since each kernel family names its own.
template <typename Ctx>
Ctx mps_build_ctx(device void* const* ptrs, device const int64_t* meta) {
  Ctx ctx;
  int next = 0;
  ctx.visit_views(MpsCtxBuilder{ptrs, meta, &next});
  return ctx;
}

// Start of a context's scalar slots in the metadata buffer.
template <typename Ctx>
device const int64_t* mps_scalars(device const int64_t* meta) {
  return meta + Ctx::n_views * TMOL_VIEW_META_STRIDE;
}

}  // namespace common
}  // namespace score
}  // namespace tmol
