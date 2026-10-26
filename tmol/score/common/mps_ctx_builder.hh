#pragma once

// Rebuilds a kernel context's views from the buffers bound to a Metal entry.
//
// Applied by the context's visit_views, so it sees the fields in the same
// order the host bound them and in the same order it wrote their shapes into
// the metadata buffer -- which is why neither side ever repeats the field
// list.

#include <tmol/score/common/mps_float_scalar.hh>
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

// Reads a context's scalars from the metadata slots after the views, in
// visit_scalars order. Floats travel as bit patterns; see mps_float_scalar.hh.
struct MpsScalarReader {
  device const int64_t* slots;
  thread int* next;

  void operator()(thread int& value) const { value = int(slots[(*next)++]); }
  void operator()(thread bool& value) const { value = slots[(*next)++] != 0; }
  void operator()(thread float& value) const {
    value = tmol_unpack_float(slots[(*next)++]);
  }
};

// Build a context's views and scalars from the bound buffers and metadata.
template <typename Ctx>
Ctx mps_build_ctx(device void* const* ptrs, device const int64_t* meta) {
  Ctx ctx;
  int next = 0;
  ctx.visit_views(MpsCtxBuilder{ptrs, meta, &next});
  int next_scalar = 0;
  ctx.visit_scalars(
      MpsScalarReader{
          meta + Ctx::n_views * TMOL_VIEW_META_STRIDE, &next_scalar});
  return ctx;
}

}  // namespace common
}  // namespace score
}  // namespace tmol
