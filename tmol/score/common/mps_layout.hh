#pragma once

// Layout of the metadata buffer shared by a Metal kernel and its host launch.
//
// A kernel body indexes its tensors through views, and a view is a data
// pointer plus its sizes and strides. The data pointers arrive as bound
// buffers -- one per view, in the order the context's visit_views enumerates
// -- while the shapes and the launch-shape scalars travel together in one
// int64 buffer described here. Both sides include this header, so the layout
// is stated once.

namespace tmol {
namespace score {
namespace common {

// Highest rank any ported view has. Every view gets a fixed-size slot so its
// metadata can be addressed arithmetically rather than by summing the ranks
// before it.
enum { TMOL_MAX_VIEW_RANK = 5 };

// Per view: TMOL_MAX_VIEW_RANK sizes followed by TMOL_MAX_VIEW_RANK strides.
enum { TMOL_VIEW_META_STRIDE = 2 * TMOL_MAX_VIEW_RANK };

// Scalars follow the per-view slots, as one int64 each. A kernel decides what
// its own scalar slots mean; see the per-term layout headers.
//
// Real-valued scalars travel as the bit pattern of a float, because the
// buffer is typed int64 and Metal has no double anyway. tmol_pack_float and
// tmol_unpack_float below are the only sanctioned way to read and write them.
// The annealer carries eleven: the launch shape, the schedule and the phase.
enum { TMOL_MAX_SCALARS = 16 };

// Buffer index 0 carries the metadata; the view buffers start at 1.
enum { TMOL_META_BUFFER_INDEX = 0, TMOL_FIRST_VIEW_BUFFER_INDEX = 1 };

}  // namespace common
}  // namespace score
}  // namespace tmol
