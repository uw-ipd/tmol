#pragma once

#include <tmol/score/common/pose_total_reduce_ctx.hh>
#include <tmol/score/common/tview_binder.hh>

namespace tmol {
namespace score {
namespace common {

template <typename Real, tmol::Device D>
using PoseTotalReduceCtx =
    PoseTotalReduceCtxT<TViewBinder<D>::template View, Real>;

}  // namespace common
}  // namespace score
}  // namespace tmol
