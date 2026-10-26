#pragma once

#include <tmol/score/common/mps_ctx_builder.hh>
#include <tmol/score/common/pose_total_reduce_ctx.hh>

namespace tmol {
namespace score {
namespace common {

template <typename Real>
using PoseTotalReduceCtxMetal =
    PoseTotalReduceCtxT<tmol::portable::MetalView, Real>;

}  // namespace common
}  // namespace score
}  // namespace tmol
