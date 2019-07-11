#include <tmol/score/common/forall_dispatch.cuda.impl.cuh>
#include <tmol/score/common/simple_dispatch.cuda.impl.cuh>

#include "dispatch.impl.hh"
#include "gen_waters.impl.hh"
#include "water.hh"

#include <moderngpu/kernel_compact.hxx>
#include <moderngpu/transform.hxx>
#include <tmol/score/hbond/identification.hh>

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

template struct LKBallDispatch<
    common::ForallDispatch,
    common::AABBDispatch,
    tmol::Device::CUDA,
    float,
    int64_t>;
template struct LKBallDispatch<
    common::ForallDispatch,
    common::AABBDispatch,
    tmol::Device::CUDA,
    double,
    int64_t>;
template struct GenerateWaters<
    common::ForallDispatch,
    tmol::Device::CUDA,
    float,
    int64_t,
    4>;
template struct GenerateWaters<
    common::ForallDispatch,
    tmol::Device::CUDA,
    double,
    int64_t,
    4>;

#undef def

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
