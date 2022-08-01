#include <tmol/score/common/simple_dispatch.cpu.impl.hh>
#include <tmol/score/common/forall_dispatch.cpu.impl.hh>

#include "dispatch.impl.hh"
#include "gen_waters.impl.hh"
#include "water.hh"

#include <tmol/score/hbond/identification.hh>

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

template struct LKBallDispatch<
    common::AABBDispatch,
    tmol::Device::CPU,
    float,
    int32_t>;
template struct LKBallDispatch<
    common::AABBDispatch,
    tmol::Device::CPU,
    double,
    int32_t>;
template struct LKBallDispatch<
    common::AABBDispatch,
    tmol::Device::CPU,
    float,
    int64_t>;
template struct LKBallDispatch<
    common::AABBDispatch,
    tmol::Device::CPU,
    double,
    int64_t>;

template struct GenerateWaters<
    common::ForallDispatch,
    tmol::Device::CPU,
    float,
    int32_t,
    4>;
template struct GenerateWaters<
    common::ForallDispatch,
    tmol::Device::CPU,
    double,
    int32_t,
    4>;
template struct GenerateWaters<
    common::ForallDispatch,
    tmol::Device::CPU,
    float,
    int64_t,
    4>;
template struct GenerateWaters<
    common::ForallDispatch,
    tmol::Device::CPU,
    double,
    int64_t,
    4>;

#undef def

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
