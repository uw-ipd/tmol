#include <tmol/score/common/device_operations.cpu.impl.hh>
#include <tmol/score/ljlk/potentials/ljlk_pose_score.impl.hh>
//#include <tmol/score/common/forall_dispatch.cpu.impl.hh>

#include <tmol/pack/sim_anneal/compiled/annealer.hh>

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template struct LJLKPoseScoreDispatch<
    // ForallDispatch,
    DeviceOperations,
    tmol::Device::CPU,
    float,
    int>;
template struct LJLKPoseScoreDispatch<
    // ForallDispatch,
    DeviceOperations,
    tmol::Device::CPU,
    double,
    int>;

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
