#include <tmol/score/common/device_operations.mps.impl.hh>
#include <tmol/score/dunbrack/potentials/dunbrack_pose_score.impl.hh>

namespace tmol {
namespace score {
namespace dunbrack {
namespace potentials {

template struct DunbrackPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::MPS,
    float,
    int>;
template struct DunbrackPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::MPS,
    double,
    int>;

template struct DunbrackRotamerScoreDispatch<
    DeviceOperations,
    tmol::Device::MPS,
    float,
    int>;
template struct DunbrackRotamerScoreDispatch<
    DeviceOperations,
    tmol::Device::MPS,
    double,
    int>;

}  // namespace potentials
}  // namespace dunbrack
}  // namespace score
}  // namespace tmol
