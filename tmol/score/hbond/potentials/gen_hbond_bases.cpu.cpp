#include <tmol/score/common/device_operations.cpu.impl.hh>
#include <tmol/score/hbond/potentials/gen_hbond_bases.impl.hh>

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

template struct GenerateHBondBases<
    common::DeviceOperations,
    tmol::Device::CPU,
    float,
    int>;
template struct GenerateHBondBases<
    common::DeviceOperations,
    tmol::Device::CPU,
    double,
    int>;

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
