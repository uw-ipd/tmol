#include <tmol/score/common/device_operations.cuda.impl.cuh>
#include <tmol/score/hbond/potentials/gen_hbond_bases.impl.hh>

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

template struct GenerateHBondBases<
    common::DeviceOperations,
    tmol::Device::CUDA,
    float,
    int>;
template struct GenerateHBondBases<
    common::DeviceOperations,
    tmol::Device::CUDA,
    double,
    int>;

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
