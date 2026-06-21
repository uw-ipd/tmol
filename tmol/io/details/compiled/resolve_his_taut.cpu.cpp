#include <tmol/score/common/device_operations.cpu.impl.hh>
#include <tmol/io/details/compiled/resolve_his_taut.impl.hh>

namespace tmol {
namespace io {
namespace details {
namespace compiled {

template struct ResolveHisTaut<
    score::common::DeviceOperations,
    tmol::Device::CPU,
    float,
    int>;
template struct ResolveHisTaut<
    score::common::DeviceOperations,
    tmol::Device::CPU,
    double,
    int>;

}  // namespace compiled
}  // namespace details
}  // namespace io
}  // namespace tmol
