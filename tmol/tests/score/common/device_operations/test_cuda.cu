#include <tmol/score/common/device_operations.cuda.impl.cuh>
#include <tmol/tests/score/common/device_operations/test.impl.hh>

namespace tmol {
namespace tests {
namespace score {
namespace common {
namespace device_operations {

template struct DevOpsTests<tmol::Device::CUDA>;

}  // namespace device_operations
}  // namespace common
}  // namespace score
}  // namespace tests
}  // namespace tmol
